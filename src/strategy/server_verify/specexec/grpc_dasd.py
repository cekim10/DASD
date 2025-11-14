import asyncio
import os
from collections import defaultdict

import grpc.aio

import log
from specedge_grpc import specedge_pb2
from specedge_grpc import specedge_pb2_grpc
import numpy as np


def _as_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ("1", "true", "yes", "on")
    return default

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _pack_all_ones_bitmap(length: int) -> bytes:
    """Return W-bit bitmap (all 1s) packed LSB-first per byte."""
    if length <= 0:
        return b""
    full, rem = divmod(length, 8)
    out = bytearray([0xFF] * full)
    if rem:
        out.append((1 << rem) - 1)
    return bytes(out)

def _pack_bitmap_from_bits(bits: list[int]) -> bytes:
    """Pack a list of 0/1 ints into bytes, LSB-first per byte (same layout as _pack_all_ones_bitmap)."""
    length = len(bits)
    if length <= 0:
        return b""
    full, rem = divmod(length, 8)
    out = bytearray()
    idx = 0
    for _ in range(full):
        byte = 0
        for k in range(8):
            if bits[idx]:
                byte |= (1 << k)
            idx += 1
        out.append(byte)
    if rem:
        byte = 0
        for k in range(rem):
            if bits[idx]:
                byte |= (1 << k)
            idx += 1
        out.append(byte)
    return bytes(out)

def _decode_qlogp_i8(data: bytes) -> list[int]:
    if not data:
        return []
    arr = np.frombuffer(data, dtype=np.int8)
    return arr.tolist()


class SpecExecDasdServer(specedge_pb2_grpc.SpecEdgeServiceServicer):
    """
    Minimal DASD V2 server so you can smoke-test the wire path:
      * Accepts all tokens in the LAST bundle of the request
      * last_accepted_pos += bundle.len
      * next_credit updated via simple AIMD using env vars set in batch_server.py
      * hint_mask empty for now
    Replace the “accept all” area with real verification once the round trip works.
    """

    def __init__(
        self,
        shutdown_event: asyncio.Event | None = None,
    ) -> None:
        self._logger = log.get_logger()
        self.shutdown_event = shutdown_event

        # read env set by batch_sever.py (already exported in batch_server.py)
        self._dasd_enable = _as_bool(os.environ.get("DASD_ENABLE", "0"))
        self._tick_ms = _get_env_int("DASD_TICK_MS", 4)

        self._aimd_r_target = _get_env_float("DASD_AIMD_R_TARGET", 0.85)
        self._aimd_inc = _get_env_int("DASD_AIMD_INC", 2)
        self._aimd_dec_factor = _get_env_float("DASD_AIMD_DEC_FACTOR", 0.5)
        self._aimd_min_credit = _get_env_int("DASD_AIMD_MIN_CREDIT", 4)
        self._aimd_max_credit = _get_env_int("DASD_AIMD_MAX_CREDIT", 64)

        self._pas_enable = _as_bool(os.environ.get("DASD_PAS_ENABLE", "0"))
        self._pas_top_m = _get_env_int("DASD_PAS_TOP_M", 8)
        self._pas_broadcast_every = _get_env_int("DASD_PAS_BROADCAST_EVERY", 2)
        self._pas_ttl = _get_env_int("DASD_PAS_TTL", 64)

        self._qlogp_i8_thresh = _get_env_int("DASD_QLOGP_I8_THRESH", -8)
        self._qlogp_i8_min_run = _get_env_int("DASD_QLOGP_I8_MIN_RUN", 1)

        # Per-sequence tracking
        # last accepted absolute position
        self._last_pos: dict[str, int] = defaultdict(int)
        # simple credit state per sequence
        self._credit: dict[str, int] = defaultdict(lambda: self._aimd_min_credit)

        self._logger.info(
            "DASD V2 server init: enable=%s tick_ms=%s aimd(r*=%.2f, +%d, x%.2f, min=%d, max=%d) "
            "pas(enable=%s, top_m=%d, every=%d, ttl=%d) qlogp(thresh=%d,min_run=%d)",
            self._dasd_enable,
            self._tick_ms,
            self._aimd_r_target,
            self._aimd_inc,
            self._aimd_dec_factor,
            self._aimd_min_credit,
            self._aimd_max_credit,
            self._pas_enable,
            self._pas_top_m,
            self._pas_broadcast_every,
            self._pas_ttl,
            self._qlogp_i8_thresh,
            self._qlogp_i8_min_run,
        )

    async def ValidateV2(
        self,
        request: specedge_pb2.ValidateRequestV2,
        context
    ) -> specedge_pb2.ValidateResponseV2:
        self._logger.info("[DASD] ValidateV2() seq=%s bundles=%d", request.seq_id, len(request.bundles))
        seq_id = request.seq_id or ""
        if not request.bundles:
            # no bundles: return current state untouched
            return specedge_pb2.ValidateResponseV2(
                seq_id=seq_id,
                ack_bundle_id=0,
                last_accepted_pos=self._last_pos[seq_id],
                accept_bitmap=b"",
                next_credit=max(self._aimd_min_credit, self._credit[seq_id]),
                hint_mask=b"",
            )

        # for now: process the last bundle only (micro-batch later)
        b = request.bundles[-1]
        W = int(b.len)
        ack_id = int(b.bundle_id)

        # When DASD is disabled, keep the simple "accept all" behavior for safety.
        if not self._dasd_enable:
            accept_bitmap = _pack_all_ones_bitmap(W)
            r_obs = 1.0 if W > 0 else 0.0
        else:
            # DASD qlogp-based verifier:
            # If qlogp_i8 is present, we accept tokens from the front while
            # their score is above a threshold. Once a token falls below the
            # threshold, we stop accepting the rest of the window.
            accept_bits = [0] * W
            if W > 0 and b.qlogp_i8:
                scores = _decode_qlogp_i8(b.qlogp_i8)
                if len(scores) < W:
                    scores = scores + [scores[-1]] * (W - len(scores))
                scores = scores[:W]
                accepted = 0
                for i in range(W):
                    if scores[i] >= self._qlogp_i8_thresh:
                        accept_bits[i] = 1
                        accepted += 1
                    else:
                        break
                # Optionally enforce a minimum run length of accepts when we
                # start accepting; if accepted is non-zero but less than
                # _qlogp_i8_min_run, discard them.
                if 0 < accepted < self._qlogp_i8_min_run:
                    for i in range(accepted):
                        accept_bits[i] = 0
                    accepted = 0
                r_obs = (accepted / W) if W > 0 else 0.0
            else:
                # Fallback: if we don't have qlogp_i8, accept everything.
                accept_bits = [1] * W
                r_obs = 1.0 if W > 0 else 0.0

            accept_bitmap = _pack_bitmap_from_bits(accept_bits)
            self._logger.info(
                "[DASD] V2 server decision seq=%s bundle_id=%d start=%d W=%d accepted=%d r_obs=%.3f bits=%s",
                seq_id,
                ack_id,
                int(b.start_pos),
                W,
                sum(accept_bits),
                r_obs,
                accept_bits,
            )

        # last_accepted_pos moves to the end of the accepted prefix within this window.
        # Tokens in the window cover positions (start_pos+1) .. (start_pos+W).
        accepted_count = sum(accept_bits)
        if accepted_count > 0:
            new_pos = int(b.start_pos) + accepted_count
            if new_pos > self._last_pos[seq_id]:
                self._last_pos[seq_id] = new_pos

        # AIMD based on observed acceptance rate r_obs.
        c = self._credit[seq_id]
        if r_obs >= self._aimd_r_target:
            c = min(self._aimd_max_credit, c + self._aimd_inc)
        else:
            c = max(self._aimd_min_credit, int(c * self._aimd_dec_factor))
        self._credit[seq_id] = c

        # No PAS yet
        return specedge_pb2.ValidateResponseV2(
            seq_id=seq_id,
            ack_bundle_id=ack_id,
            last_accepted_pos=self._last_pos[seq_id],
            accept_bitmap=accept_bitmap,
            next_credit=c,
            hint_mask=b""  # PAS off for now
        )

    async def cleanup(self):
        # Nothing to release yet; placeholder for partiy with V1 server.
        self._logger.info("DASD cleanup complete")

from specedge_grpc import specedge_pb2 as pb2, specedge_pb2_grpc as stubs

async def main():
    async with grpc.aio.insecure_channel("127.0.0.1:8000") as ch:
        dasd = stubs.DASDServiceStub(ch)

        b1 = pb2.Bundle(seq_id="s1", bundle_id=1, start_pos=0,  len=8, token_ids=[10]*8)
        b2 = pb2.Bundle(seq_id="s1", bundle_id=2, start_pos=8,  len=8, token_ids=[11]*8)
        b3 = pb2.Bundle(seq_id="s1", bundle_id=3, start_pos=16, len=8, token_ids=[12]*8)

        for i, bundles in enumerate(([b1], [b2], [b3]), 1):
            resp = await dasd.ValidateV2(pb2.ValidateRequestV2(seq_id="s1", bundles=bundles), timeout=5)
            print(f"call#{i} ack={resp.ack_bundle_id} last={resp.last_accepted_pos} credit={resp.next_credit} bm_len={len(resp.accept_bitmap)}B")

if __name__ == "__main__":
    asyncio.run(main())