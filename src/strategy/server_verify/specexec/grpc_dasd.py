import asyncio
import os
from collections import defaultdict

import grpc.aio

import log
from specedge_grpc import specedge_pb2
from specedge_grpc import specedge_pb2_grpc


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

        # Per-sequence tracking
        # last accepted absolute position
        self._last_pos: dict[str, int] = defaultdict(int)
        # simple credit state per sequence
        self._credit: dict[str, int] = defaultdict(lambda: self._aimd_min_credit)

        self._logger.info(
            "DASD V2 server init: enable=%s tick_ms=%s aimd(r*=%.2f, +%d, x%.2f, min=%d, max=%d) "
            "pas(enable=%s, top_m=%d, every=%d, ttl=%d)",
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

        # dummy policy: accept all W tokens
        accept_bitmap = _pack_all_ones_bitmap(W)

        # Move the last accepted position forward
        new_pos = int(b.start_pos) + W
        if new_pos > self._last_pos[seq_id]:
            self._last_pos[seq_id] = new_pos

        # Toy AIMD: pretend perfect acceptance r=1.0
        r_obs = 1.0
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
            hint_mask=b"" # PAS off for now
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