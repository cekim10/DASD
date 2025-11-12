from typing import Optional, Sequence

import grpc.aio
import torch
import os

from specedge_grpc import specedge_pb2, specedge_pb2_grpc
from util import decode, encode


class GrpcClientController:
    def __init__(self, host: str, device: torch.device) -> None:
        self.client_idx = 0

        self._host = host
        self._device = device
        self._channel = grpc.aio.insecure_channel(self._host)
        self._stub = specedge_pb2_grpc.SpecEdgeServiceStub(self._channel)

    async def request(
        self,
        client_idx: int,
        req_idx: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        parent_indices: torch.Tensor,
        prefill: bool = False,
        prefix: Optional[str] = None,
        token_ids: Optional[torch.Tensor] = None,
    ):
        if prefill and prefix is None:
            raise ValueError("Prefix must be provided for prefill requests.")

        input_ids_encoded = encode(input_ids)
        position_ids_encoded = encode(position_ids)
        cache_seq_indices_encoded = encode(cache_seq_indices)
        attention_mask_encoded = encode(attention_mask)
        parent_indices_encoded = encode(parent_indices)

        request = specedge_pb2.ValidateRequest(
            client_idx=client_idx,
            req_idx=req_idx,
            input_ids=input_ids_encoded,
            position_ids=position_ids_encoded,
            cache_seq_indices=cache_seq_indices_encoded,
            parent_indices=parent_indices_encoded,
            attention_mask=attention_mask_encoded,
            prefill=prefill,
            prefix=prefix,
        )

        resp = await self._stub.Validate(request)

        return decode(
            resp.selection,
            device=self._device,
            dtype=torch.long,
            shape=input_ids.size(-1),
        ), resp.prefill


class DasdGrpcClient:
    """Thin async wrapper for DASDService.ValidateV2."""

    def __init__(self, host: str, device: torch.device) -> None:
        self._host = host
        self._device = device
        self._channel = grpc.aio.insecure_channel(self._host)
        # NOTE: DASD stub lives in specedge_pb2_grpc as generated from your .proto
        self._stub = specedge_pb2_grpc.DASDServiceStub(self._channel)

    async def validate_v2(
        self,
        seq_id: str,
        bundles: Sequence[specedge_pb2.Bundle],
        timeout: float = 10.0,
    ) -> specedge_pb2.ValidateResponseV2:
        req = specedge_pb2.ValidateRequestV2(seq_id=seq_id, bundles=bundles)
        return await self._stub.ValidateV2(req, timeout=timeout)

    async def request(
        self,
        client_idx: int,
        req_idx: int,
        input_ids: torch.Tensor,  # [1, N] tokens in INPUT ORDER (mask order)
        position_ids: torch.Tensor,  # [1, N] absolute positions (may contain duplicates)
        cache_seq_indices: torch.Tensor,  # [N] TREE INDICES in INPUT ORDER  <-- use this
        attention_mask: torch.Tensor,  # unused by V2 shim
        parent_indices: torch.Tensor,  # [W] TREE INDICES of the parents  <-- use these
        prefill: bool = False,
        prefix: Optional[str] = None,
        token_ids: Optional[torch.Tensor] = None,  # [W] draft child tokens (same order as parents)
    ):
        """
        V2 adapter: send one ValidateV2 request, then synthesize a V1-style `selection`
        vector aligned to the INPUT ORDER (same order as cache_seq_indices).
        """
        if token_ids is None:
            raise ValueError("V2 adapter requires `token_ids` (one per candidate).")

        # Flatten to CPU lists
        input_1d = input_ids.view(-1)
        N = int(input_1d.size(0))
        input_indices = cache_seq_indices.view(-1).to("cpu").tolist()  # tree indices in slot order
        idx_to_slot = {int(ti): i for i, ti in enumerate(input_indices)}  # TREE INDEX -> SLOT

        parents = parent_indices.view(-1).to("cpu").tolist()  # tree indices of parents
        cand_tokens = token_ids.view(-1).to("cpu").tolist()
        W = len(parents)
        if len(cand_tokens) != W:
            raise ValueError(f"token_ids length ({len(cand_tokens)}) != parent_indices length ({W})")

        # Build bundle (minimal fields). You can enrich start_pos, qlogp_i8, etc. if your server uses them.
        bundle = specedge_pb2.Bundle(
            seq_id=f"{client_idx}:{req_idx}",
            bundle_id=0,
            start_pos=0,
            len=W,
            token_ids=cand_tokens,
            qlogp_i8=b"",
            credit_left=0,
            flags=0,
        )

        # ---- T1 echo gate (optional): keep this OFF unless debugging transport. ----
        if os.getenv("SPECEDGE_ECHO_TEST", "0") == "1":
            selection = input_1d.clone().to(self._device, dtype=torch.long)
            return selection, 0

        # Call DASD
        resp = await self.validate_v2(
            seq_id=f"{client_idx}:{req_idx}",
            bundles=[bundle],
            timeout=10.0,
        )

        # --- Decode accept bitmap (little-endian) ---
        bm = resp.accept_bitmap  # bytes

        def bit(i: int) -> int:
            return (bm[i // 8] >> (i % 8)) & 1 if i < W else 0

        # Build selection aligned to INPUT ORDER (slots 0..N-1). Only fill parent slots.
        sel = [0] * N
        for k in range(W):
            parent_tree_idx = int(parents[k])
            slot = idx_to_slot.get(parent_tree_idx)
            if slot is None:
                # The parent must be part of the INPUT set; if not, something upstream is misaligned.
                # Skip defensively rather than crashing.
                continue
            tok = int(cand_tokens[k])
            if bit(k):
                # accepted: equality at parent slot
                sel[slot] = tok
            else:
                # rejected: ensure inequality; choose a small id != tok
                sel[slot] = 1 if tok != 1 else 2

        selection = torch.tensor(sel, device=self._device, dtype=torch.long)
        return selection, 0


# --- ADD: helper factory so caller can pick v1/v2 by a flag ---
def get_grpc_client(use_v2: bool, host: str, device: torch.device):
    return DasdGrpcClient(host, device) if use_v2 else GrpcClientController(host, device)

