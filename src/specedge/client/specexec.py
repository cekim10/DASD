import os
import asyncio
from typing import Optional, Tuple, List

import numpy as np
import torch

import log
import util
from config import SpecEdgeClientConfig as config
from specedge.client.proactive import SpecExecProactiveDraft
from specedge.network.grpc import get_grpc_client
from specedge_grpc import specedge_pb2
from specedge.tree import Tree


class SpecExecClient:
    def __init__(
        self,
        engine,
        tokenizer,
        prompt: str,
        max_len: int,
    ) -> None:
        # logging
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        self._logger.debug("Initializing SpecExecClient")

        self._optimization = config.optimization
        self._draft_forward_time_mode = (
            "no-sync" if self._optimization >= 2 else "event"
        )
        self._target_time_mode = "no-sync" if self._optimization >= 2 else "sync"

        self._device = config.device
        self._dtype = config.dtype

        self._max_n_beams = config.max_n_beams
        self._max_beam_len = config.max_beam_len
        self._max_branch_width = config.max_branch_width
        self._max_budget = config.max_budget

        self._proactive_type = config.proactive_type

        self._max_new_tokens = config.max_new_tokens
        self._client_idx = config.client_idx

        self._verify_configs()

        self._engine = engine
        self._tokenizer = tokenizer
        self._engine.reset()

        # ---- Proper prompt init (chat template if available) ----
        if getattr(self._tokenizer, "apply_chat_template", None) is not None:
            messages = [{"role": "user", "content": prompt}]
            prefix_tokens = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            prefix_tokens = self._tokenizer.encode(
                prompt, add_special_tokens=True, return_tensors="pt"
            )
        self._prompt = prompt
        self._prefix_tokens = prefix_tokens.to(self._device)[:, : config.max_len]
        self._num_original_tokens = self._prefix_tokens.numel()
        self._max_len = max_len

        self._tree = Tree(
            prefix_tokens=self._prefix_tokens,
            device=self._device,
            dtype=self._dtype,
            max_len=self._engine.max_len,
        )

        # feature flag + DASD runtime knobs
        self._use_v2 = os.getenv("DASD_ENABLE", "0") == "1"
        self._logger.info(f"use_v2? {self._use_v2}")
        self._seq_id = f"c{config.client_idx}_req"
        self._bundle_id = 0
        self._credit = int(os.getenv("DASD_AIMD_MIN_CREDIT", "4"))
        self._max_bundle_len = self._credit

        self._validator = get_grpc_client(self._use_v2, config.host, self._device)

        self._t2_single = os.getenv("SPECEDGE_T2_SINGLE_CAND", "0") == "1"
        self._t2_toggle = 0  # flip 0/1 each step

        self._proactive_client: Optional[SpecExecProactiveDraft] = None
        if self._proactive_type != "disabled":
            self._proactive_client = SpecExecProactiveDraft(
                tree=self._tree,
                engine=self._engine,
                max_len=self._max_len,
            )
            self._previous_proactive_draft = False
            self._proactive_draft = False

    def _verify_configs(self):
        if self._proactive_type not in ["included", "excluded", "disabled"]:
            raise ValueError(f"Invalid proactive_type: {self._proactive_type}")

    # ---------------- Core public API ----------------

    async def generate(self, req_idx: int):
        """Generate a sequence up to max_new_tokens."""
        self._logger.info("Generating sequence req_idx=%d", req_idx)
        util.set_seed(config.seed)
        step_idx = 0

        # Prefill phase
        self._logger.debug("Prefill phase: req_idx=%d, step_idx=%d", req_idx, step_idx)
        warmup_tokens = await self._cycle(req_idx, step_idx, prefill=True)
        self._prefix_tokens = torch.cat([self._prefix_tokens, warmup_tokens], dim=-1)

        step_idx = 1
        eos_flag = False

        # Speculative decoding phase
        while (
            self._prefix_tokens.numel()
            < self._max_new_tokens + self._num_original_tokens + warmup_tokens.numel()
            and not eos_flag
        ):
            self._logger.debug(
                "Speculative Decoding phase: req_idx=%d, step_idx=%d", req_idx, step_idx
            )
            if self._use_v2:
                fresh_tokens = await self._cycle_dasd(req_idx, step_idx)
            else:
                fresh_tokens = await self._cycle(req_idx, step_idx)

            eos_positions = (fresh_tokens == self._tokenizer.eos_token_id).nonzero()
            if eos_positions.numel() > 0:
                eos_idx = eos_positions[0, 0].item()
                fresh_tokens = fresh_tokens[: eos_idx + 1]
                eos_flag = True

            self._prefix_tokens = torch.cat([self._prefix_tokens, fresh_tokens], dim=-1)
            step_idx += 1

        self._logger.info("Finished generating sequence req_idx=%d", req_idx)
        self._logger.info(
            "Generated sequence: \n%s",
            self._tokenizer.decode(self._prefix_tokens[0], skip_special_tokens=True),
        )

    # ---------------- Speculative cycles ----------------

    async def _cycle_dasd(self, req_idx: int, step_idx: int) -> torch.Tensor:
        """One DASD (V2) step: grow draft tree, send bundles to ValidateV2, apply bitmap."""
        self._logger.info("DASD/V2 cycle: req_idx=%d, step_idx=%d", req_idx, step_idx)

        # 1) Draft: grow the speculative tree as in V1
        with util.Timing(device=self._device, mode="sync") as draft_t:
            draft_stats = self._grow_tree(prefill=False)

        # 2) Build DASD bundles up to current credit (and remember the path)
        bundles, start_pos, path_indices = self._build_dasd_bundles(
            max_total_len=self._credit
        )

        # 3) Call ValidateV2 on the server
        with util.Timing(device=self._device, mode="sync") as wait_t:
            resp = await self._validator.validate_v2(
                seq_id=f"{self._seq_id}_{req_idx}",
                bundles=bundles,
            )

        # 4) Apply accept bitmap along that path and update local tree / KV cache
        with util.Timing(device=self._device, mode="sync") as post_t:
            fresh_token_ids = self._apply_dasd_accept(resp, start_pos, path_indices)

        # 5) Update credit from server AIMD
        try:
            self._credit = max(1, int(resp.next_credit))
        except Exception:
            # Be robust if server does not set next_credit
            self._credit = max(1, self._credit)
        self._max_bundle_len = self._credit

        # 6) Log stats in the same schema as _cycle
        target_stats = {
            "preprocess_t": 0.0,                  # bundle building is cheap, folded into draft_t/post_t
            "wait_t": wait_t.elapsed,
            "postprocess_t": post_t.elapsed,
            "num_accepted_tokens": int(fresh_token_ids.size(-1)),
            "prefill": 0,
            "proactive": False,
            "previous_proactive": False,
        }

        self._result_logger.log(
            {
                "client_idx": self._client_idx,
                "req_idx": req_idx,
                "step_idx": step_idx,
                "draft": {
                    "forward": draft_stats["forward_t"],
                    "end_to_end": draft_t.elapsed,
                },
                "target": {
                    "client_preprocess": target_stats["preprocess_t"],
                    "client_wait": target_stats["wait_t"],
                    "client_postprocess": target_stats["postprocess_t"],
                    "end_to_end": wait_t.elapsed + post_t.elapsed,
                    "prefill": target_stats["prefill"],
                    "proactive": target_stats["proactive"],
                    "prev_proactive": target_stats["previous_proactive"],
                },
                "num_accepted_tokens": target_stats["num_accepted_tokens"],
            }
        )

        return fresh_token_ids

    async def _cycle(self, req_idx: int, step_idx: int, prefill=False) -> torch.Tensor:
        with util.Timing(device=self._device, mode="sync") as draft_t:
            draft_stats = self._grow_tree(prefill)

        with util.Timing(device=self._device, mode="sync") as target_t:
            fresh_token_ids, target_stats = await self._validate_tree(req_idx, prefill)

        self._result_logger.log(
            {
                "client_idx": self._client_idx,
                "req_idx": req_idx,
                "step_idx": step_idx,
                "draft": {
                    "forward": draft_stats["forward_t"],
                    "end_to_end": draft_t.elapsed,
                },
                "target": {
                    "client_preprocess": target_stats["preprocess_t"],
                    "client_wait": target_stats["wait_t"],
                    "client_postprocess": target_stats["postprocess_t"],
                    "end_to_end": target_t.elapsed,
                    "prefill": target_stats["prefill"],
                    "proactive": target_stats["proactive"],
                    "prev_proactive": target_stats["previous_proactive"],
                },
                "num_accepted_tokens": target_stats["num_accepted_tokens"],
            }
        )
        return fresh_token_ids

    # ---------------- Draft growth ----------------

    def _grow_tree(self, prefill: bool):
        self._logger.debug("Growing tree")

        draft_forward_times: List[float] = []

        max_beam_len = self._max_beam_len
        if self._proactive_type == "included" and getattr(self, "_proactive_draft", False):
            max_beam_len = max(0, self._max_beam_len - config.proactive_max_beam_len)

        if torch.where(self._tree.status == self._tree.CANDIDATE)[0].numel() == 0:
            max_beam_len = 0

        for cnt in range(max_beam_len):
            self._logger.debug("Growing tree: %d / %d", cnt, max_beam_len)

            logits, beam_indices, beam_positions, beam_scores, draft_forward_t = (
                self._process_candidates(prefill)
            )
            prefill = False
            draft_forward_times.append(draft_forward_t)

            (
                next_beam_ids,
                next_beam_positions,
                next_beam_indices,
                beam_logprobs,
            ) = self._get_next_beams(
                logits=logits,
                beam_indices=beam_indices,
                beam_positions=beam_positions,
                beam_scores=beam_scores,
            )

            if next_beam_ids.numel() == 0:
                self._logger.debug("No more beams to grow")
                break

            if (
                self._tree.end - self._tree.prefix_len >= self._max_budget
                and not self._check_new_token_in_budget(beam_logprobs)
            ):
                self._logger.debug("Max budget reached. early stopping")
                break

            self._tree.add(
                token_ids=next_beam_ids,
                token_positions=next_beam_positions,
                parent_indices=next_beam_indices,
                logprobs=beam_logprobs,
            )

        if self._tree.end - self._tree.prefix_len >= self._max_budget:
            self._logger.debug("Trimming tree")
            self._trim_by_budget()

        return {"forward_t": draft_forward_times}

    def _process_candidates(self, warmup: bool):
        self._logger.debug("Processing candidates")
        candidate_indices = torch.where(
            self._tree.status[: self._tree.end] == self._tree.CANDIDATE
        )[0]

        if candidate_indices.numel() > self._max_n_beams:
            self._logger.debug("Choosing top %d candidates", self._max_n_beams)
            cumulative_logprobs = self._tree.logprobs[candidate_indices]
            top_k_indices = cumulative_logprobs.topk(
                k=self._max_n_beams, sorted=False
            ).indices
            candidate_indices = candidate_indices[top_k_indices]
            candidate_indices, _ = candidate_indices.sort()

        if candidate_indices.numel() == 0:
            # No work this round
            return (
                torch.empty(0, 0, device=self._device),
                candidate_indices,
                torch.empty(0, device=self._device, dtype=torch.long),
                torch.empty(0, device=self._device),
                0.0,
            )

        if warmup:
            prefill_input_indices = torch.arange(
                candidate_indices.min().item(), device=self._device
            )
            prefill_input_ids = self._tree.tokens[prefill_input_indices].unsqueeze(0)
            prefill_position_ids = self._tree.positions[
                prefill_input_indices
            ].unsqueeze(0)
            prefill_cache_seq_indices = prefill_input_indices
            prefill_attention_mask = self._tree.amask[..., prefill_input_indices, :]

            self._engine.prefill(
                input_ids=prefill_input_ids,
                position_ids=prefill_position_ids,
                batch_idx=0,
                cache_seq_indices=prefill_cache_seq_indices,
                attention_mask=prefill_attention_mask,
            )

        input_indices = candidate_indices
        input_ids = self._tree.tokens[input_indices].unsqueeze(0)
        position_ids = self._tree.positions[input_indices].unsqueeze(0)
        cache_seq_indices = input_indices
        cache_batch_indices = torch.full_like(
            cache_seq_indices, 0, dtype=torch.long, device=self._device
        )
        attention_mask = self._tree.amask[..., input_indices, :]

        with util.Timing(device=self._device, mode=self._draft_forward_time_mode) as t:
            logits = self._engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )

        self._tree.status[candidate_indices] = self._tree.PROCESSED
        beam_scores = self._tree.logprobs[candidate_indices]
        beam_positions = self._tree.positions[candidate_indices]
        logits = logits[0, -candidate_indices.size(-1) :, :]

        return (logits, candidate_indices, beam_positions, beam_scores, t.elapsed)

    def _get_next_beams(
        self,
        logits: torch.Tensor,
        beam_indices: torch.Tensor,
        beam_positions: torch.Tensor,
        beam_scores: torch.Tensor,
    ):
        self._logger.debug("Getting next beams")
        if logits.numel() == 0:
            return (
                torch.empty(0, device=self._device, dtype=torch.long),
                torch.empty(0, device=self._device, dtype=torch.long),
                torch.empty(0, device=self._device, dtype=torch.long),
                torch.empty(0, device=self._device),
            )

        DECAY_FACTOR = np.log(0.9)

        logprobs = torch.log_softmax(logits, dim=-1)  # [n_beams, vocab]
        logprobs_k = logprobs.topk(k=self._max_branch_width, dim=-1, sorted=False)
        leaves_ids = logprobs_k.indices
        leaves_probs = logprobs_k.values

        flat_incoming_probs = (
            beam_scores.unsqueeze(-1) + DECAY_FACTOR + leaves_probs
        ).flatten()
        flat_incoming_ids = leaves_ids.flatten()

        joint_probs = torch.concat(
            [
                self._tree.logprobs[self._tree.prefix_len : self._tree.end],
                flat_incoming_probs,
            ]
        )

        if (
            joint_probs.size(-1) > self._max_budget
            or joint_probs.size(-1) + (self._tree.end - self._tree.prefix_len)
            > self._max_len
        ):
            min_joint_prob = joint_probs.topk(
                k=self._max_budget, sorted=False, dim=-1
            ).values.min()
            flat_best_mask = torch.where(flat_incoming_probs >= min_joint_prob)[0]
            flat_best_indices = flat_best_mask
            best_children_token_ids = flat_incoming_ids[flat_best_indices]
        else:
            flat_best_indices = torch.arange(
                flat_incoming_probs.size(0), device=logprobs.device
            )
            best_children_token_ids = flat_incoming_ids

        best_hypo_ids = flat_best_indices // self._max_branch_width
        best_beam_indices = beam_indices[best_hypo_ids]
        best_children_positions = beam_positions[best_hypo_ids] + 1

        return (
            best_children_token_ids,
            best_children_positions,
            best_beam_indices,
            flat_incoming_probs[flat_best_indices],
        )

    def _check_new_token_in_budget(self, cumulative_beam_scores: torch.Tensor):
        if (self._tree.end - self._tree.prefix_len) == 0:
            return True
        lowest_tree_logprob = (
            self._tree.logprobs[self._tree.prefix_len : self._tree.end]
            .topk(k=self._max_budget, dim=-1, sorted=False)
            .values.min()
        )
        best_new_logprob = cumulative_beam_scores.max()
        return best_new_logprob >= lowest_tree_logprob

    def _trim_by_budget(self):
        src_indices = (
            self._tree.logprobs[self._tree.prefix_len : self._tree.end]
            .topk(k=self._max_budget, sorted=False)
            .indices
            + self._tree.prefix_len
        )
        dest_indices = torch.arange(
            self._tree.prefix_len,
            self._tree.prefix_len + src_indices.size(-1),
            device=self._device,
        )

        self._tree.gather(src_indices, dest_indices)
        self._engine.gather(src_indices, dest_indices)

    # ---------------- Selection / Validation paths ----------------

    def _append_model_next_token(self) -> torch.Tensor:
        """
        Append exactly one next token by running a single forward step at the *true* last prefix.
        This replaces any 'bonus-token' hacks and keeps the continuation on-manifold.
        Returns: [1, 1] tensor with the appended token id.
        """
        last_idx = self._tree.end - 1
        input_ids = self._tree.tokens[last_idx:last_idx + 1].unsqueeze(0)           # [1,1]
        position_ids = self._tree.positions[last_idx:last_idx + 1].unsqueeze(0)     # [1,1]
        cache_seq_idx = torch.tensor([last_idx], device=self._device)
        cache_batch_idx = torch.tensor([0], device=self._device)
        attention_mask = self._tree.amask[..., last_idx:last_idx + 1, :]            # [1,1,1,E]

        logits = self._engine.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_idx,
            cache_seq_indices=cache_seq_idx,
            attention_mask=attention_mask,
        )  # [1,1,V]
        next_id = torch.argmax(logits[0, -1], dim=-1).view(1)                       # [1]

        self._tree.add(
            token_ids=next_id,
            token_positions=self._tree.positions[last_idx] + 1,
            parent_indices=torch.tensor([last_idx], device=self._device),
            logprobs=torch.tensor([0.0], device=self._device),
        )
        self._tree.prefix_len = self._tree.end
        self._tree.status[: self._tree.prefix_len - 1] = self._tree.PROMPT
        return next_id.unsqueeze(0)                                                  # [1,1]

    def _apply_selection_like_v1(
        self,
        selection_1xN: torch.Tensor,
        input_token_map_bool: torch.Tensor,
        target_token_map_bool: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a V1-style 'selection' vector (size N over slots), compute accepts,
        reorder KV/tree by best sequence, then append ONE real model-predicted token.
        """
        interim_t = torch.ones_like(self._tree.tokens[: self._tree.end])
        interim_t[input_token_map_bool] = selection_1xN.view(-1)

        draft_token_choices = self._tree.tokens[: self._tree.end][target_token_map_bool]
        parent_indices = self._tree.parents[: self._tree.end][target_token_map_bool]
        target_token_choices = interim_t[parent_indices]
        accept_flags = (draft_token_choices == target_token_choices)

        target_token_indices = torch.where(target_token_map_bool)[0]
        accept_indices = target_token_indices[accept_flags]

        # Build sequence mask and reorder
        accept_mask = torch.zeros(self._tree.end, device=self._device)
        accept_mask[: self._tree.prefix_len] = 1
        accept_mask[accept_indices] = 1

        # Use amask rows to pick the longest valid sequence
        accepted_amask = self._tree.amask[0, 0, :, : self._tree.end] * accept_mask
        mask_row_sums = self._tree.amask[0, 0, :, : self._tree.end].sum(dim=1).to(torch.long)
        seq_lengths = accepted_amask.sum(dim=1).to(torch.long)
        best_seq_idx = (mask_row_sums * (mask_row_sums == seq_lengths)).argmax()
        best_seq_mask = self._tree.amask[0, 0, best_seq_idx, : self._tree.end].to(torch.bool)

        # Fresh = all accepted after prefix
        fresh_token_indices = torch.where(best_seq_mask[self._tree.prefix_len:])[0] + self._tree.prefix_len
        fresh_token_ids = self._tree.tokens[fresh_token_indices]

        # Reorder engine/tree
        self._reorder_by_sequence(best_seq_mask)

        # Append ONE real next token from model
        next_token = self._append_model_next_token()  # [1,1]
        if fresh_token_ids.numel() == 0:
            return next_token  # [1,1]
        else:
            return torch.cat([fresh_token_ids.unsqueeze(0), next_token], dim=-1)

    async def _validate_tree(self, req_idx: int, prefill=False):
        self._logger.debug("Validating tree")

        with util.Timing(device=self._device, mode=self._target_time_mode) as preprocess_t:
            # Candidates = PROCESSED nodes after prefix
            target_token_map_bool = (self._tree.status[: self._tree.end] >= self._tree.PROCESSED)
            target_token_map_bool[: self._tree.prefix_len] = False
            target_token_indices = torch.where(target_token_map_bool)[0]
            target_parent_indices = self._tree.parents[: self._tree.end][target_token_map_bool]

            # Build slots: unique parent absolute positions (sorted ascending)
            parent_pos_unique = torch.unique(target_parent_indices, sorted=True)
            if parent_pos_unique.numel() == 0:
                fresh = torch.empty((1, 0), dtype=torch.long, device=self._device)
                stats = {
                    "preprocess_t": 0.0,
                    "wait_t": 0.0,
                    "postprocess_t": 0.0,
                    "num_accepted_tokens": 0,
                    "prefill": 0,
                    "previous_proactive": False,
                    "proactive": False,
                }
                return fresh, stats

            input_token_map_bool = torch.zeros(self._tree.end, dtype=torch.bool, device=self._device)
            input_token_map_bool[parent_pos_unique] = True

            input_ids = self._tree.tokens[: self._tree.end][input_token_map_bool].unsqueeze(0)
            position_ids = self._tree.positions[: self._tree.end][input_token_map_bool].unsqueeze(0)
            cache_seq_indices = torch.where(input_token_map_bool)[0]
            attention_mask = self._tree.amask[..., cache_seq_indices, :]

            # ---- Hard asserts for T2 path correctness ----
            pos_list = [int(p) for p in position_ids.view(-1).tolist()]
            assert len(pos_list) == len(set(pos_list)), f"[ASSERT] Duplicate position_ids: {pos_list}"

            pos_to_slot = {p: i for i, p in enumerate(pos_list)}
            missing = [int(p) for p in target_parent_indices.tolist() if int(p) not in pos_to_slot]
            assert not missing, f"[ASSERT] Missing parent positions in slots: {missing}"

            # shapes
            assert position_ids.shape == input_ids.shape, \
                f"[ASSERT] position_ids shape {position_ids.shape} != input_ids shape {input_ids.shape}"

        # Echo test short-circuit (no mutation)
        if os.getenv("SPECEDGE_ECHO_TEST", "0") == "1":
            self._logger.info("SPECEDGE_ECHO_TEST enabled: returning 0-accepts, no mutation.")
            fresh = torch.empty((1, 0), dtype=torch.long, device=self._device)
            stats = {
                "preprocess_t": 0.0,
                "wait_t": 0.0,
                "postprocess_t": 0.0,
                "num_accepted_tokens": 0,
                "prefill": 0,
                "previous_proactive": getattr(self, "_previous_proactive_draft", False),
                "proactive": False,
            }
            return fresh, stats

        with util.Timing(device=self._device, mode=self._target_time_mode) as wait_t:
            prefix = self._prompt if prefill else None
            draft_token_choices = self._tree.tokens[: self._tree.end][target_token_map_bool]
            target_result = asyncio.create_task(
                self._validator.request(
                    client_idx=self._client_idx,
                    req_idx=req_idx,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=attention_mask,
                    parent_indices=target_parent_indices,
                    prefill=prefill,
                    prefix=prefix,
                    token_ids=draft_token_choices,  # V2 adapter needs this
                )
            )
            await asyncio.sleep(0.00001)

            if self._proactive_client is not None:
                (
                    root_leaf_idx,
                    root_token_id,
                    proactive_tree_prefix_len,
                    proactive_tree_end,
                ) = self._proactive_client.draft()

            selection, prefill_cnt = (
                target_result.result() if target_result.done() else await target_result
            )

        with util.Timing(device=self._device, mode=self._target_time_mode) as postprocess_t:
            fresh = self._apply_selection_like_v1(
                selection_1xN=selection,
                input_token_map_bool=input_token_map_bool,
                target_token_map_bool=target_token_map_bool,
            )

        stats = {
            "preprocess_t": preprocess_t.elapsed,
            "wait_t": wait_t.elapsed,
            "postprocess_t": postprocess_t.elapsed,
            "num_accepted_tokens": fresh.size(-1),
            "prefill": prefill_cnt,
            "previous_proactive": getattr(self, "_previous_proactive_draft", False)
            if self._proactive_client
            else False,
            "proactive": getattr(self, "_proactive_draft", False) if self._proactive_client else False,
        }
        return fresh, stats

    # ---------------- Reorder helpers ----------------

    def _reorder_by_sequence(self, seq_mask: torch.Tensor):
        """Reorder tree and engine's kv cache according to seq_mask."""
        seq_indices = torch.where(seq_mask != 0)[0]
        self._engine.gather(
            seq_indices,
            torch.arange(seq_indices.size(-1), device=self._device),
        )
        self._tree.reorder_by_sequence(seq_mask, seq_indices)

    def _reorder_by_sequence_proactive(
        self,
        seq_mask: torch.Tensor,
        proactive_tree_prefix_len: int,
        proactive_tree_end: int,
    ):
        """
        (Kept intact) Reorders KV when Proactive Draft is valid.
        """
        seq_indices = torch.where(seq_mask != 0)[0]
        max_src_idx = proactive_tree_end
        mapping_tensor = torch.full(
            (max_src_idx,), -1, dtype=torch.long, device=self._device
        )

        new_prefix_len = int(torch.sum(seq_mask).item())
        if torch.any(seq_mask[self._tree.prefix_len :]):
            src_indices = seq_indices[seq_indices >= self._tree.prefix_len]
            dest_indices = torch.arange(
                self._tree.prefix_len, new_prefix_len, device=self._device
            )
            mapping_tensor[src_indices] = dest_indices

            self._tree.tokens[dest_indices] = self._tree.tokens[src_indices]
            self._tree.positions[dest_indices] = dest_indices
            self._tree.parents[dest_indices] = dest_indices - 1
            self._tree.status[dest_indices] = self._tree.GENERATED

        src_indices = torch.arange(
            proactive_tree_prefix_len, proactive_tree_end, device=self._device
        )
        dest_indices = torch.arange(
            new_prefix_len,
            new_prefix_len + proactive_tree_end - proactive_tree_prefix_len,
            device=self._device,
        )
        mapping_tensor[src_indices] = dest_indices

        self._tree.tokens[dest_indices] = self._tree.tokens[src_indices]
        self._tree.positions[dest_indices] = self._tree.positions[src_indices]
        self._tree.parents[dest_indices] = mapping_tensor[
            self._tree.parents[src_indices]
        ]
        self._tree.status[dest_indices] = self._tree.status[src_indices]
        self._tree.logprobs[dest_indices] = self._tree.logprobs[src_indices]
        self._tree.amask[
            ...,
            dest_indices,
            new_prefix_len : new_prefix_len
            + proactive_tree_end
            - proactive_tree_prefix_len,
        ] = self._tree.amask[
            ..., src_indices, proactive_tree_prefix_len:proactive_tree_end
        ]

        self._tree.end = new_prefix_len + proactive_tree_end - proactive_tree_prefix_len
        self._tree.prefix_len = new_prefix_len + 1

        self._tree.status[: self._tree.prefix_len - 1] = self._tree.PROMPT
        self._tree.status[self._tree.prefix_len - 1 : self._tree.prefix_len + 1] = (
            self._tree.PROCESSED
        )
        self._tree.status[self._tree.status == self._tree.POST_CANDIDATE] = (
            self._tree.CANDIDATE
        )
        self._tree.status[self._tree.status == self._tree.POST_PROCESSED] = (
            self._tree.PROCESSED
        )

        self._tree.logprobs[self._tree.end :].zero_()
        self._tree._data[:, self._tree.end :].zero_()

        _causal_mask = torch.tril(
            torch.ones(
                self._tree.prefix_len,
                self._tree.prefix_len,
                dtype=self._dtype,
                device=self._device,
            )
        )
        self._tree.amask[..., : self._tree.prefix_len, : self._tree.prefix_len] = _causal_mask
        self._tree.amask[..., self._tree.prefix_len : self._tree.end, : self._tree.prefix_len] = 1.0

        src_indices = seq_mask[: self._tree.prefix_len]
        src_indices = torch.where(src_indices)[0]
        dst_indices = torch.arange(src_indices.size(-1), device=self._device)
        self._engine.gather(src_indices, dst_indices)

    # ---------------- DASD bundle path ----------------

    def _build_dasd_bundles(self, max_total_len: int) -> Tuple[list, int, torch.Tensor]:
        """Pick up to `max_total_len` candidate tokens along a *single* linear path.

        We choose the best leaf (highest cumulative logprob among PROCESSED nodes
        after the prefix), then walk its ancestors back to the prefix to form a
        straight chain. This matches the ValidateV2 semantics of scanning a
        single linear draft segment.
        """
        # Candidates that have been drafted and processed, but are not yet in prefix
        mask = (self._tree.status[: self._tree.end] >= self._tree.PROCESSED)
        mask[: self._tree.prefix_len] = False
        cand_idx = torch.where(mask)[0]

        if cand_idx.numel() == 0:
            # No drafted tokens this round: send an empty bundle so server can
            # still update credit, and let the client advance with base model.
            start_pos = int(self._tree.positions[self._tree.end - 1].item())
            bundle = specedge_pb2.Bundle(
                seq_id="",
                bundle_id=self._bundle_id + 1,
                start_pos=start_pos,
                len=0,
                token_ids=[],
                qlogp_i8=b"",
                credit_left=self._credit,
                flags=0,
            )
            empty_path = torch.empty(0, dtype=torch.long, device=self._device)
            return [bundle], start_pos, empty_path

        # Pick the best leaf among current candidates.
        cand_logprobs = self._tree.logprobs[cand_idx]
        best_leaf_rel = torch.argmax(cand_logprobs)
        best_leaf = cand_idx[best_leaf_rel].item()

        # Walk back from best_leaf to the current prefix to form a chain.
        path_indices: List[int] = []
        cur = best_leaf
        # We stop once we reach a position that is already in the committed prefix.
        while cur >= self._tree.prefix_len and len(path_indices) < max_total_len:
            path_indices.append(cur)
            cur = int(self._tree.parents[cur].item())

        if not path_indices:
            # Should not generally happen, but be defensive.
            start_pos = int(self._tree.positions[self._tree.end - 1].item())
            bundle = specedge_pb2.Bundle(
                seq_id="",
                bundle_id=self._bundle_id + 1,
                start_pos=start_pos,
                len=0,
                token_ids=[],
                qlogp_i8=b"",
                credit_left=self._credit,
                flags=0,
            )
            empty_path = torch.empty(0, dtype=torch.long, device=self._device)
            return [bundle], start_pos, empty_path

        # Reverse to go from shallowest to deepest along this path.
        path_indices.reverse()
        path_t = torch.tensor(path_indices, dtype=torch.long, device=self._device)

        tok_ids = self._tree.tokens[path_t].tolist()
        take = len(tok_ids)
        # The parent absolute position is one before the first token in the chain.
        start_pos = int(self._tree.positions[path_t[0]].item() - 1)

        # (Optional) pack per-token qlogp as int8; for now we send zeros.
        qlogp = torch.zeros(take, dtype=torch.int8, device=self._device)

        self._bundle_id += 1
        bundle = specedge_pb2.Bundle(
            seq_id="",
            bundle_id=self._bundle_id,
            start_pos=start_pos,
            len=take,
            token_ids=tok_ids,
            qlogp_i8=util.encode(qlogp),
            credit_left=max_total_len - take,
            flags=0,
        )
        return [bundle], start_pos, path_t

    def _apply_dasd_accept(
        self,
        resp: "specedge_pb2.ValidateResponseV2",
        start_pos: int,
        path_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply DASD accept bitmap along a linear path by reconstructing a V1-style selection vector,
        and delegate to _apply_selection_like_v1 so token ordering matches the validated path.
        """
        # If there was no path for this round, just advance the base model.
        if path_indices.numel() == 0 or not resp.accept_bitmap:
            return self._append_model_next_token()

        # Decode resp.accept_bitmap (LSB-first per byte) into bits
        bits: List[int] = []
        for b in resp.accept_bitmap:
            for k in range(8):
                bits.append((b >> k) & 1)

        self._logger.info(f"accept_bitmap: {bits}")
        L = min(len(bits), path_indices.numel())
        bits = bits[:L]
        if L == 0:
            return self._append_model_next_token()

        # Build target_token_map_bool: True for all indices in path_indices[:L], else False
        target_token_map_bool = torch.zeros(self._tree.end, dtype=torch.bool, device=self._device)
        if L > 0:
            target_token_map_bool[path_indices[:L]] = True

        # Compute target_parent_indices and parent_pos_unique (sorted)
        target_parent_indices = self._tree.parents[: self._tree.end][target_token_map_bool]
        parent_pos_unique = torch.unique(target_parent_indices, sorted=True)

        # Build input_token_map_bool: True at parent_pos_unique positions
        input_token_map_bool = torch.zeros(self._tree.end, dtype=torch.bool, device=self._device)
        if parent_pos_unique.numel() > 0:
            input_token_map_bool[parent_pos_unique] = True

        # Allocate selection and mapping from parent position to slot index
        selection = torch.empty((1, parent_pos_unique.numel()), dtype=torch.long, device=self._device)
        pos_to_slot = {int(p): i for i, p in enumerate(parent_pos_unique.tolist())}
        # Initialize with a default "reject" token: eos_token_id
        selection.fill_(self._tokenizer.eos_token_id)

        vocab_size = getattr(self._tokenizer, "vocab_size", 65536)
        # For each i in range(L), set selection[0, slot] = tok if bits[i]==1
        for i in range(L):
            idx = int(path_indices[i].item())
            parent = int(self._tree.parents[idx].item())
            slot = pos_to_slot.get(parent)
            if slot is None:
                continue
            tok = int(self._tree.tokens[idx].item())
            if bits[i] == 1:
                # If default token happens to equal tok, change it to something else for rejects
                if selection[0, slot] == tok:
                    # Change reject token to (tok + 1) % vocab_size
                    selection[0, slot] = (tok + 1) % vocab_size
                selection[0, slot] = tok
            else:
                # Leave as default reject token, but ensure it differs from tok
                if selection[0, slot] == tok:
                    selection[0, slot] = (tok + 1) % vocab_size

        # Delegate to _apply_selection_like_v1
        fresh = self._apply_selection_like_v1(selection, input_token_map_bool, target_token_map_bool)
        return fresh