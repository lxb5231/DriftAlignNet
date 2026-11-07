import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@torch.no_grad()
def build_global_channel_graph(
    train_history,
    candidate_budget=16,
    eps=1e-5,
    batch_size=8,
    device=None,
):
    """Build a fixed Raw/Diff neighbor graph from training histories only.

    Args:
        train_history: Historical inputs shaped ``[W, L, N]`` or ``[L, N]``.
            The caller owns split selection; this helper never accesses a data
            loader, labels, validation data, or test data.
        candidate_budget: Maximum number of saved non-target neighbors.
        eps: Numerical stability epsilon used by correlation normalization.
        batch_size: Number of history windows accumulated at once.
        device: Optional device used for the one-time graph computation.
    """
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    history = torch.as_tensor(train_history)
    if history.ndim == 2:
        history = history.unsqueeze(0)
    if history.ndim != 3:
        raise ValueError("train_history must have shape [W, L, N] or [L, N].")
    if history.size(1) < 2:
        raise ValueError("Each training history must contain at least 2 steps.")
    num_channels = history.size(2)
    if num_channels < 2:
        raise ValueError("At least two channels are required to build a graph.")

    compute_device = torch.device(device) if device is not None else history.device
    score_sum = torch.zeros(
        num_channels,
        num_channels,
        dtype=torch.float32,
        device=compute_device,
    )
    window_count = 0

    def normalized(values):
        centered = values - values.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(variance + eps)

    for start in range(0, history.size(0), batch_size):
        windows = history[start : start + batch_size].to(
            device=compute_device,
            dtype=torch.float32,
        )
        channel_first = windows.permute(0, 2, 1)
        raw_norm = normalized(channel_first)
        raw_corr = torch.matmul(
            raw_norm,
            raw_norm.transpose(-1, -2),
        ) / raw_norm.size(-1)

        differences = channel_first[..., 1:] - channel_first[..., :-1]
        diff_norm = normalized(differences)
        diff_corr = torch.matmul(
            diff_norm,
            diff_norm.transpose(-1, -2),
        ) / diff_norm.size(-1)

        scores = 0.5 * raw_corr.abs() + 0.5 * diff_corr.abs()
        score_sum.add_(scores.clamp(0.0, 1.0).sum(dim=0))
        window_count += windows.size(0)

    global_scores = score_sum / float(window_count)
    diagonal = torch.eye(
        num_channels,
        dtype=torch.bool,
        device=compute_device,
    )
    global_scores.masked_fill_(diagonal, float("-inf"))
    num_neighbors = min(candidate_budget, num_channels - 1)
    neighbor_scores, neighbor_indices = torch.topk(
        global_scores,
        k=num_neighbors,
        dim=-1,
        largest=True,
        sorted=True,
    )
    return neighbor_indices, neighbor_scores.clamp(0.0, 1.0)


class StatisticalTopKChannelSelector(nn.Module):
    """Select the most statistically similar channels for each target channel."""

    def __init__(
        self,
        top_k,
        raw_weight=0.5,
        diff_weight=0.5,
        eps=1e-5,
        return_auxiliary=True,
    ):
        super().__init__()

        self.top_k = top_k
        self.raw_weight = raw_weight
        self.diff_weight = diff_weight
        self.eps = eps
        self.return_auxiliary = return_auxiliary

    def _normalize_over_time(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)

    @staticmethod
    def _gather_auxiliary_channels(x_channel, topk_indices):
        batch_size, num_channels, seq_len = x_channel.shape
        top_k = topk_indices.size(-1)

        source = x_channel.unsqueeze(1).expand(
            batch_size,
            num_channels,
            num_channels,
            seq_len,
        )
        gather_index = topk_indices.unsqueeze(-1).expand(
            batch_size,
            num_channels,
            top_k,
            seq_len,
        )
        return torch.gather(source, dim=2, index=gather_index)

    def forward(self, x):
        num_channels = x.size(-1)

        # [B, L, N] -> [B, N, L]
        x_channel = x.permute(0, 2, 1)
        x_norm = self._normalize_over_time(x_channel)

        raw_corr = torch.matmul(
            x_norm,
            x_norm.transpose(-1, -2),
        ) / x_norm.size(-1)
        raw_corr = raw_corr.clamp(-1.0, 1.0)

        diff = x_norm[..., 1:] - x_norm[..., :-1]
        diff_norm = self._normalize_over_time(diff)
        diff_corr = torch.matmul(
            diff_norm,
            diff_norm.transpose(-1, -2),
        ) / diff_norm.size(-1)
        diff_corr = diff_corr.clamp(-1.0, 1.0)

        similarity = (
            self.raw_weight * raw_corr.abs() + self.diff_weight * diff_corr.abs()
        ).clamp(0.0, 1.0)

        diagonal_mask = torch.eye(
            num_channels,
            dtype=torch.bool,
            device=x.device,
        ).unsqueeze(0)
        selection_score = similarity.masked_fill(
            diagonal_mask,
            float("-inf"),
        )
        topk_scores, topk_indices = torch.topk(
            selection_score,
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )

        result = {
            "similarity": similarity,
            "topk_indices": topk_indices,
            "topk_scores": topk_scores,
            "raw_correlation": raw_corr,
            "diff_correlation": diff_corr,
        }
        if self.return_auxiliary:
            result["auxiliary_x"] = self._gather_auxiliary_channels(
                x_channel,
                topk_indices,
            )

        return result


class DirectionalLagMMRChannelSelector(nn.Module):
    """Select predictive, non-redundant channels with directional lag awareness.

    The selector first performs inexpensive synchronous Raw-Diff screening,
    searches only lags where a candidate leads its target, estimates lag
    stability over sub-windows, and finally applies target-aligned MMR.
    """

    def __init__(
        self,
        top_k,
        max_lag=8,
        candidate_multiplier=8,
        raw_weight=0.5,
        diff_weight=0.5,
        lag_penalty=0.1,
        stability_windows=4,
        stability_gamma=1.0,
        mmr_lambda=0.75,
        redundancy_threshold=None,
        target_chunk_size=16,
        eps=1e-5,
        padding_mode="replicate",
        apply_sign=False,
        return_auxiliary=True,
        checkpoint_candidate_statistics=True,
    ):
        super().__init__()

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1.")
        if stability_windows < 1:
            raise ValueError("stability_windows must be at least 1.")
        if target_chunk_size < 1:
            raise ValueError("target_chunk_size must be at least 1.")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1].")
        if redundancy_threshold is not None and not (
            0.0 <= redundancy_threshold <= 1.0
        ):
            raise ValueError("redundancy_threshold must be in [0, 1].")
        if padding_mode not in {"replicate", "zero"}:
            raise ValueError("padding_mode must be either 'replicate' or 'zero'.")
        if raw_weight < 0 or diff_weight < 0:
            raise ValueError("raw_weight and diff_weight must be non-negative.")
        weight_sum = raw_weight + diff_weight
        if weight_sum <= 0:
            raise ValueError("raw_weight and diff_weight cannot both be zero.")
        normalized_weights = torch.tensor(
            [raw_weight / weight_sum, diff_weight / weight_sum],
            dtype=torch.float32,
        )

        self.top_k = top_k
        self.max_lag = max_lag
        self.candidate_multiplier = candidate_multiplier
        self.correlation_weight_logits = nn.Parameter(
            normalized_weights.clamp_min(eps).log()
        )
        self.lag_penalty = lag_penalty
        self.stability_windows = stability_windows
        self.stability_gamma = stability_gamma
        self.mmr_lambda = mmr_lambda
        self.redundancy_threshold = redundancy_threshold
        self.target_chunk_size = int(target_chunk_size)
        self.eps = eps
        self.padding_mode = padding_mode
        self.apply_sign = apply_sign
        self.return_auxiliary = return_auxiliary
        self.checkpoint_candidate_statistics = bool(checkpoint_candidate_statistics)

        # These tensors depend only on input shape/device.  Keeping them as
        # non-persistent buffers avoids rebuilding them for every target chunk
        # without changing checkpoints or selector semantics.
        self.register_buffer(
            "_diagonal_mask_cache",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_time_index_cache",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self._stability_boundaries_cache = {}

    def _correlation_weights(self):
        weights = torch.softmax(self.correlation_weight_logits, dim=0)
        return weights[0], weights[1]

    def _correlation(self, x, y, mask=None):
        if mask is None:
            x_mean = x.mean(dim=-1, keepdim=True)
            y_mean = y.mean(dim=-1, keepdim=True)
            x_centered = x - x_mean
            y_centered = y - y_mean
            covariance = (x_centered * y_centered).mean(dim=-1)
            x_var = x_centered.pow(2).mean(dim=-1)
            y_var = y_centered.pow(2).mean(dim=-1)
        else:
            weight = mask.to(dtype=x.dtype)
            count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
            x_mean = (x * weight).sum(dim=-1, keepdim=True) / count
            y_mean = (y * weight).sum(dim=-1, keepdim=True) / count
            x_centered = (x - x_mean) * weight
            y_centered = (y - y_mean) * weight
            covariance = (x_centered * y_centered).sum(dim=-1) / count.squeeze(-1)
            x_var = x_centered.pow(2).sum(dim=-1) / count.squeeze(-1)
            y_var = y_centered.pow(2).sum(dim=-1) / count.squeeze(-1)

        denominator = torch.sqrt(x_var * y_var + self.eps)
        return (covariance / denominator).clamp(-1.0, 1.0)

    def _coarse_similarity(self, x_channel, correlation_weights=None):
        if correlation_weights is None:
            correlation_weights = torch.stack(self._correlation_weights())
        raw_weight, diff_weight = correlation_weights.unbind()
        x_mean = x_channel.mean(dim=-1, keepdim=True)
        x_var = (x_channel - x_mean).pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x_channel - x_mean) / torch.sqrt(x_var + self.eps)
        raw_corr = torch.matmul(
            x_norm,
            x_norm.transpose(-1, -2),
        ) / x_norm.size(-1)
        raw_corr = raw_corr.clamp(-1.0, 1.0)

        x_diff = x_channel[..., 1:] - x_channel[..., :-1]
        diff_mean = x_diff.mean(dim=-1, keepdim=True)
        diff_var = (x_diff - diff_mean).pow(2).mean(dim=-1, keepdim=True)
        diff_norm = (x_diff - diff_mean) / torch.sqrt(diff_var + self.eps)
        diff_corr = torch.matmul(
            diff_norm,
            diff_norm.transpose(-1, -2),
        ) / diff_norm.size(-1)
        diff_corr = diff_corr.clamp(-1.0, 1.0)
        similarity = (
            raw_weight * raw_corr.abs() + diff_weight * diff_corr.abs()
        ).clamp(0.0, 1.0)
        return similarity, raw_corr, diff_corr

    @staticmethod
    def _gather_channels(x_channel, indices):
        batch_size, num_channels, seq_len = x_channel.shape
        num_targets = indices.size(1)
        num_selected = indices.size(-1)
        batch_offsets = (
            torch.arange(batch_size, device=indices.device) * num_channels
        ).view(batch_size, 1, 1)
        flat_indices = (indices + batch_offsets).reshape(-1)
        return (
            x_channel.reshape(batch_size * num_channels, seq_len)
            .index_select(
                0,
                flat_indices,
            )
            .view(batch_size, num_targets, num_selected, seq_len)
        )

    def _diagonal_mask(self, num_channels, device):
        expected_shape = (1, num_channels, num_channels)
        if (
            self._diagonal_mask_cache.device != device
            or tuple(self._diagonal_mask_cache.shape) != expected_shape
        ):
            self._diagonal_mask_cache = torch.eye(
                num_channels,
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
        return self._diagonal_mask_cache

    def _time_index(self, seq_len, device):
        expected_shape = (1, 1, 1, seq_len)
        if (
            self._time_index_cache.device != device
            or tuple(self._time_index_cache.shape) != expected_shape
        ):
            self._time_index_cache = torch.arange(
                seq_len,
                device=device,
            ).view(expected_shape)
        return self._time_index_cache

    def _stability_boundaries(self, seq_len, num_windows, device):
        key = (seq_len, num_windows, device.type, device.index)
        boundaries = self._stability_boundaries_cache.get(key)
        if boundaries is None:
            boundaries = tuple(
                torch.linspace(
                    0,
                    seq_len,
                    steps=num_windows + 1,
                    device=device,
                )
                .round()
                .to(torch.long)
                .tolist()
            )
            self._stability_boundaries_cache[key] = boundaries
        return boundaries

    def _lag_scores(
        self,
        target,
        candidates,
        start,
        end,
        apply_lag_penalty=True,
        correlation_weights=None,
    ):
        if correlation_weights is None:
            correlation_weights = torch.stack(self._correlation_weights())
        raw_weight, diff_weight = correlation_weights.unbind()
        scores = []
        raw_correlations = []
        diff_correlations = []

        for lag in range(self.max_lag + 1):
            target_segment = target[..., start + lag : end].unsqueeze(2)
            candidate_end = end - lag if lag > 0 else end
            candidate_segment = candidates[..., start:candidate_end]

            raw_corr = self._correlation(target_segment, candidate_segment)
            target_diff = target_segment[..., 1:] - target_segment[..., :-1]
            candidate_diff = candidate_segment[..., 1:] - candidate_segment[..., :-1]
            diff_corr = self._correlation(target_diff, candidate_diff)

            score = raw_weight * raw_corr.abs() + diff_weight * diff_corr.abs()
            if apply_lag_penalty and self.max_lag > 0:
                score = score - self.lag_penalty * lag / self.max_lag

            scores.append(score.clamp_min(0.0))
            raw_correlations.append(raw_corr)
            diff_correlations.append(diff_corr)

        return (
            torch.stack(scores, dim=-1),
            torch.stack(raw_correlations, dim=-1),
            torch.stack(diff_correlations, dim=-1),
        )

    def _lag_stability(self, target, candidates, correlation_weights=None):
        seq_len = target.size(-1)
        minimum_window = self.max_lag + 2
        num_windows = min(
            self.stability_windows,
            max(1, seq_len // minimum_window),
        )
        if num_windows == 1:
            return torch.ones(
                candidates.shape[:-1],
                dtype=candidates.dtype,
                device=candidates.device,
            )

        boundaries = self._stability_boundaries(
            seq_len,
            num_windows,
            target.device,
        )
        window_lags = []
        for window_index in range(num_windows):
            start = boundaries[window_index]
            end = boundaries[window_index + 1]
            scores, _, _ = self._lag_scores(
                target,
                candidates,
                start,
                end,
                apply_lag_penalty=False,
                correlation_weights=correlation_weights,
            )
            window_lags.append(scores.argmax(dim=-1))

        lag_variance = (
            torch.stack(window_lags, dim=-1)
            .float()
            .var(
                dim=-1,
                unbiased=False,
            )
        )
        lag_variance = lag_variance / max(
            float(self.max_lag**2),
            1.0,
        )
        return torch.exp(-self.stability_gamma * lag_variance)

    def _causal_align(self, candidates, lags):
        seq_len = candidates.size(-1)
        time_index = self._time_index(seq_len, candidates.device)
        source_index = time_index - lags.unsqueeze(-1)
        alignment_mask = source_index >= 0
        source_index = source_index.clamp_min(0).expand_as(candidates)
        aligned = torch.gather(candidates, dim=-1, index=source_index)

        if self.padding_mode == "zero":
            aligned = aligned.masked_fill(~alignment_mask, 0.0)

        return aligned, alignment_mask.expand_as(candidates)

    def _aligned_redundancy(
        self,
        aligned_candidates,
        alignment_masks,
        selected_positions,
        correlation_weights=None,
    ):
        if correlation_weights is None:
            correlation_weights = torch.stack(self._correlation_weights())
        raw_weight, diff_weight = correlation_weights.unbind()
        seq_len = aligned_candidates.size(-1)
        gather_index = selected_positions[..., None, None].expand(
            *selected_positions.shape,
            1,
            seq_len,
        )
        selected = torch.gather(
            aligned_candidates,
            dim=2,
            index=gather_index,
        ).expand_as(aligned_candidates)
        selected_mask = torch.gather(
            alignment_masks,
            dim=2,
            index=gather_index,
        ).expand_as(alignment_masks)

        joint_mask = alignment_masks & selected_mask
        raw_corr = self._correlation(
            aligned_candidates,
            selected,
            joint_mask,
        )

        candidate_diff = aligned_candidates[..., 1:] - aligned_candidates[..., :-1]
        selected_diff = selected[..., 1:] - selected[..., :-1]
        diff_mask = joint_mask[..., 1:] & joint_mask[..., :-1]
        diff_corr = self._correlation(
            candidate_diff,
            selected_diff,
            diff_mask,
        )
        return (raw_weight * raw_corr.abs() + diff_weight * diff_corr.abs()).clamp(
            0.0, 1.0
        )

    def _mmr_select(
        self,
        relevance,
        aligned_candidates,
        alignment_masks,
        correlation_weights=None,
    ):
        available = torch.ones_like(relevance, dtype=torch.bool)
        selected_positions = []
        selected_redundancy = []
        selected_mmr_scores = []
        threshold_relaxed = []
        max_redundancy = torch.zeros_like(relevance)

        position = relevance.argmax(dim=-1)
        selected_positions.append(position)
        selected_redundancy.append(torch.zeros_like(position, dtype=relevance.dtype))
        selected_mmr_scores.append(
            self.mmr_lambda
            * torch.gather(relevance, dim=-1, index=position.unsqueeze(-1)).squeeze(-1)
        )
        threshold_relaxed.append(torch.zeros_like(position, dtype=torch.bool))

        for _ in range(1, self.top_k):
            available.scatter_(dim=-1, index=position.unsqueeze(-1), value=False)
            redundancy = self._aligned_redundancy(
                aligned_candidates,
                alignment_masks,
                position,
                correlation_weights=correlation_weights,
            )
            max_redundancy = torch.maximum(max_redundancy, redundancy)
            mmr_score = (
                self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * max_redundancy
            )
            unconstrained_score = mmr_score.masked_fill(
                ~available,
                float("-inf"),
            )

            if self.redundancy_threshold is not None:
                constrained_score = unconstrained_score.masked_fill(
                    max_redundancy > self.redundancy_threshold,
                    float("-inf"),
                )
                has_candidate = torch.isfinite(constrained_score).any(
                    dim=-1,
                    keepdim=True,
                )
                relaxed = ~has_candidate.squeeze(-1)
                selection_score = torch.where(
                    has_candidate,
                    constrained_score,
                    unconstrained_score,
                )
            else:
                selection_score = unconstrained_score
                relaxed = torch.zeros_like(position, dtype=torch.bool)

            position = selection_score.argmax(dim=-1)
            selected_positions.append(position)
            selected_redundancy.append(
                torch.gather(
                    max_redundancy,
                    dim=-1,
                    index=position.unsqueeze(-1),
                ).squeeze(-1)
            )
            selected_mmr_scores.append(
                torch.gather(
                    mmr_score,
                    dim=-1,
                    index=position.unsqueeze(-1),
                ).squeeze(-1)
            )
            threshold_relaxed.append(relaxed)

        return (
            torch.stack(selected_positions, dim=-1),
            torch.stack(selected_redundancy, dim=-1),
            torch.stack(selected_mmr_scores, dim=-1),
            torch.stack(threshold_relaxed, dim=-1),
        )

    @staticmethod
    def _gather_candidate_values(values, positions):
        return torch.gather(values, dim=-1, index=positions)

    @staticmethod
    def _gather_candidate_sequences(values, positions):
        seq_len = values.size(-1)
        gather_index = positions.unsqueeze(-1).expand(
            *positions.shape,
            seq_len,
        )
        return torch.gather(values, dim=2, index=gather_index)

    def forward(self, x, dynamic_weights=None):
        if x.ndim != 3:
            raise ValueError("Expected x with shape [B, L, N].")

        seq_len = x.size(1)
        num_channels = x.size(2)
        if self.top_k >= num_channels:
            raise ValueError("top_k must be smaller than the number of channels.")
        if self.max_lag > seq_len - 2:
            raise ValueError("max_lag must leave at least two aligned time steps.")

        x_channel = x.permute(0, 2, 1)
        correlation_weights = torch.stack(self._correlation_weights())
        similarity, raw_corr, diff_corr = self._coarse_similarity(
            x_channel,
            correlation_weights=correlation_weights,
        )

        diagonal_mask = self._diagonal_mask(num_channels, x.device)
        coarse_score = similarity.masked_fill(
            diagonal_mask,
            float("-inf"),
        )
        if num_channels <= 64:
            num_candidates = num_channels - 1
        else:
            num_candidates = min(
                self.candidate_multiplier * self.top_k,
                num_channels - 1,
            )
        candidate_scores, candidate_indices = torch.topk(
            coarse_score,
            k=num_candidates,
            dim=-1,
            largest=True,
            sorted=True,
        )
        if dynamic_weights is not None and dynamic_weights.shape != (
            x.size(0),
            num_channels,
            3,
        ):
            raise ValueError("dynamic_weights must have shape [B, N, 3].")

        # Candidate sequences are the dominant ECL allocation. Process target
        # channels in bounded chunks and retain only scalar diagnostics plus
        # the selected channel indices/lags. The final Top-K sequences are
        # gathered once from x_channel after all discrete choices are known.
        chunk_outputs = {
            "best_lags": [],
            "lag_stability": [],
            "topk_indices": [],
            "topk_scores": [],
            "topk_lags": [],
            "topk_signs": [],
            "topk_lag_stability": [],
            "topk_redundancy": [],
            "topk_mmr_scores": [],
            "threshold_relaxed": [],
        }
        raw_weight, diff_weight = correlation_weights.unbind()
        max_candidate_targets = 0
        for target_start in range(0, num_channels, self.target_chunk_size):
            target_end = min(
                target_start + self.target_chunk_size,
                num_channels,
            )
            target_slice = slice(target_start, target_end)
            candidate_index_chunk = candidate_indices[:, target_slice]
            max_candidate_targets = max(
                max_candidate_targets,
                target_end - target_start,
            )

            def statistics_from_values(
                targets,
                candidate_values,
                weights,
            ):
                lag_scores, lag_raw_corr, lag_diff_corr = self._lag_scores(
                    targets,
                    candidate_values,
                    start=0,
                    end=seq_len,
                    apply_lag_penalty=False,
                    correlation_weights=weights,
                )
                best_lag_scores, best_lags = lag_scores.max(dim=-1)
                best_raw_corr = torch.gather(
                    lag_raw_corr,
                    dim=-1,
                    index=best_lags.unsqueeze(-1),
                ).squeeze(-1)
                best_diff_corr = torch.gather(
                    lag_diff_corr,
                    dim=-1,
                    index=best_lags.unsqueeze(-1),
                ).squeeze(-1)
                lag_stability = self._lag_stability(
                    targets,
                    candidate_values,
                    correlation_weights=weights,
                )
                return (
                    best_lag_scores,
                    best_lags,
                    best_raw_corr,
                    best_diff_corr,
                    lag_stability,
                )

            def candidate_statistics(
                source,
                selected_indices,
                weights,
                chunk_start=target_start,
                chunk_end=target_end,
            ):
                targets = source[:, chunk_start:chunk_end]
                candidate_values = self._gather_channels(
                    source,
                    selected_indices,
                )
                return statistics_from_values(
                    targets,
                    candidate_values,
                    weights,
                )

            if (
                self.checkpoint_candidate_statistics
                and self.training
                and torch.is_grad_enabled()
            ):
                candidate_values = None
                statistics = checkpoint(
                    candidate_statistics,
                    x_channel,
                    candidate_index_chunk,
                    correlation_weights,
                    use_reentrant=False,
                )
            else:
                targets = x_channel[:, target_slice]
                candidate_values = self._gather_channels(
                    x_channel,
                    candidate_index_chunk,
                )
                statistics = statistics_from_values(
                    targets,
                    candidate_values,
                    correlation_weights,
                )
            (
                best_lag_scores,
                best_lags,
                best_raw_corr,
                best_diff_corr,
                lag_stability,
            ) = statistics
            signed_relation = raw_weight * best_raw_corr + diff_weight * best_diff_corr
            candidate_signs = torch.where(
                signed_relation < 0,
                -torch.ones_like(signed_relation),
                torch.ones_like(signed_relation),
            )
            if dynamic_weights is None:
                stable_relevance = best_lag_scores * lag_stability
            else:
                synchronous_raw = torch.gather(
                    raw_corr[:, target_slice].abs(),
                    dim=-1,
                    index=candidate_index_chunk,
                )
                synchronous_diff = torch.gather(
                    diff_corr[:, target_slice].abs(),
                    dim=-1,
                    index=candidate_index_chunk,
                )
                lag_relation = torch.maximum(best_raw_corr.abs(), best_diff_corr.abs())
                weights = dynamic_weights[:, target_slice].unsqueeze(2)
                stable_relevance = (
                    weights[..., 0] * synchronous_raw
                    + weights[..., 1] * synchronous_diff
                    + weights[..., 2] * lag_relation
                ) * lag_stability

            with torch.no_grad():
                if candidate_values is None:
                    candidates = self._gather_channels(
                        x_channel,
                        candidate_index_chunk,
                    )
                else:
                    candidates = candidate_values
                aligned_candidates, candidate_alignment_masks = self._causal_align(
                    candidates, best_lags
                )
                (
                    selected_positions,
                    topk_redundancy,
                    topk_mmr_scores,
                    threshold_relaxed,
                ) = self._mmr_select(
                    stable_relevance.detach(),
                    aligned_candidates.detach(),
                    candidate_alignment_masks,
                    correlation_weights=correlation_weights,
                )

            chunk_outputs["best_lags"].append(best_lags)
            chunk_outputs["lag_stability"].append(lag_stability)
            chunk_outputs["topk_indices"].append(
                self._gather_candidate_values(candidate_index_chunk, selected_positions)
            )
            chunk_outputs["topk_scores"].append(
                self._gather_candidate_values(stable_relevance, selected_positions)
            )
            chunk_outputs["topk_lags"].append(
                self._gather_candidate_values(best_lags, selected_positions)
            )
            chunk_outputs["topk_signs"].append(
                self._gather_candidate_values(candidate_signs, selected_positions)
            )
            chunk_outputs["topk_lag_stability"].append(
                self._gather_candidate_values(lag_stability, selected_positions)
            )
            chunk_outputs["topk_redundancy"].append(topk_redundancy)
            chunk_outputs["topk_mmr_scores"].append(topk_mmr_scores)
            chunk_outputs["threshold_relaxed"].append(threshold_relaxed)

        chunk_outputs = {
            key: torch.cat(values, dim=1) for key, values in chunk_outputs.items()
        }
        best_lags = chunk_outputs["best_lags"]
        lag_stability = chunk_outputs["lag_stability"]
        topk_indices = chunk_outputs["topk_indices"]
        topk_scores = chunk_outputs["topk_scores"]
        topk_lags = chunk_outputs["topk_lags"]
        topk_signs = chunk_outputs["topk_signs"]
        topk_lag_stability = chunk_outputs["topk_lag_stability"]
        topk_redundancy = chunk_outputs["topk_redundancy"]
        topk_mmr_scores = chunk_outputs["topk_mmr_scores"]
        threshold_relaxed = chunk_outputs["threshold_relaxed"]

        result = {
            "similarity": similarity,
            "raw_correlation": raw_corr,
            "diff_correlation": diff_corr,
            "candidate_indices": candidate_indices,
            "candidate_scores": candidate_scores,
            "candidate_best_lags": best_lags,
            "candidate_lag_stability": lag_stability,
            "dynamic_weights": dynamic_weights,
            "correlation_weights": correlation_weights,
            "topk_indices": topk_indices,
            "topk_scores": topk_scores,
            "topk_lags": topk_lags,
            "topk_signs": topk_signs,
            "topk_lag_stability": topk_lag_stability,
            "topk_redundancy": topk_redundancy,
            "topk_mmr_scores": topk_mmr_scores,
            "threshold_relaxed": threshold_relaxed,
            "max_candidate_targets": max_candidate_targets,
        }
        if self.return_auxiliary:
            auxiliary_x = self._gather_channels(x_channel, topk_indices)
            aligned_auxiliary_x, alignment_mask = self._causal_align(
                auxiliary_x,
                topk_lags,
            )
            if self.apply_sign:
                aligned_auxiliary_x = aligned_auxiliary_x * topk_signs.unsqueeze(-1)
            result["auxiliary_x"] = auxiliary_x
            result["aligned_auxiliary_x"] = aligned_auxiliary_x
            result["alignment_mask"] = alignment_mask

        return result


class FastExactDirectionalLagMMRChannelSelector(DirectionalLagMMRChannelSelector):
    """Sparse three-stage implementation of legacy Directional Lag-MMR.

    High-dimensional inputs use a train-only global recall set of ``M``
    channels, current-window Raw/Diff screening to ``R``, and the unchanged
    exact lag, lag-stability, and sequence-MMR kernels to select final ``K``.
    Thus only global candidate recall is approximate. For
    ``N <= strict_legacy_threshold``, the original selector is reused directly
    so small-channel datasets keep their exact legacy behavior.
    """

    def __init__(
        self,
        top_k,
        max_lag=8,
        global_candidate_budget=64,
        refine_budget=32,
        candidate_chunk_size=32,
        raw_weight=0.5,
        diff_weight=0.5,
        lag_penalty=0.1,
        stability_windows=4,
        stability_gamma=1.0,
        mmr_lambda=0.75,
        redundancy_threshold=None,
        target_chunk_size=16,
        eps=1e-5,
        padding_mode="replicate",
        apply_sign=False,
        return_auxiliary=True,
        return_diagnostics=False,
        profile=False,
        strict_legacy_threshold=16,
    ):
        super().__init__(
            top_k=top_k,
            max_lag=max_lag,
            candidate_multiplier=max(
                1,
                (int(refine_budget) + int(top_k) - 1) // int(top_k),
            ),
            raw_weight=raw_weight,
            diff_weight=diff_weight,
            lag_penalty=lag_penalty,
            stability_windows=stability_windows,
            stability_gamma=stability_gamma,
            mmr_lambda=mmr_lambda,
            redundancy_threshold=redundancy_threshold,
            target_chunk_size=target_chunk_size,
            eps=eps,
            padding_mode=padding_mode,
            apply_sign=apply_sign,
            return_auxiliary=return_auxiliary,
            checkpoint_candidate_statistics=False,
        )
        if global_candidate_budget < refine_budget:
            raise ValueError(
                "global_candidate_budget (M) must be at least refine_budget (R)."
            )
        if refine_budget < top_k:
            raise ValueError("refine_budget (R) must be at least top_k (K).")
        if candidate_chunk_size < 1:
            raise ValueError("candidate_chunk_size must be positive.")
        if strict_legacy_threshold < 0:
            raise ValueError("strict_legacy_threshold must be non-negative.")
        self.global_candidate_budget = int(global_candidate_budget)
        self.refine_budget = int(refine_budget)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.return_diagnostics = bool(return_diagnostics)
        self.profile_enabled = bool(profile)
        self.strict_legacy_threshold = int(strict_legacy_threshold)
        self.register_buffer(
            "global_candidate_indices",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "global_candidate_scores",
            torch.empty(0),
            persistent=False,
        )
        self.last_profile = None
        # CUDA Graph warm-up populates these immutable shape/device masks.
        # Capture then reuses the resident CUDA tensors instead of attempting
        # a forbidden CPU-to-CUDA copy from Python boundary lists.
        self._region_masks_cache = {}

    @property
    def has_global_candidates(self):
        return self.global_candidate_indices.numel() > 0

    def set_global_candidate_graph(self, indices, scores):
        """Install the train-only ``[N, M]`` first-stage candidate graph."""
        indices = torch.as_tensor(indices, dtype=torch.long)
        scores = torch.as_tensor(scores)
        if indices.ndim != 2 or scores.ndim != 2:
            raise ValueError("Global candidate indices and scores must be [N, M].")
        if indices.shape != scores.shape:
            raise ValueError("Global candidate indices and scores must match.")
        required = min(self.global_candidate_budget, indices.size(0) - 1)
        if indices.size(1) < required:
            raise ValueError("Global candidate graph contains fewer than M neighbors.")
        if torch.any(indices < 0) or torch.any(indices >= indices.size(0)):
            raise ValueError("Global candidate indices are out of range.")
        targets = torch.arange(indices.size(0)).unsqueeze(-1)
        if torch.any(indices.cpu() == targets):
            raise ValueError("Global candidate graph must exclude self channels.")
        if not torch.isfinite(scores).all():
            raise ValueError("Global candidate scores must be finite.")
        keep = required
        self.global_candidate_indices = (
            indices[:, :keep]
            .detach()
            .to(
                device=self.correlation_weight_logits.device,
            )
            .clone()
        )
        self.global_candidate_scores = (
            scores[:, :keep]
            .detach()
            .to(
                device=self.correlation_weight_logits.device,
                dtype=self.correlation_weight_logits.dtype,
            )
            .clamp(0.0, 1.0)
            .clone()
        )

    @staticmethod
    def _profile_clock(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        import time

        return time.perf_counter()

    @staticmethod
    def _masked_correlation_moments(target, lagged_candidates, masks, eps):
        """Correlate all lags and regions without materializing region x time."""
        weight = masks.to(dtype=target.dtype)
        count = weight.sum(dim=-1).clamp_min(1.0)
        target_sum = torch.einsum("btl,hrl->bthr", target, weight)
        target_square_sum = torch.einsum("btl,hrl->bthr", target.square(), weight)
        candidate_sum = torch.einsum("btmhl,hrl->btmhr", lagged_candidates, weight)
        candidate_square_sum = torch.einsum(
            "btmhl,hrl->btmhr", lagged_candidates.square(), weight
        )
        cross_sum = torch.einsum(
            "btmhl,btl,hrl->btmhr", lagged_candidates, target, weight
        )

        target_mean = target_sum / count
        candidate_mean = candidate_sum / count
        covariance = cross_sum / count - target_mean.unsqueeze(2) * candidate_mean
        target_variance = (target_square_sum / count - target_mean.square()).clamp_min(
            0.0
        )
        candidate_variance = (
            candidate_square_sum / count - candidate_mean.square()
        ).clamp_min(0.0)
        denominator = torch.sqrt(
            target_variance.unsqueeze(2) * candidate_variance + eps
        )
        return (covariance / denominator).clamp(-1.0, 1.0)

    def _lagged_exact_views(self, target, candidates):
        seq_len = target.size(-1)
        lags = torch.arange(
            self.max_lag + 1,
            device=target.device,
        ).view(1, 1, 1, -1, 1)
        times = torch.arange(seq_len, device=target.device).view(1, 1, 1, 1, -1)
        source_index = (
            (times - lags)
            .clamp_min(0)
            .expand(
                candidates.size(0),
                candidates.size(1),
                candidates.size(2),
                self.max_lag + 1,
                seq_len,
            )
        )
        lagged = torch.gather(
            candidates.unsqueeze(3).expand(-1, -1, -1, self.max_lag + 1, -1),
            dim=-1,
            index=source_index,
        )
        return lagged

    def _score_exact_regions(
        self,
        target,
        lagged_candidates,
        raw_masks,
        correlation_weights,
        target_diff=None,
        candidate_diff=None,
    ):
        raw_weight, diff_weight = correlation_weights.unbind()
        raw_corr = self._masked_correlation_moments(
            target,
            lagged_candidates,
            raw_masks,
            self.eps,
        )
        if target_diff is None:
            target_diff = target[..., 1:] - target[..., :-1]
        if candidate_diff is None:
            candidate_diff = lagged_candidates[..., 1:] - lagged_candidates[..., :-1]
        diff_masks = raw_masks[..., 1:] & raw_masks[..., :-1]
        diff_corr = self._masked_correlation_moments(
            target_diff,
            candidate_diff,
            diff_masks,
            self.eps,
        )
        scores = (
            raw_weight * raw_corr.abs() + diff_weight * diff_corr.abs()
        ).clamp_min(0.0)
        return scores, raw_corr, diff_corr

    def _region_masks(self, seq_len, starts, ends, device):
        starts = tuple(int(value) for value in starts)
        ends = tuple(int(value) for value in ends)
        key = (
            int(seq_len),
            starts,
            ends,
            device.type,
            device.index,
        )
        cached = self._region_masks_cache.get(key)
        if cached is not None:
            return cached
        lags = torch.arange(self.max_lag + 1, device=device).view(-1, 1, 1)
        times = torch.arange(seq_len, device=device).view(1, 1, -1)
        starts = torch.as_tensor(starts, device=device).view(1, -1, 1)
        ends = torch.as_tensor(ends, device=device).view(1, -1, 1)
        masks = (times >= starts + lags) & (times < ends)
        self._region_masks_cache[key] = masks
        return masks

    def _vectorized_lag_scores_exact(
        self,
        target,
        candidates,
        start=0,
        end=None,
        correlation_weights=None,
    ):
        """Legacy-equivalent scores for every lag, with no lag penalty."""
        if correlation_weights is None:
            correlation_weights = torch.stack(self._correlation_weights())
        if end is None:
            end = target.size(-1)
        lagged = self._lagged_exact_views(target, candidates)
        masks = self._region_masks(target.size(-1), [start], [end], target.device)
        values = self._score_exact_regions(
            target,
            lagged,
            masks,
            correlation_weights,
        )
        return tuple(value[..., 0] for value in values)

    def _vectorized_lag_statistics_exact(
        self,
        target,
        candidates,
        correlation_weights=None,
        profile=False,
    ):
        """Compute full lag scores and the legacy multi-window stability."""
        if correlation_weights is None:
            correlation_weights = torch.stack(self._correlation_weights())
        lag_start = self._profile_clock(target.device) if profile else None
        lagged = self._lagged_exact_views(target, candidates)
        target_diff = target[..., 1:] - target[..., :-1]
        candidate_diff = lagged[..., 1:] - lagged[..., :-1]
        full_masks = self._region_masks(
            target.size(-1), [0], [target.size(-1)], target.device
        )
        full = self._score_exact_regions(
            target,
            lagged,
            full_masks,
            correlation_weights,
            target_diff=target_diff,
            candidate_diff=candidate_diff,
        )
        lag_end = self._profile_clock(target.device) if profile else None

        minimum_window = self.max_lag + 2
        num_windows = min(
            self.stability_windows,
            max(1, target.size(-1) // minimum_window),
        )
        if num_windows == 1:
            lag_stability = torch.ones_like(full[0][..., 0, 0])
        else:
            boundaries = self._stability_boundaries(
                target.size(-1), num_windows, target.device
            )
            window_masks = self._region_masks(
                target.size(-1),
                boundaries[:-1],
                boundaries[1:],
                target.device,
            )
            window_scores, _, _ = self._score_exact_regions(
                target,
                lagged,
                window_masks,
                correlation_weights,
                target_diff=target_diff,
                candidate_diff=candidate_diff,
            )
            window_lags = window_scores.argmax(dim=-2)
            lag_variance = window_lags.float().var(
                dim=-1,
                unbiased=False,
            )
            lag_variance = lag_variance / max(
                float(self.max_lag**2),
                1.0,
            )
            lag_stability = torch.exp(-self.stability_gamma * lag_variance)
        stability_end = self._profile_clock(target.device) if profile else None
        result = {
            "scores": full[0][..., 0],
            "raw_corr": full[1][..., 0],
            "diff_corr": full[2][..., 0],
            "lag_stability": lag_stability,
        }
        if profile:
            result["_profile"] = {
                "lag_full_ms": (lag_end - lag_start) * 1000.0,
                "stability_ms": (stability_end - lag_end) * 1000.0,
            }
        return result

    def _forward_exact_all_legacy_search_fp32(
        self,
        x,
        dynamic_weights,
        do_profile,
    ):
        """Exact legacy full-candidate search plus selected-only gradients."""
        total_start = self._profile_clock(x.device) if do_profile else None
        search_dynamic = None if dynamic_weights is None else dynamic_weights.detach()
        with torch.no_grad():
            result = DirectionalLagMMRChannelSelector.forward(
                self,
                x.detach(),
                search_dynamic,
            )

        x_channel = x.permute(0, 2, 1).float()
        correlation_weights = torch.stack(self._correlation_weights()).float()
        result["topk_scores"] = self._recompute_selected_relevance(
            x_channel,
            result["topk_indices"],
            result["topk_lags"],
            result["topk_lag_stability"],
            None if dynamic_weights is None else dynamic_weights.float(),
            correlation_weights,
        )
        result["dynamic_weights"] = dynamic_weights
        result["correlation_weights"] = correlation_weights
        result["used_global_rescue"] = False
        result["strict_legacy_path"] = True
        result["exact_all_legacy_search"] = True
        result["sparse_exact_path"] = False
        if not self.return_diagnostics:
            result.pop("similarity", None)
            result.pop("raw_correlation", None)
            result.pop("diff_correlation", None)
        if self.return_auxiliary:
            auxiliary_x = self._gather_channels(x_channel, result["topk_indices"])
            aligned_auxiliary_x, alignment_mask = self._causal_align(
                auxiliary_x, result["topk_lags"]
            )
            if self.apply_sign:
                aligned_auxiliary_x = aligned_auxiliary_x * result[
                    "topk_signs"
                ].unsqueeze(-1)
            result["auxiliary_x"] = auxiliary_x
            result["aligned_auxiliary_x"] = aligned_auxiliary_x
            result["alignment_mask"] = alignment_mask
        if do_profile:
            total_ms = (self._profile_clock(x.device) - total_start) * 1000.0
            self.last_profile = {
                "coarse_ms": 0.0,
                "candidate_merge_ms": 0.0,
                "lag_full_ms": 0.0,
                "stability_ms": 0.0,
                "alignment_ms": 0.0,
                "mmr_ms": 0.0,
                "selected_recompute_ms": 0.0,
                "auxiliary_gather_ms": 0.0,
                "total_ms": total_ms,
            }
        return result

    def _candidate_statistics_exact(
        self,
        target,
        candidates,
        correlation_weights,
        profile=False,
    ):
        outputs = {
            "scores": [],
            "raw_corr": [],
            "diff_corr": [],
            "lag_stability": [],
        }
        profile_totals = {"lag_full_ms": 0.0, "stability_ms": 0.0}
        for start in range(0, candidates.size(2), self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, candidates.size(2))
            chunk = self._vectorized_lag_statistics_exact(
                target,
                candidates[:, :, start:end],
                correlation_weights=correlation_weights,
                profile=profile,
            )
            for key in outputs:
                outputs[key].append(chunk[key])
            if profile:
                for key in profile_totals:
                    profile_totals[key] += chunk["_profile"][key]
        outputs = {key: torch.cat(values, dim=2) for key, values in outputs.items()}
        if profile:
            outputs["_profile"] = profile_totals
        return outputs

    def _mmr_select_exact_cached(
        self,
        relevance,
        aligned_candidates,
        alignment_masks,
        candidate_lags,
        correlation_weights=None,
    ):
        """Legacy sequence MMR with an exact cached redundancy matrix."""
        if correlation_weights is None:
            correlation_weights = torch.stack(self._correlation_weights())
        raw_weight, diff_weight = correlation_weights.unbind()
        candidate_diff = aligned_candidates[..., 1:] - aligned_candidates[..., :-1]
        joint_start = torch.maximum(
            candidate_lags.unsqueeze(-1),
            candidate_lags.unsqueeze(-2),
        )
        redundancy_matrix = torch.zeros(
            *candidate_lags.shape,
            candidate_lags.size(-1),
            device=aligned_candidates.device,
            dtype=aligned_candidates.dtype,
        )

        def suffix_correlation(values, start):
            segment = values[..., start:]
            count = float(segment.size(-1))
            mean = segment.mean(dim=-1)
            variance = (segment.square().mean(dim=-1) - mean.square()).clamp_min(0.0)
            cross = (
                torch.matmul(
                    segment,
                    segment.transpose(-1, -2),
                )
                / count
            )
            covariance = cross - mean.unsqueeze(-1) * mean.unsqueeze(-2)
            denominator = torch.sqrt(
                variance.unsqueeze(-1) * variance.unsqueeze(-2) + self.eps
            )
            return (covariance / denominator).clamp(-1.0, 1.0)

        # Joint validity is always a suffix beginning at max(lag_i, lag_j).
        # Computing each possible suffix matrix once replaces 49 repeated
        # selected-vs-all sequence reductions without changing MMR semantics.
        for start in range(self.max_lag + 1):
            raw_corr = suffix_correlation(aligned_candidates, start)
            diff_corr = suffix_correlation(candidate_diff, start)
            redundancy = (
                raw_weight * raw_corr.abs() + diff_weight * diff_corr.abs()
            ).clamp(0.0, 1.0)
            redundancy_matrix = torch.where(
                joint_start == start,
                redundancy,
                redundancy_matrix,
            )
        available = torch.ones_like(relevance, dtype=torch.bool)
        max_redundancy = torch.zeros_like(relevance)
        selected_positions = []
        selected_redundancy = []
        selected_mmr_scores = []
        threshold_relaxed = []

        position = relevance.argmax(dim=-1)
        selected_positions.append(position)
        selected_redundancy.append(torch.zeros_like(relevance[..., 0]))
        selected_mmr_scores.append(
            self.mmr_lambda
            * torch.gather(relevance, -1, position.unsqueeze(-1)).squeeze(-1)
        )
        threshold_relaxed.append(torch.zeros_like(position, dtype=torch.bool))

        for _ in range(1, self.top_k):
            available.scatter_(-1, position.unsqueeze(-1), False)
            redundancy = torch.gather(
                redundancy_matrix,
                dim=2,
                index=position[..., None, None].expand(
                    *position.shape,
                    1,
                    relevance.size(-1),
                ),
            ).squeeze(2)
            max_redundancy = torch.maximum(max_redundancy, redundancy)
            mmr_score = (
                self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * max_redundancy
            )
            unconstrained = mmr_score.masked_fill(~available, float("-inf"))
            if self.redundancy_threshold is None:
                selection_score = unconstrained
                relaxed = torch.zeros_like(position, dtype=torch.bool)
            else:
                constrained = unconstrained.masked_fill(
                    max_redundancy > self.redundancy_threshold,
                    float("-inf"),
                )
                has_candidate = torch.isfinite(constrained).any(
                    dim=-1,
                    keepdim=True,
                )
                selection_score = torch.where(
                    has_candidate,
                    constrained,
                    unconstrained,
                )
                relaxed = ~has_candidate.squeeze(-1)
            position = selection_score.argmax(dim=-1)
            selected_positions.append(position)
            selected_redundancy.append(
                torch.gather(max_redundancy, -1, position.unsqueeze(-1)).squeeze(-1)
            )
            selected_mmr_scores.append(
                torch.gather(mmr_score, -1, position.unsqueeze(-1)).squeeze(-1)
            )
            threshold_relaxed.append(relaxed)

        return (
            torch.stack(selected_positions, dim=-1),
            torch.stack(selected_redundancy, dim=-1),
            torch.stack(selected_mmr_scores, dim=-1),
            torch.stack(threshold_relaxed, dim=-1),
        )

    def _sparse_coarse_similarity(
        self,
        targets,
        candidates,
        correlation_weights=None,
    ):
        """Raw/Diff similarity for only the train-recalled M candidates."""
        if correlation_weights is None:
            correlation_weights = torch.stack(self._correlation_weights())
        raw_weight, diff_weight = correlation_weights.unbind()

        def normalized(values):
            centered = values - values.mean(dim=-1, keepdim=True)
            variance = centered.square().mean(dim=-1, keepdim=True)
            return centered / torch.sqrt(variance + self.eps)

        target_raw = normalized(targets).unsqueeze(2)
        candidate_raw = normalized(candidates)
        raw_corr = (target_raw * candidate_raw).mean(dim=-1).clamp(-1.0, 1.0)

        target_diff = targets[..., 1:] - targets[..., :-1]
        candidate_diff = candidates[..., 1:] - candidates[..., :-1]
        diff_corr = (
            (normalized(target_diff).unsqueeze(2) * normalized(candidate_diff))
            .mean(dim=-1)
            .clamp(-1.0, 1.0)
        )
        similarity = (
            raw_weight * raw_corr.abs() + diff_weight * diff_corr.abs()
        ).clamp(0.0, 1.0)
        return similarity, raw_corr, diff_corr

    def _global_candidates_for_input(self, num_channels, device):
        """Return the fixed M candidates, constructing exact all-others if needed."""
        keep = min(self.global_candidate_budget, num_channels - 1)
        if keep < self.refine_budget:
            raise ValueError(
                "The input has fewer available channels than refine_budget."
            )
        if self.has_global_candidates:
            if self.global_candidate_indices.size(0) != num_channels:
                raise ValueError(
                    "Global candidate graph channel count does not match input."
                )
            return self.global_candidate_indices[:, :keep].to(device=device)
        if keep != num_channels - 1:
            raise RuntimeError(
                "Sparse Exact high-dimensional selection requires the "
                "train-only global candidate graph before forward."
            )
        all_channels = torch.arange(num_channels, device=device)
        matrix = all_channels.view(1, -1).expand(num_channels, -1)
        targets = all_channels.view(-1, 1)
        return matrix[matrix != targets].view(num_channels, num_channels - 1)

    def _refine_global_candidates(self, x_channel, correlation_weights):
        """Current-window Raw/Diff refinement from global M down to R."""
        batch_size, num_channels, _ = x_channel.shape
        global_indices = self._global_candidates_for_input(
            num_channels,
            x_channel.device,
        )
        global_batch = global_indices.unsqueeze(0).expand(batch_size, -1, -1)
        global_values = self._gather_channels(x_channel, global_batch)
        similarity, raw_corr, diff_corr = self._sparse_coarse_similarity(
            x_channel,
            global_values,
            correlation_weights,
        )
        keep = min(self.refine_budget, global_indices.size(-1))
        candidate_scores, positions = torch.topk(
            similarity,
            k=keep,
            dim=-1,
            largest=True,
            sorted=True,
        )
        candidate_indices = torch.gather(global_batch, -1, positions)
        return {
            "global_indices": global_indices,
            "candidate_indices": candidate_indices,
            "candidate_scores": candidate_scores,
            "similarity": similarity,
            "raw_corr": torch.gather(raw_corr, -1, positions),
            "diff_corr": torch.gather(diff_corr, -1, positions),
        }

    def _recompute_selected_relevance(
        self,
        x_channel,
        topk_indices,
        topk_lags,
        topk_lag_stability,
        dynamic_weights,
        correlation_weights,
    ):
        selected = self._gather_channels(x_channel, topk_indices)
        target = x_channel.unsqueeze(2)

        def normalized(values):
            centered = values - values.mean(dim=-1, keepdim=True)
            variance = centered.square().mean(dim=-1, keepdim=True)
            return centered / torch.sqrt(variance + self.eps)

        target_raw_norm = normalized(target)
        selected_raw_norm = normalized(selected)
        synchronous_raw = (
            (target_raw_norm * selected_raw_norm).mean(dim=-1).clamp(-1.0, 1.0)
        )
        target_diff = target[..., 1:] - target[..., :-1]
        selected_diff = selected[..., 1:] - selected[..., :-1]
        synchronous_diff = (
            (normalized(target_diff) * normalized(selected_diff))
            .mean(dim=-1)
            .clamp(-1.0, 1.0)
        )

        aligned, alignment_mask = self._causal_align(selected, topk_lags)
        fixed_raw = self._correlation(target, aligned, alignment_mask)
        aligned_diff = aligned[..., 1:] - aligned[..., :-1]
        diff_mask = alignment_mask[..., 1:] & alignment_mask[..., :-1]
        fixed_diff = self._correlation(
            target_diff,
            aligned_diff,
            diff_mask,
        )
        stability = topk_lag_stability.detach()
        if dynamic_weights is None:
            raw_weight, diff_weight = correlation_weights.unbind()
            return (
                raw_weight * fixed_raw.abs() + diff_weight * fixed_diff.abs()
            ) * stability
        weights = dynamic_weights.unsqueeze(2)
        lag_relation = torch.maximum(fixed_raw.abs(), fixed_diff.abs())
        return (
            weights[..., 0] * synchronous_raw.abs()
            + weights[..., 1] * synchronous_diff.abs()
            + weights[..., 2] * lag_relation
        ) * stability

    def forward(self, x, dynamic_weights=None, profile=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_fp32(x, dynamic_weights, profile)

    def _forward_strict_legacy_fp32(
        self,
        x,
        x_channel,
        dynamic_weights,
        correlation_weights,
        do_profile,
    ):
        """Bitwise legacy small-N path without checkpoint recomputation.

        The full differentiable candidate tensors retain the legacy CUDA
        reduction and gradient order.  The speedup comes solely from keeping
        those bounded small-N activations instead of recomputing them during
        backward.  Discrete MMR remains under ``no_grad`` exactly as before.
        """
        import time

        total_start = time.perf_counter() if do_profile else None
        _, num_channels, seq_len = x_channel.shape
        search_weights = correlation_weights.detach()
        similarity, raw_full, diff_full = self._coarse_similarity(
            x_channel,
            correlation_weights=correlation_weights,
        )
        coarse_score = similarity.masked_fill(
            self._diagonal_mask(num_channels, x.device),
            float("-inf"),
        )
        candidate_scores, candidate_indices = torch.topk(
            coarse_score,
            k=num_channels - 1,
            dim=-1,
            largest=True,
            sorted=True,
        )

        chunk_outputs = {
            "candidate_best_lags": [],
            "candidate_lag_stability": [],
            "topk_indices": [],
            "topk_scores": [],
            "topk_lags": [],
            "topk_signs": [],
            "topk_lag_stability": [],
            "topk_redundancy": [],
            "topk_mmr_scores": [],
            "threshold_relaxed": [],
        }
        raw_weight, diff_weight = correlation_weights.unbind()
        max_candidate_targets = 0
        for target_start in range(0, num_channels, self.target_chunk_size):
            target_end = min(
                target_start + self.target_chunk_size,
                num_channels,
            )
            target_slice = slice(target_start, target_end)
            max_candidate_targets = max(
                max_candidate_targets,
                target_end - target_start,
            )
            indices = candidate_indices[:, target_slice]
            targets = x_channel[:, target_slice]
            candidates = self._gather_channels(x_channel, indices)
            lag_scores, lag_raw, lag_diff = self._lag_scores(
                targets,
                candidates,
                start=0,
                end=seq_len,
                apply_lag_penalty=False,
                correlation_weights=correlation_weights,
            )
            best_lag_scores, best_lags = lag_scores.max(dim=-1)
            best_raw = torch.gather(lag_raw, -1, best_lags.unsqueeze(-1)).squeeze(-1)
            best_diff = torch.gather(lag_diff, -1, best_lags.unsqueeze(-1)).squeeze(-1)
            lag_stability = self._lag_stability(
                targets,
                candidates,
                correlation_weights=correlation_weights,
            )
            signed_relation = raw_weight * best_raw + diff_weight * best_diff
            candidate_signs = torch.where(
                signed_relation < 0,
                -torch.ones_like(signed_relation),
                torch.ones_like(signed_relation),
            )
            if dynamic_weights is None:
                relevance = best_lag_scores * lag_stability
            else:
                synchronous_raw = torch.gather(
                    raw_full[:, target_slice].abs(), -1, indices
                )
                synchronous_diff = torch.gather(
                    diff_full[:, target_slice].abs(), -1, indices
                )
                weights = dynamic_weights.float()[:, target_slice].unsqueeze(2)
                relevance = (
                    weights[..., 0] * synchronous_raw
                    + weights[..., 1] * synchronous_diff
                    + weights[..., 2] * torch.maximum(best_raw.abs(), best_diff.abs())
                ) * lag_stability

            with torch.no_grad():
                aligned, masks = self._causal_align(candidates.detach(), best_lags)
                (
                    selected_positions,
                    topk_redundancy,
                    topk_mmr_scores,
                    threshold_relaxed,
                ) = self._mmr_select(
                    relevance.detach(),
                    aligned,
                    masks,
                    correlation_weights=search_weights,
                )
            gather = self._gather_candidate_values
            chunk_outputs["candidate_best_lags"].append(best_lags)
            chunk_outputs["candidate_lag_stability"].append(lag_stability)
            chunk_outputs["topk_indices"].append(gather(indices, selected_positions))
            chunk_outputs["topk_scores"].append(gather(relevance, selected_positions))
            chunk_outputs["topk_lags"].append(gather(best_lags, selected_positions))
            chunk_outputs["topk_signs"].append(
                gather(candidate_signs, selected_positions)
            )
            chunk_outputs["topk_lag_stability"].append(
                gather(lag_stability, selected_positions)
            )
            chunk_outputs["topk_redundancy"].append(topk_redundancy)
            chunk_outputs["topk_mmr_scores"].append(topk_mmr_scores)
            chunk_outputs["threshold_relaxed"].append(threshold_relaxed)

        chunk_outputs = {
            key: torch.cat(values, dim=1) for key, values in chunk_outputs.items()
        }

        result = {
            "candidate_indices": candidate_indices,
            "candidate_scores": candidate_scores,
            "candidate_best_lags": chunk_outputs["candidate_best_lags"],
            "candidate_lag_stability": (chunk_outputs["candidate_lag_stability"]),
            "dynamic_weights": dynamic_weights,
            "correlation_weights": correlation_weights,
            "used_global_rescue": False,
            "topk_indices": chunk_outputs["topk_indices"],
            "topk_scores": chunk_outputs["topk_scores"],
            "topk_lags": chunk_outputs["topk_lags"],
            "topk_signs": chunk_outputs["topk_signs"],
            "topk_lag_stability": chunk_outputs["topk_lag_stability"],
            "topk_redundancy": chunk_outputs["topk_redundancy"],
            "topk_mmr_scores": chunk_outputs["topk_mmr_scores"],
            "threshold_relaxed": chunk_outputs["threshold_relaxed"],
            "max_candidate_targets": max_candidate_targets,
            "strict_legacy_path": True,
        }
        if self.return_diagnostics:
            result["similarity"] = similarity
            result["raw_correlation"] = raw_full
            result["diff_correlation"] = diff_full
        if self.return_auxiliary:
            auxiliary_x = self._gather_channels(
                x_channel, chunk_outputs["topk_indices"]
            )
            aligned_auxiliary_x, alignment_mask = self._causal_align(
                auxiliary_x, chunk_outputs["topk_lags"]
            )
            if self.apply_sign:
                aligned_auxiliary_x = aligned_auxiliary_x * chunk_outputs[
                    "topk_signs"
                ].unsqueeze(-1)
            result["auxiliary_x"] = auxiliary_x
            result["aligned_auxiliary_x"] = aligned_auxiliary_x
            result["alignment_mask"] = alignment_mask
        if do_profile:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.last_profile = {
                "coarse_ms": 0.0,
                "candidate_merge_ms": 0.0,
                "lag_full_ms": 0.0,
                "stability_ms": 0.0,
                "alignment_ms": 0.0,
                "mmr_ms": 0.0,
                "selected_recompute_ms": 0.0,
                "auxiliary_gather_ms": 0.0,
                "total_ms": elapsed_ms,
            }
        return result

    def _forward_fp32(self, x, dynamic_weights=None, profile=None):
        if x.ndim != 3:
            raise ValueError("Expected x with shape [B, L, N].")
        seq_len, num_channels = x.size(1), x.size(2)
        if self.top_k >= num_channels:
            raise ValueError("top_k must be smaller than the number of channels.")
        if self.max_lag > seq_len - 2:
            raise ValueError("max_lag must leave at least two aligned time steps.")
        if dynamic_weights is not None and dynamic_weights.shape != (
            x.size(0),
            num_channels,
            3,
        ):
            raise ValueError("dynamic_weights must have shape [B, N, 3].")
        do_profile = self.profile_enabled if profile is None else bool(profile)
        if num_channels <= self.strict_legacy_threshold:
            total_start = self._profile_clock(x.device) if do_profile else None
            result = DirectionalLagMMRChannelSelector.forward(self, x, dynamic_weights)
            result["used_global_rescue"] = False
            result["strict_legacy_path"] = True
            result["sparse_exact_path"] = False
            if not self.return_diagnostics:
                result.pop("similarity", None)
                result.pop("raw_correlation", None)
                result.pop("diff_correlation", None)
            if do_profile:
                total_ms = (self._profile_clock(x.device) - total_start) * 1000.0
                self.last_profile = {
                    "coarse_ms": 0.0,
                    "candidate_merge_ms": 0.0,
                    "lag_full_ms": 0.0,
                    "stability_ms": 0.0,
                    "alignment_ms": 0.0,
                    "mmr_ms": 0.0,
                    "selected_recompute_ms": 0.0,
                    "auxiliary_gather_ms": 0.0,
                    "total_ms": total_ms,
                }
            return result
        exact_all_requested = (
            self.global_candidate_budget >= num_channels - 1
            and self.refine_budget >= num_channels - 1
        )
        if exact_all_requested:
            return self._forward_exact_all_legacy_search_fp32(
                x,
                dynamic_weights,
                do_profile,
            )
        x_channel = x.permute(0, 2, 1).float()
        correlation_weights = torch.stack(self._correlation_weights()).float()
        profile_values = {
            "coarse_ms": 0.0,
            "candidate_merge_ms": 0.0,
            "lag_full_ms": 0.0,
            "stability_ms": 0.0,
            "alignment_ms": 0.0,
            "mmr_ms": 0.0,
            "selected_recompute_ms": 0.0,
            "auxiliary_gather_ms": 0.0,
            "total_ms": 0.0,
        }
        total_start = self._profile_clock(x.device) if do_profile else None

        # Ranking stays in FP32 even when the caller is under autocast.
        search_weights = correlation_weights.detach()
        coarse_start = self._profile_clock(x.device) if do_profile else None
        with torch.no_grad():
            refined = self._refine_global_candidates(
                x_channel.detach(),
                search_weights,
            )
            similarity = refined["similarity"]
            raw_full = refined["raw_corr"]
            diff_full = refined["diff_corr"]
        if do_profile:
            coarse_end = self._profile_clock(x.device)
            profile_values["coarse_ms"] = (coarse_end - coarse_start) * 1000.0
        merge_start = self._profile_clock(x.device) if do_profile else None
        with torch.no_grad():
            candidate_indices = refined["candidate_indices"]
            candidate_scores = refined["candidate_scores"]
        if do_profile:
            merge_end = self._profile_clock(x.device)
            profile_values["candidate_merge_ms"] = (merge_end - merge_start) * 1000.0

        outputs = {
            "candidate_best_lags": [],
            "candidate_lag_stability": [],
            "topk_indices": [],
            "topk_lags": [],
            "topk_signs": [],
            "topk_lag_stability": [],
            "topk_redundancy": [],
            "topk_mmr_scores": [],
            "threshold_relaxed": [],
        }
        max_candidate_targets = 0
        with torch.no_grad():
            for target_start in range(0, num_channels, self.target_chunk_size):
                target_end = min(
                    target_start + self.target_chunk_size,
                    num_channels,
                )
                target_slice = slice(target_start, target_end)
                max_candidate_targets = max(
                    max_candidate_targets,
                    target_end - target_start,
                )
                indices = candidate_indices[:, target_slice]
                candidates = self._gather_channels(x_channel, indices)
                statistics = self._candidate_statistics_exact(
                    x_channel[:, target_slice],
                    candidates,
                    search_weights,
                    profile=do_profile,
                )
                if do_profile:
                    profile_values["lag_full_ms"] += statistics["_profile"][
                        "lag_full_ms"
                    ]
                    profile_values["stability_ms"] += statistics["_profile"][
                        "stability_ms"
                    ]
                lag_scores = statistics["scores"]
                best_lag_scores, best_lags = lag_scores.max(dim=-1)
                best_raw = torch.gather(
                    statistics["raw_corr"],
                    -1,
                    best_lags.unsqueeze(-1),
                ).squeeze(-1)
                best_diff = torch.gather(
                    statistics["diff_corr"],
                    -1,
                    best_lags.unsqueeze(-1),
                ).squeeze(-1)
                lag_stability = statistics["lag_stability"]
                signed_relation = (
                    search_weights[0] * best_raw + search_weights[1] * best_diff
                )
                candidate_signs = torch.where(
                    signed_relation < 0,
                    -torch.ones_like(signed_relation),
                    torch.ones_like(signed_relation),
                )
                if dynamic_weights is None:
                    stable_relevance = best_lag_scores * lag_stability
                else:
                    sync_raw = raw_full[:, target_slice].abs()
                    sync_diff = diff_full[:, target_slice].abs()
                    weights = (
                        dynamic_weights.detach().float()[:, target_slice].unsqueeze(2)
                    )
                    lag_relation = torch.maximum(best_raw.abs(), best_diff.abs())
                    stable_relevance = (
                        weights[..., 0] * sync_raw
                        + weights[..., 1] * sync_diff
                        + weights[..., 2] * lag_relation
                    ) * lag_stability

                alignment_start = self._profile_clock(x.device) if do_profile else None
                aligned, alignment_masks = self._causal_align(candidates, best_lags)
                if do_profile:
                    alignment_end = self._profile_clock(x.device)
                    profile_values["alignment_ms"] += (
                        alignment_end - alignment_start
                    ) * 1000.0
                mmr_start = self._profile_clock(x.device) if do_profile else None
                (
                    selected_positions,
                    topk_redundancy,
                    topk_mmr_scores,
                    threshold_relaxed,
                ) = self._mmr_select_exact_cached(
                    stable_relevance,
                    aligned,
                    alignment_masks,
                    best_lags,
                    correlation_weights=search_weights,
                )
                if do_profile:
                    mmr_end = self._profile_clock(x.device)
                    profile_values["mmr_ms"] += (mmr_end - mmr_start) * 1000.0

                gather = self._gather_candidate_values
                outputs["candidate_best_lags"].append(best_lags)
                outputs["candidate_lag_stability"].append(lag_stability)
                outputs["topk_indices"].append(gather(indices, selected_positions))
                outputs["topk_lags"].append(gather(best_lags, selected_positions))
                outputs["topk_signs"].append(
                    gather(candidate_signs, selected_positions)
                )
                outputs["topk_lag_stability"].append(
                    gather(lag_stability, selected_positions)
                )
                outputs["topk_redundancy"].append(topk_redundancy)
                outputs["topk_mmr_scores"].append(topk_mmr_scores)
                outputs["threshold_relaxed"].append(threshold_relaxed)

        outputs = {key: torch.cat(values, dim=1) for key, values in outputs.items()}
        recompute_start = self._profile_clock(x.device) if do_profile else None
        topk_scores = self._recompute_selected_relevance(
            x_channel,
            outputs["topk_indices"],
            outputs["topk_lags"],
            outputs["topk_lag_stability"],
            None if dynamic_weights is None else dynamic_weights.float(),
            correlation_weights,
        )
        if do_profile:
            recompute_end = self._profile_clock(x.device)
            profile_values["selected_recompute_ms"] = (
                recompute_end - recompute_start
            ) * 1000.0

        result = {
            "candidate_indices": candidate_indices,
            "candidate_scores": candidate_scores,
            "candidate_best_lags": outputs["candidate_best_lags"],
            "candidate_lag_stability": outputs["candidate_lag_stability"],
            "dynamic_weights": dynamic_weights,
            "correlation_weights": correlation_weights,
            "used_global_rescue": False,
            "sparse_exact_path": True,
            "topk_indices": outputs["topk_indices"],
            "topk_scores": topk_scores,
            "topk_lags": outputs["topk_lags"],
            "topk_signs": outputs["topk_signs"],
            "topk_lag_stability": outputs["topk_lag_stability"],
            "topk_redundancy": outputs["topk_redundancy"],
            "topk_mmr_scores": outputs["topk_mmr_scores"],
            "threshold_relaxed": outputs["threshold_relaxed"],
            "max_candidate_targets": max_candidate_targets,
        }
        result["global_candidate_indices"] = refined["global_indices"]
        if self.return_diagnostics:
            result["similarity"] = similarity
            result["raw_correlation"] = raw_full
            result["diff_correlation"] = diff_full
        auxiliary_start = self._profile_clock(x.device) if do_profile else None
        if self.return_auxiliary:
            auxiliary_x = self._gather_channels(
                x_channel,
                outputs["topk_indices"],
            )
            aligned_auxiliary_x, alignment_mask = self._causal_align(
                auxiliary_x,
                outputs["topk_lags"],
            )
            if self.apply_sign:
                aligned_auxiliary_x = aligned_auxiliary_x * outputs[
                    "topk_signs"
                ].unsqueeze(-1)
            result["auxiliary_x"] = auxiliary_x
            result["aligned_auxiliary_x"] = aligned_auxiliary_x
            result["alignment_mask"] = alignment_mask
        if do_profile:
            auxiliary_end = self._profile_clock(x.device)
            profile_values["auxiliary_gather_ms"] = (
                auxiliary_end - auxiliary_start
            ) * 1000.0
            profile_values["total_ms"] = (
                self._profile_clock(x.device) - total_start
            ) * 1000.0
            self.last_profile = profile_values
        return result


@torch.no_grad()
def build_global_directional_candidate_graph(
    train_history,
    seq_len,
    max_lag=8,
    candidate_budget=64,
    stability_gamma=5.0,
    eps=1e-5,
    sample_stride=24,
    max_windows=1024,
    batch_size=8,
    device=None,
):
    """Build the train-only M-candidate directional recall graph."""
    if seq_len < max_lag + 2:
        raise ValueError("seq_len must leave two samples after max_lag.")
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be positive.")
    if sample_stride < 1 or max_windows < 1 or batch_size < 1:
        raise ValueError("Graph sampling and batch sizes must be positive.")
    history = torch.as_tensor(train_history)
    if history.ndim == 2:
        if history.size(0) < seq_len:
            raise ValueError("Train history is shorter than seq_len.")
        windows = history.unfold(0, seq_len, sample_stride).permute(0, 2, 1)
    elif history.ndim == 3:
        if history.size(1) != seq_len:
            raise ValueError("Pre-windowed train history has the wrong seq_len.")
        windows = history[::sample_stride]
    else:
        raise ValueError("train_history must be [S, N] or [W, L, N].")
    if windows.size(0) > max_windows:
        keep = (
            torch.linspace(
                0,
                windows.size(0) - 1,
                steps=max_windows,
            )
            .round()
            .long()
        )
        windows = windows.index_select(0, keep)
    num_channels = windows.size(-1)
    if num_channels < 2:
        raise ValueError("At least two channels are required.")
    compute_device = torch.device(device) if device is not None else windows.device
    synchronous_sum = torch.zeros(num_channels, num_channels, device=compute_device)
    lag_score_sum = torch.zeros_like(synchronous_sum)
    lag_sum = torch.zeros_like(synchronous_sum)
    lag_square_sum = torch.zeros_like(synchronous_sum)
    window_count = 0

    def normalized(values):
        centered = values - values.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(variance + eps)

    for start in range(0, windows.size(0), batch_size):
        batch = (
            windows[start : start + batch_size]
            .to(
                device=compute_device,
                dtype=torch.float32,
            )
            .permute(0, 2, 1)
        )
        raw_norm = normalized(batch)
        raw_corr = torch.matmul(raw_norm, raw_norm.transpose(-1, -2)) / raw_norm.size(
            -1
        )
        differences = batch[..., 1:] - batch[..., :-1]
        diff_norm = normalized(differences)
        diff_corr = torch.matmul(
            diff_norm, diff_norm.transpose(-1, -2)
        ) / diff_norm.size(-1)
        synchronous = 0.5 * raw_corr.abs() + 0.5 * diff_corr.abs()

        lag_scores = []
        for lag in range(max_lag + 1):
            target_segment = batch[..., lag:]
            candidate_segment = batch[..., : seq_len - lag]
            target_norm = normalized(target_segment)
            candidate_norm = normalized(candidate_segment)
            lag_raw = torch.matmul(
                target_norm,
                candidate_norm.transpose(-1, -2),
            ) / target_norm.size(-1)
            target_diff = target_segment[..., 1:] - target_segment[..., :-1]
            candidate_diff = candidate_segment[..., 1:] - candidate_segment[..., :-1]
            target_diff_norm = normalized(target_diff)
            candidate_diff_norm = normalized(candidate_diff)
            lag_diff = torch.matmul(
                target_diff_norm,
                candidate_diff_norm.transpose(-1, -2),
            ) / target_diff_norm.size(-1)
            lag_scores.append(0.5 * lag_raw.abs() + 0.5 * lag_diff.abs())
        lag_scores = torch.stack(lag_scores, dim=-1)
        best_scores, best_lags = lag_scores.max(dim=-1)
        synchronous_sum.add_(synchronous.sum(dim=0))
        lag_score_sum.add_(best_scores.sum(dim=0))
        lag_sum.add_(best_lags.float().sum(dim=0))
        lag_square_sum.add_(best_lags.float().square().sum(dim=0))
        window_count += batch.size(0)

    synchronous_score = synchronous_sum / float(window_count)
    lag_mean = lag_sum / float(window_count)
    lag_variance = (lag_square_sum / float(window_count) - lag_mean.square()).clamp_min(
        0.0
    )
    lag_variance = lag_variance / max(float(max_lag**2), 1.0)
    stable_lag_score = (lag_score_sum / float(window_count)) * torch.exp(
        -stability_gamma * lag_variance
    )
    global_score = torch.maximum(synchronous_score, stable_lag_score)
    global_score.masked_fill_(
        torch.eye(num_channels, dtype=torch.bool, device=compute_device),
        float("-inf"),
    )
    keep = min(candidate_budget, num_channels - 1)
    scores, indices = torch.topk(
        global_score,
        k=keep,
        dim=-1,
        largest=True,
        sorted=True,
    )
    metadata = {
        "num_channels": num_channels,
        "seq_len": int(seq_len),
        "max_lag": int(max_lag),
        "train_size": int(torch.as_tensor(train_history).shape[0]),
        "sampled_windows": int(window_count),
        "sample_stride": int(sample_stride),
        "max_windows": int(max_windows),
        "stability_gamma": float(stability_gamma),
        "eps": float(eps),
    }
    return indices.cpu(), scores.clamp(0.0, 1.0).cpu(), metadata


class EfficientDirectionalLagMMRChannelSelector(nn.Module):
    """Efficient Directional-Lag-MMR over a fixed global candidate graph.

    A train-only graph limits online Raw/Diff ranking to ``M`` candidates per
    target. Only the local Top-``R`` candidates enter one vectorized lag search,
    and MMR uses six-dimensional descriptors instead of full sequences.
    """

    def __init__(
        self,
        top_k,
        max_lag=8,
        candidate_budget=16,
        lag_candidate_budget=4,
        raw_weight=0.5,
        diff_weight=0.5,
        lag_penalty=0.1,
        mmr_lambda=0.75,
        redundancy_threshold=None,
        target_chunk_size=16,
        eps=1e-5,
        padding_mode="replicate",
        apply_sign=False,
        return_auxiliary=True,
        return_diagnostics=False,
    ):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        if candidate_budget < 1:
            raise ValueError("candidate_budget must be at least 1.")
        if lag_candidate_budget < top_k:
            raise ValueError(
                "lag_candidate_budget must be at least top_k because MMR "
                "selects fixed Top-K from the lag candidate pool."
            )
        if candidate_budget < lag_candidate_budget:
            raise ValueError("candidate_budget must be at least lag_candidate_budget.")
        if target_chunk_size < 1:
            raise ValueError("target_chunk_size must be at least 1.")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1].")
        if redundancy_threshold is not None and not (
            0.0 <= redundancy_threshold <= 1.0
        ):
            raise ValueError("redundancy_threshold must be in [0, 1].")
        if padding_mode not in {"replicate", "zero"}:
            raise ValueError("padding_mode must be either 'replicate' or 'zero'.")
        if lag_penalty < 0:
            raise ValueError("lag_penalty must be non-negative.")
        if raw_weight < 0 or diff_weight < 0:
            raise ValueError("raw_weight and diff_weight must be non-negative.")
        weight_sum = raw_weight + diff_weight
        if weight_sum <= 0:
            raise ValueError("raw_weight and diff_weight cannot both be zero.")

        normalized_weights = torch.tensor(
            [raw_weight / weight_sum, diff_weight / weight_sum],
            dtype=torch.float32,
        )
        self.top_k = int(top_k)
        self.max_lag = int(max_lag)
        self.candidate_budget = int(candidate_budget)
        self.lag_candidate_budget = int(lag_candidate_budget)
        self.correlation_weight_logits = nn.Parameter(
            normalized_weights.clamp_min(eps).log()
        )
        self.lag_penalty = float(lag_penalty)
        self.mmr_lambda = float(mmr_lambda)
        self.redundancy_threshold = redundancy_threshold
        self.target_chunk_size = int(target_chunk_size)
        self.eps = float(eps)
        self.padding_mode = padding_mode
        self.apply_sign = apply_sign
        self.return_auxiliary = return_auxiliary
        self.return_diagnostics = return_diagnostics

        # The graph is deliberately non-persistent. It is rebuilt from the
        # active train split when an experiment starts, so checkpoints neither
        # embed dataset-specific topology nor acquire incompatible state keys.
        self.register_buffer(
            "global_neighbors",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "global_neighbor_scores",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_diagonal_mask_cache",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_time_index_cache",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.last_profile = None

    @property
    def has_global_neighbors(self):
        return self.global_neighbors.numel() > 0

    def set_global_neighbors(self, indices, scores):
        """Install a precomputed ``[N, M]`` train-only neighbor graph."""
        indices = torch.as_tensor(indices, dtype=torch.long)
        scores = torch.as_tensor(scores)
        if indices.ndim != 2 or scores.ndim != 2:
            raise ValueError("Global neighbor indices and scores must be [N, M].")
        if indices.shape != scores.shape:
            raise ValueError("Global neighbor indices and scores must match.")
        if indices.size(1) < self.top_k:
            raise ValueError("The global graph must contain at least top_k neighbors.")
        if indices.size(1) > self.candidate_budget:
            raise ValueError("The global graph exceeds candidate_budget.")
        if torch.any(indices < 0) or torch.any(indices >= indices.size(0)):
            raise ValueError("Global neighbor indices are out of range.")
        targets = torch.arange(indices.size(0)).unsqueeze(-1)
        if torch.any(indices.cpu() == targets):
            raise ValueError("Global neighbor graph must exclude self channels.")
        if not torch.isfinite(scores).all():
            raise ValueError("Global neighbor scores must be finite.")

        self.global_neighbors = (
            indices.detach()
            .to(
                device=self.correlation_weight_logits.device,
            )
            .clone()
        )
        self.global_neighbor_scores = (
            scores.detach()
            .to(
                device=self.correlation_weight_logits.device,
                dtype=self.correlation_weight_logits.dtype,
            )
            .clamp(0.0, 1.0)
            .clone()
        )

    def _correlation_weights(self):
        weights = torch.softmax(self.correlation_weight_logits, dim=0)
        return weights[0], weights[1]

    def _correlation(self, x, y, mask=None):
        if mask is None:
            x_centered = x - x.mean(dim=-1, keepdim=True)
            y_centered = y - y.mean(dim=-1, keepdim=True)
            covariance = (x_centered * y_centered).mean(dim=-1)
            x_var = x_centered.square().mean(dim=-1)
            y_var = y_centered.square().mean(dim=-1)
        else:
            weight = mask.to(dtype=x.dtype)
            count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
            x_mean = (x * weight).sum(dim=-1, keepdim=True) / count
            y_mean = (y * weight).sum(dim=-1, keepdim=True) / count
            x_centered = (x - x_mean) * weight
            y_centered = (y - y_mean) * weight
            covariance = (x_centered * y_centered).sum(dim=-1) / count.squeeze(-1)
            x_var = x_centered.square().sum(dim=-1) / count.squeeze(-1)
            y_var = y_centered.square().sum(dim=-1) / count.squeeze(-1)
        return (covariance / torch.sqrt(x_var * y_var + self.eps)).clamp(-1.0, 1.0)

    def _coarse_similarity(self, x_channel, correlation_weights=None):
        """Fallback-only full online Raw/Diff similarity."""
        if correlation_weights is None:
            correlation_weights = torch.stack(self._correlation_weights())
        raw_weight, diff_weight = correlation_weights.unbind()

        def normalized(values):
            centered = values - values.mean(dim=-1, keepdim=True)
            variance = centered.square().mean(dim=-1, keepdim=True)
            return centered / torch.sqrt(variance + self.eps)

        raw_norm = normalized(x_channel)
        raw_corr = torch.matmul(
            raw_norm,
            raw_norm.transpose(-1, -2),
        ) / raw_norm.size(-1)
        differences = x_channel[..., 1:] - x_channel[..., :-1]
        diff_norm = normalized(differences)
        diff_corr = torch.matmul(
            diff_norm,
            diff_norm.transpose(-1, -2),
        ) / diff_norm.size(-1)
        raw_corr = raw_corr.clamp(-1.0, 1.0)
        diff_corr = diff_corr.clamp(-1.0, 1.0)
        similarity = (
            raw_weight * raw_corr.abs() + diff_weight * diff_corr.abs()
        ).clamp(0.0, 1.0)
        return similarity, raw_corr, diff_corr

    @staticmethod
    def _gather_channels(x_channel, indices):
        return DirectionalLagMMRChannelSelector._gather_channels(
            x_channel,
            indices,
        )

    @staticmethod
    def _gather_candidate_values(values, positions):
        return torch.gather(values, dim=-1, index=positions)

    @staticmethod
    def _gather_candidate_sequences(values, positions):
        gather_index = positions.unsqueeze(-1).expand(
            *positions.shape,
            values.size(-1),
        )
        return torch.gather(values, dim=2, index=gather_index)

    def _diagonal_mask(self, num_channels, device):
        expected_shape = (1, num_channels, num_channels)
        if (
            self._diagonal_mask_cache.device != device
            or tuple(self._diagonal_mask_cache.shape) != expected_shape
        ):
            self._diagonal_mask_cache = torch.eye(
                num_channels,
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
        return self._diagonal_mask_cache

    def _time_index(self, seq_len, device):
        expected_shape = (1, 1, 1, seq_len)
        if (
            self._time_index_cache.device != device
            or tuple(self._time_index_cache.shape) != expected_shape
        ):
            self._time_index_cache = torch.arange(
                seq_len,
                device=device,
            ).view(expected_shape)
        return self._time_index_cache

    def _causal_align(self, candidates, lags):
        seq_len = candidates.size(-1)
        time_index = self._time_index(seq_len, candidates.device)
        source_index = time_index - lags.unsqueeze(-1)
        alignment_mask = source_index >= 0
        source_index = source_index.clamp_min(0).expand_as(candidates)
        aligned = torch.gather(candidates, dim=-1, index=source_index)
        if self.padding_mode == "zero":
            aligned = aligned.masked_fill(~alignment_mask, 0.0)
        return aligned, alignment_mask.expand_as(candidates)

    def _local_similarity(self, target, candidates, correlation_weights):
        raw_weight, diff_weight = correlation_weights.unbind()
        target_values = target.unsqueeze(2)
        raw_corr = self._correlation(target_values, candidates)
        target_diff = target_values[..., 1:] - target_values[..., :-1]
        candidate_diff = candidates[..., 1:] - candidates[..., :-1]
        diff_corr = self._correlation(target_diff, candidate_diff)
        local_score = (
            raw_weight * raw_corr.abs() + diff_weight * diff_corr.abs()
        ).clamp(0.0, 1.0)
        return local_score, raw_corr, diff_corr

    def _vectorized_lag_scores(
        self,
        target,
        candidates,
        correlation_weights=None,
    ):
        """Score all candidate-leading lags in a single tensorized pass."""
        if correlation_weights is None:
            correlation_weights = torch.stack(self._correlation_weights())
        raw_weight, diff_weight = correlation_weights.unbind()
        seq_len = target.size(-1)
        lags = torch.arange(
            self.max_lag + 1,
            device=target.device,
        ).view(1, 1, 1, -1, 1)
        times = torch.arange(seq_len, device=target.device).view(1, 1, 1, 1, -1)
        source_index = (times - lags).clamp_min(0)
        source_index = source_index.expand(
            candidates.size(0),
            candidates.size(1),
            candidates.size(2),
            self.max_lag + 1,
            seq_len,
        )
        lagged_candidates = torch.gather(
            candidates.unsqueeze(3).expand(-1, -1, -1, self.max_lag + 1, -1),
            dim=-1,
            index=source_index,
        )
        target_views = target[:, :, None, None, :].expand_as(lagged_candidates)
        raw_mask = (times >= lags).expand_as(lagged_candidates)
        raw_corr = self._correlation(
            target_views,
            lagged_candidates,
            raw_mask,
        )

        target_diff = target_views[..., 1:] - target_views[..., :-1]
        candidate_diff = lagged_candidates[..., 1:] - lagged_candidates[..., :-1]
        diff_mask = raw_mask[..., 1:] & raw_mask[..., :-1]
        diff_corr = self._correlation(
            target_diff,
            candidate_diff,
            diff_mask,
        )
        score = raw_weight * raw_corr.abs() + diff_weight * diff_corr.abs()
        if self.max_lag > 0:
            penalty = (
                self.lag_penalty
                * torch.arange(
                    self.max_lag + 1,
                    device=target.device,
                    dtype=score.dtype,
                )
                / self.max_lag
            )
            score = score - penalty.view(1, 1, 1, -1)
        return score.clamp_min(0.0), raw_corr, diff_corr

    def _candidate_descriptors(self, aligned_candidates):
        mean = aligned_candidates.mean(dim=-1)
        std = aligned_candidates.std(dim=-1, unbiased=False)
        differences = aligned_candidates[..., 1:] - aligned_candidates[..., :-1]
        diff_mean = differences.mean(dim=-1)
        diff_std = differences.std(dim=-1, unbiased=False)
        time = torch.linspace(
            -1.0,
            1.0,
            aligned_candidates.size(-1),
            device=aligned_candidates.device,
            dtype=aligned_candidates.dtype,
        )
        slope = ((aligned_candidates - mean.unsqueeze(-1)) * time).mean(
            dim=-1
        ) / time.square().mean().clamp_min(self.eps)
        recent_width = max(1, aligned_candidates.size(-1) // 4)
        recent_mean = aligned_candidates[..., -recent_width:].mean(dim=-1)
        return torch.stack(
            (mean, std, diff_mean, diff_std, slope, recent_mean),
            dim=-1,
        )

    def _cheap_mmr_select(self, relevance, descriptors):
        normalized = F.normalize(descriptors, p=2, dim=-1, eps=self.eps)
        redundancy_matrix = (
            torch.matmul(
                normalized,
                normalized.transpose(-1, -2),
            )
            .abs()
            .clamp(0.0, 1.0)
        )
        available = torch.ones_like(relevance, dtype=torch.bool)
        max_redundancy = torch.zeros_like(relevance)
        selected_positions = []
        selected_redundancy = []
        selected_mmr_scores = []
        threshold_relaxed = []

        position = relevance.argmax(dim=-1)
        selected_positions.append(position)
        selected_redundancy.append(torch.zeros_like(relevance[..., 0]))
        selected_mmr_scores.append(
            self.mmr_lambda
            * torch.gather(relevance, -1, position.unsqueeze(-1)).squeeze(-1)
        )
        threshold_relaxed.append(torch.zeros_like(position, dtype=torch.bool))

        for _ in range(1, self.top_k):
            available.scatter_(-1, position.unsqueeze(-1), False)
            selected_row = torch.gather(
                redundancy_matrix,
                dim=2,
                index=position[..., None, None].expand(
                    *position.shape, 1, relevance.size(-1)
                ),
            ).squeeze(2)
            max_redundancy = torch.maximum(max_redundancy, selected_row)
            mmr_score = (
                self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * max_redundancy
            )
            unconstrained = mmr_score.masked_fill(~available, float("-inf"))
            if self.redundancy_threshold is None:
                selection_score = unconstrained
                relaxed = torch.zeros_like(position, dtype=torch.bool)
            else:
                constrained = unconstrained.masked_fill(
                    max_redundancy > self.redundancy_threshold,
                    float("-inf"),
                )
                has_candidate = torch.isfinite(constrained).any(
                    dim=-1,
                    keepdim=True,
                )
                selection_score = torch.where(
                    has_candidate,
                    constrained,
                    unconstrained,
                )
                relaxed = ~has_candidate.squeeze(-1)
            position = selection_score.argmax(dim=-1)
            selected_positions.append(position)
            selected_redundancy.append(
                torch.gather(
                    max_redundancy,
                    -1,
                    position.unsqueeze(-1),
                ).squeeze(-1)
            )
            selected_mmr_scores.append(
                torch.gather(
                    mmr_score,
                    -1,
                    position.unsqueeze(-1),
                ).squeeze(-1)
            )
            threshold_relaxed.append(relaxed)

        return (
            torch.stack(selected_positions, dim=-1),
            torch.stack(selected_redundancy, dim=-1),
            torch.stack(selected_mmr_scores, dim=-1),
            torch.stack(threshold_relaxed, dim=-1),
        )

    @staticmethod
    def _profile_clock(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        import time

        return time.perf_counter()

    def forward(self, x, dynamic_weights=None, profile=False):
        if x.ndim != 3:
            raise ValueError("Expected x with shape [B, L, N].")
        seq_len, num_channels = x.size(1), x.size(2)
        if self.top_k >= num_channels:
            raise ValueError("top_k must be smaller than the number of channels.")
        if self.max_lag > seq_len - 2:
            raise ValueError("max_lag must leave at least two aligned time steps.")
        if dynamic_weights is not None and dynamic_weights.shape != (
            x.size(0),
            num_channels,
            3,
        ):
            raise ValueError("dynamic_weights must have shape [B, N, 3].")

        total_start = self._profile_clock(x.device) if profile else None
        x_channel = x.permute(0, 2, 1)
        correlation_weights = torch.stack(self._correlation_weights())
        used_global_graph = self.has_global_neighbors
        full_diagnostics = None

        if used_global_graph:
            if self.global_neighbors.size(0) != num_channels:
                raise ValueError("Global graph channel count does not match the input.")
            candidate_indices = self.global_neighbors.unsqueeze(0).expand(
                x.size(0), -1, -1
            )
        else:
            similarity, raw_full, diff_full = self._coarse_similarity(
                x_channel,
                correlation_weights,
            )
            coarse_score = similarity.masked_fill(
                self._diagonal_mask(num_channels, x.device),
                float("-inf"),
            )
            num_candidates = min(self.candidate_budget, num_channels - 1)
            _, candidate_indices = torch.topk(
                coarse_score,
                k=num_candidates,
                dim=-1,
                largest=True,
                sorted=True,
            )
            full_diagnostics = (similarity, raw_full, diff_full)

        candidates = self._gather_channels(x_channel, candidate_indices)
        local_scores, local_raw_corr, local_diff_corr = self._local_similarity(
            x_channel,
            candidates,
            correlation_weights,
        )
        num_lag_candidates = min(
            self.lag_candidate_budget,
            candidates.size(2),
        )
        if num_lag_candidates < self.top_k:
            raise ValueError("The available lag candidate pool is smaller than top_k.")
        _, lag_candidate_positions = torch.topk(
            local_scores,
            k=num_lag_candidates,
            dim=-1,
            largest=True,
            sorted=True,
        )
        lag_candidate_indices = self._gather_candidate_values(
            candidate_indices,
            lag_candidate_positions,
        )
        lag_candidates = self._gather_candidate_sequences(
            candidates,
            lag_candidate_positions,
        )
        lag_local_raw = self._gather_candidate_values(
            local_raw_corr,
            lag_candidate_positions,
        )
        lag_local_diff = self._gather_candidate_values(
            local_diff_corr,
            lag_candidate_positions,
        )
        local_end = self._profile_clock(x.device) if profile else None

        outputs = {
            "best_lags": [],
            "lag_confidence": [],
            "topk_indices": [],
            "topk_scores": [],
            "topk_lags": [],
            "topk_signs": [],
            "topk_lag_stability": [],
            "topk_redundancy": [],
            "topk_mmr_scores": [],
            "threshold_relaxed": [],
        }
        lag_seconds = 0.0
        mmr_seconds = 0.0
        for target_start in range(0, num_channels, self.target_chunk_size):
            target_end = min(target_start + self.target_chunk_size, num_channels)
            target_slice = slice(target_start, target_end)
            lag_start = self._profile_clock(x.device) if profile else None
            lag_scores, lag_raw_corr, lag_diff_corr = self._vectorized_lag_scores(
                x_channel[:, target_slice],
                lag_candidates[:, target_slice],
                correlation_weights,
            )
            best_lag_scores, best_lags = lag_scores.max(dim=-1)
            best_raw_corr = torch.gather(
                lag_raw_corr,
                -1,
                best_lags.unsqueeze(-1),
            ).squeeze(-1)
            best_diff_corr = torch.gather(
                lag_diff_corr,
                -1,
                best_lags.unsqueeze(-1),
            ).squeeze(-1)
            peak_count = min(2, self.max_lag + 1)
            best2 = torch.topk(lag_scores, k=peak_count, dim=-1).values
            if peak_count == 1:
                lag_confidence = torch.ones_like(best_lag_scores)
            else:
                lag_confidence = (
                    (best2[..., 0] - best2[..., 1]).clamp_min(0.0)
                    / (best2[..., 0] + self.eps)
                ).clamp(0.0, 1.0)
            if dynamic_weights is None:
                stable_relevance = best_lag_scores * lag_confidence
            else:
                weights = dynamic_weights[:, target_slice].unsqueeze(2)
                stable_relevance = (
                    weights[..., 0] * lag_local_raw[:, target_slice].abs()
                    + weights[..., 1] * lag_local_diff[:, target_slice].abs()
                    + weights[..., 2] * best_lag_scores
                ) * lag_confidence
            stable_relevance = stable_relevance.clamp(0.0, 1.0)
            signed_relation = (
                correlation_weights[0] * best_raw_corr
                + correlation_weights[1] * best_diff_corr
            )
            candidate_signs = torch.where(
                signed_relation < 0,
                -torch.ones_like(signed_relation),
                torch.ones_like(signed_relation),
            )
            aligned_candidates, _ = self._causal_align(
                lag_candidates[:, target_slice],
                best_lags,
            )
            if profile:
                lag_seconds += self._profile_clock(x.device) - lag_start

            mmr_start = self._profile_clock(x.device) if profile else None
            with torch.no_grad():
                descriptors = self._candidate_descriptors(aligned_candidates.detach())
                (
                    selected_positions,
                    topk_redundancy,
                    topk_mmr_scores,
                    threshold_relaxed,
                ) = self._cheap_mmr_select(
                    stable_relevance.detach(),
                    descriptors,
                )
            if profile:
                mmr_seconds += self._profile_clock(x.device) - mmr_start

            outputs["best_lags"].append(best_lags)
            outputs["lag_confidence"].append(lag_confidence)
            outputs["topk_indices"].append(
                self._gather_candidate_values(
                    lag_candidate_indices[:, target_slice], selected_positions
                )
            )
            outputs["topk_scores"].append(
                self._gather_candidate_values(stable_relevance, selected_positions)
            )
            outputs["topk_lags"].append(
                self._gather_candidate_values(best_lags, selected_positions)
            )
            outputs["topk_signs"].append(
                self._gather_candidate_values(candidate_signs, selected_positions)
            )
            outputs["topk_lag_stability"].append(
                self._gather_candidate_values(lag_confidence, selected_positions)
            )
            outputs["topk_redundancy"].append(topk_redundancy)
            outputs["topk_mmr_scores"].append(topk_mmr_scores)
            outputs["threshold_relaxed"].append(threshold_relaxed)

        outputs = {key: torch.cat(values, dim=1) for key, values in outputs.items()}
        result = {
            "candidate_indices": candidate_indices,
            "lag_candidate_indices": lag_candidate_indices,
            "local_candidate_scores": local_scores,
            "candidate_best_lags": outputs["best_lags"],
            "lag_confidence": outputs["lag_confidence"],
            "dynamic_weights": dynamic_weights,
            "correlation_weights": correlation_weights,
            "used_global_graph": used_global_graph,
            "topk_indices": outputs["topk_indices"],
            "topk_scores": outputs["topk_scores"],
            "topk_lags": outputs["topk_lags"],
            "topk_signs": outputs["topk_signs"],
            "topk_lag_stability": outputs["topk_lag_stability"],
            "topk_redundancy": outputs["topk_redundancy"],
            "topk_mmr_scores": outputs["topk_mmr_scores"],
            "threshold_relaxed": outputs["threshold_relaxed"],
        }
        if self.return_diagnostics:
            if full_diagnostics is None:
                result["similarity"] = local_scores
                result["raw_correlation"] = local_raw_corr
                result["diff_correlation"] = local_diff_corr
            else:
                (
                    result["similarity"],
                    result["raw_correlation"],
                    result["diff_correlation"],
                ) = full_diagnostics
        if self.return_auxiliary:
            auxiliary_x = self._gather_channels(
                x_channel,
                result["topk_indices"],
            )
            aligned_auxiliary_x, alignment_mask = self._causal_align(
                auxiliary_x,
                result["topk_lags"],
            )
            if self.apply_sign:
                aligned_auxiliary_x = aligned_auxiliary_x * result[
                    "topk_signs"
                ].unsqueeze(-1)
            result["auxiliary_x"] = auxiliary_x
            result["aligned_auxiliary_x"] = aligned_auxiliary_x
            result["alignment_mask"] = alignment_mask

        if profile:
            total_seconds = self._profile_clock(x.device) - total_start
            self.last_profile = {
                "coarse_local_ms": (local_end - total_start) * 1000.0,
                "lag_ms": lag_seconds * 1000.0,
                "mmr_ms": mmr_seconds * 1000.0,
                "total_ms": total_seconds * 1000.0,
            }
        return result
