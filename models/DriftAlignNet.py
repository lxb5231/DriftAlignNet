import torch
import torch.nn as nn
from module.channel_selectors import (
    DirectionalLagMMRChannelSelector,
    EfficientDirectionalLagMMRChannelSelector,
    FastExactDirectionalLagMMRChannelSelector,
    StatisticalTopKChannelSelector,
)
from module.predict_head import build_local_predictor


class AnchorGuidanceAdapter(nn.Module):
    """Apply conservative horizon-aware anchor guidance to a direct forecast."""

    def __init__(self, drift_dim, hidden_dim, pred_len):
        super().__init__()
        position_dim = max(4, min(8, hidden_dim // 4))
        self.anchor_delta_net = nn.Sequential(
            nn.Conv1d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        self.horizon_embedding = nn.Embedding(pred_len, position_dim)
        self.anchor_gate = nn.Sequential(
            nn.Linear(drift_dim + 1 + position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.anchor_delta_net[-1].weight)
        nn.init.zeros_(self.anchor_delta_net[-1].bias)
        nn.init.normal_(self.horizon_embedding.weight, std=0.02)
        nn.init.constant_(self.anchor_gate[-1].bias, -3.0)

    def forward(self, direct_forecast, anchor_future, drift_state):
        batch_size, pred_len, num_channels = direct_forecast.shape
        features = torch.stack(
            (direct_forecast, anchor_future, direct_forecast - anchor_future), dim=-1
        )
        temporal_features = features.permute(0, 2, 3, 1).reshape(
            batch_size * num_channels, 3, pred_len
        )
        anchor_delta = (
            self.anchor_delta_net(temporal_features)
            .reshape(batch_size, num_channels, pred_len)
            .permute(0, 2, 1)
        )
        disagreement = (direct_forecast - anchor_future).abs().permute(0, 2, 1)
        drift_context = drift_state.unsqueeze(2).expand(-1, -1, pred_len, -1)
        horizon_positions = torch.arange(pred_len, device=direct_forecast.device)
        position_context = (
            self.horizon_embedding(horizon_positions)
            .view(1, 1, pred_len, -1)
            .expand(batch_size, num_channels, -1, -1)
        )
        gate_features = torch.cat(
            (drift_context, disagreement.unsqueeze(-1), position_context), dim=-1
        )
        gate_logits = self.anchor_gate(gate_features).squeeze(-1)
        need_mask = torch.tanh(disagreement)
        anchor_gate = torch.sigmoid(gate_logits) * need_mask
        forecast = direct_forecast + anchor_gate.permute(0, 2, 1) * anchor_delta
        return (forecast, anchor_delta, anchor_gate)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.use_revin = configs.use_revin
        self.top_k = int(configs.top_k)
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative.")
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_anchor_guidance = bool(getattr(configs, "use_anchor_guidance", 1))
        configured_predict_type = getattr(configs, "predict_type", "CycleAligned")
        supported_predict_types = {"cyclealigned"}
        if configured_predict_type.lower() not in supported_predict_types:
            raise ValueError(
                f"Unsupported predict_type: {configured_predict_type!r}. Available predictor: 'CycleAligned'."
            )
        self.predict_type = configured_predict_type
        self.global_period = int(
            getattr(configs, "anchor_cycle", None) or getattr(configs, "cycle", 24)
        )
        self.local_period = int(
            getattr(configs, "local_cycle", None) or getattr(configs, "cycle", 24)
        )
        if self.global_period < 1:
            raise ValueError("anchor_cycle must be positive.")
        if self.local_period < 1:
            raise ValueError("local_cycle must be positive.")
        self.period = self.global_period
        self.eps = float(getattr(configs, "similarity_eps", 1e-05))
        self.selector_type = getattr(
            configs, "selector_type", "directional_lag_mmr"
        ).lower()
        if self.top_k == 0:
            self.channel_selector = None
        elif self.selector_type == "statistical":
            self.channel_selector = StatisticalTopKChannelSelector(
                top_k=self.top_k,
                raw_weight=getattr(configs, "raw_weight", 0.5),
                diff_weight=getattr(configs, "diff_weight", 0.5),
                eps=getattr(configs, "similarity_eps", 1e-05),
                return_auxiliary=False,
            )
        elif self.selector_type == "directional_lag_mmr":
            self.channel_selector = DirectionalLagMMRChannelSelector(
                top_k=self.top_k,
                max_lag=getattr(configs, "max_lag", 8),
                candidate_multiplier=getattr(configs, "candidate_multiplier", 8),
                target_chunk_size=getattr(configs, "selector_target_chunk_size", 16),
                raw_weight=getattr(configs, "raw_weight", 0.5),
                diff_weight=getattr(configs, "diff_weight", 0.5),
                lag_penalty=getattr(configs, "lag_penalty", 0.1),
                stability_windows=getattr(configs, "stability_windows", 4),
                stability_gamma=getattr(configs, "stability_gamma", 5.0),
                mmr_lambda=getattr(configs, "mmr_lambda", 0.75),
                redundancy_threshold=getattr(configs, "redundancy_threshold", None),
                eps=getattr(configs, "similarity_eps", 1e-05),
                padding_mode=getattr(configs, "alignment_padding", "replicate"),
                apply_sign=True,
                return_auxiliary=False,
            )
        elif self.selector_type == "efficient_directional_lag_mmr":
            self.channel_selector = EfficientDirectionalLagMMRChannelSelector(
                top_k=self.top_k,
                max_lag=getattr(configs, "max_lag", 8),
                candidate_budget=getattr(configs, "candidate_budget", 16),
                lag_candidate_budget=getattr(configs, "lag_candidate_budget", 4),
                target_chunk_size=getattr(configs, "selector_target_chunk_size", 16),
                raw_weight=getattr(configs, "raw_weight", 0.5),
                diff_weight=getattr(configs, "diff_weight", 0.5),
                lag_penalty=getattr(configs, "lag_penalty", 0.1),
                mmr_lambda=getattr(configs, "mmr_lambda", 0.75),
                redundancy_threshold=getattr(configs, "redundancy_threshold", None),
                eps=getattr(configs, "similarity_eps", 1e-05),
                padding_mode=getattr(configs, "alignment_padding", "replicate"),
                apply_sign=True,
                return_auxiliary=False,
                return_diagnostics=False,
            )
        elif self.selector_type == "fast_exact_directional_lag_mmr":
            self.channel_selector = FastExactDirectionalLagMMRChannelSelector(
                top_k=self.top_k,
                max_lag=getattr(configs, "max_lag", 8),
                global_candidate_budget=getattr(
                    configs, "fast_global_candidate_budget", 64
                ),
                refine_budget=getattr(configs, "fast_refine_budget", 32),
                candidate_chunk_size=getattr(
                    configs, "selector_candidate_chunk_size", 32
                ),
                target_chunk_size=getattr(configs, "selector_target_chunk_size", 16),
                raw_weight=getattr(configs, "raw_weight", 0.5),
                diff_weight=getattr(configs, "diff_weight", 0.5),
                lag_penalty=getattr(configs, "lag_penalty", 0.1),
                stability_windows=getattr(configs, "stability_windows", 4),
                stability_gamma=getattr(configs, "stability_gamma", 5.0),
                mmr_lambda=getattr(configs, "mmr_lambda", 0.75),
                redundancy_threshold=getattr(configs, "redundancy_threshold", None),
                eps=getattr(configs, "similarity_eps", 1e-05),
                padding_mode=getattr(configs, "alignment_padding", "replicate"),
                apply_sign=True,
                return_auxiliary=False,
                return_diagnostics=getattr(
                    configs, "selector_return_diagnostics", False
                ),
                profile=getattr(configs, "selector_profile", False),
                strict_legacy_threshold=getattr(
                    configs, "fast_strict_legacy_threshold", 16
                ),
            )
        else:
            raise ValueError(
                f"Unsupported selector_type: {self.selector_type!r}. Available selectors: 'statistical', 'directional_lag_mmr', 'efficient_directional_lag_mmr', 'fast_exact_directional_lag_mmr'."
            )
        self.phase_anchor = nn.Parameter(torch.zeros(self.global_period, self.enc_in))
        drift_hidden = max(16, min(self.d_model, 128))
        self.drift_encoder = nn.Sequential(
            nn.Linear(5, drift_hidden), nn.GELU(), nn.LayerNorm(drift_hidden)
        )
        self.dynamic_score_head = nn.Linear(drift_hidden, 3)
        anchor_hidden = max(8, min(self.d_model, 32))
        self.anchor_guidance_adapter = AnchorGuidanceAdapter(
            drift_dim=drift_hidden, hidden_dim=anchor_hidden, pred_len=self.pred_len
        )
        self.self_predictor = build_local_predictor(
            self.predict_type, configs, self.top_k + 1
        )
        if self.top_k == 0:
            self.channel_correction = None
            self.channel_gate = None
        else:
            correction_in = (self.top_k + 1) * self.seq_len
            self.channel_correction = nn.Sequential(
                nn.Linear(correction_in, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.pred_len),
            )
            self.channel_gate = nn.Linear(drift_hidden + 3, 1)
            nn.init.zeros_(self.channel_correction[-1].weight)
            nn.init.zeros_(self.channel_correction[-1].bias)
            nn.init.constant_(self.channel_gate.bias, -3.0)

    def _phase_values(self, phase_index, length):
        offsets = torch.arange(length, device=phase_index.device)
        indices = (phase_index[:, None] + offsets[None, :]) % self.global_period
        return self.phase_anchor[indices]

    def _apply_anchor_guidance(self, direct_forecast, anchor_future, drift_state):
        if self.use_anchor_guidance:
            return self.anchor_guidance_adapter(
                direct_forecast, anchor_future, drift_state
            )
        anchor_delta = torch.zeros_like(direct_forecast)
        anchor_gate = direct_forecast.new_zeros(
            direct_forecast.size(0), direct_forecast.size(2), direct_forecast.size(1)
        )
        return (direct_forecast, anchor_delta, anchor_gate)

    def _drift_features(self, residual):
        """Five scale-robust drift statistics, returned as [B, N, 5]."""
        r = residual.permute(0, 2, 1)
        window = min(self.local_period, self.seq_len)
        recent = r[..., -window:]
        previous = r[..., -2 * window : -window]
        if previous.size(-1) == 0:
            previous = r[..., :window]
        width = min(recent.size(-1), previous.size(-1))
        recent, previous = (recent[..., -width:], previous[..., -width:])
        cycle_distance = recent.abs().mean(-1)
        mean_shift = (recent.mean(-1) - previous.mean(-1)).abs()
        recent_std = recent.std(-1, unbiased=False)
        previous_std = previous.std(-1, unbiased=False)
        scale_shift = torch.log(
            (recent_std + self.eps) / (previous_std + self.eps)
        ).abs()
        time = torch.linspace(-1, 1, width, device=r.device, dtype=r.dtype)
        denominator = time.square().mean().clamp_min(self.eps)
        recent_slope = (recent * time).mean(-1) / denominator
        previous_slope = (previous * time).mean(-1) / denominator
        trend_shift = (recent_slope - previous_slope).abs()
        if self.seq_len > self.local_period:
            left = r[..., self.local_period :]
            right = r[..., : -self.local_period]
            left = left - left.mean(-1, keepdim=True)
            right = right - right.mean(-1, keepdim=True)
            acf = (left * right).mean(-1) / torch.sqrt(
                left.square().mean(-1) * right.square().mean(-1) + self.eps
            )
            instability = 1.0 - acf.clamp(-1, 1)
        else:
            instability = torch.ones_like(cycle_distance)
        return torch.stack(
            (cycle_distance, mean_shift, scale_shift, trend_shift, instability), dim=-1
        )

    def _indexed_channel_correction(self, target, selection):
        """Apply the shared correction MLP to every target in one tensor."""
        auxiliary = DirectionalLagMMRChannelSelector._gather_channels(
            target, selection["topk_indices"]
        )
        lags = selection.get("topk_lags")
        if lags is not None:
            auxiliary, _ = self.channel_selector._causal_align(auxiliary, lags)
            auxiliary = auxiliary * selection["topk_signs"].unsqueeze(-1)
        local_group = torch.cat((target.unsqueeze(2), auxiliary), dim=2)
        return self.channel_correction(local_group.flatten(start_dim=2)).permute(
            0, 2, 1
        )

    def forward(self, x, cycle_index=None, return_auxiliary=False):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
            seq_std = torch.sqrt(seq_var)
            z = (x - seq_mean) / seq_std
        else:
            z, seq_mean, seq_std = (x, None, None)
        if cycle_index is None:
            cycle_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        cycle_index = cycle_index.to(device=x.device, dtype=torch.long).view(-1)
        anchor_history = self._phase_values(cycle_index, self.seq_len)
        future_phase = (cycle_index + self.seq_len) % self.global_period
        anchor_future = self._phase_values(future_phase, self.pred_len)
        residual = z - anchor_history
        drift = self._drift_features(residual)
        drift_state = self.drift_encoder(drift)
        predictor_target = z.permute(0, 2, 1)
        self_details = self.self_predictor(
            predictor_target, drift_features=drift, return_components=return_auxiliary
        )
        if return_auxiliary:
            self_forecast = self_details["forecast"].permute(0, 2, 1)
        else:
            self_forecast = self_details.permute(0, 2, 1)
        final_self_forecast, anchor_delta, anchor_gate = self._apply_anchor_guidance(
            self_forecast, anchor_future, drift_state
        )
        if self.top_k == 0:
            selection = None
            dynamic_weights = None
            channel_gate = residual.new_zeros(residual.size(0), residual.size(2), 1)
            output = final_self_forecast
        else:
            dynamic_weights = torch.softmax(self.dynamic_score_head(drift_state), -1)
            if self.selector_type in {
                "directional_lag_mmr",
                "efficient_directional_lag_mmr",
                "fast_exact_directional_lag_mmr",
            }:
                selection = self.channel_selector(residual, dynamic_weights)
            else:
                selection = self.channel_selector(residual)
            channel_delta = self._indexed_channel_correction(
                residual.permute(0, 2, 1), selection
            )
            scores = selection["topk_scores"]
            margin = (
                scores[..., :1] - scores[..., 1:2]
                if self.top_k > 1
                else scores[..., :1]
            )
            mean_score = scores.mean(-1, keepdim=True)
            stability = selection.get(
                "topk_lag_stability", torch.ones_like(scores)
            ).mean(-1, keepdim=True)
            confidence = torch.cat((margin, mean_score, stability), -1)
            channel_gate = torch.sigmoid(
                self.channel_gate(torch.cat((drift_state, confidence), -1))
            )
            output = final_self_forecast + channel_gate.permute(0, 2, 1) * channel_delta
        if self.use_revin:
            output = output * seq_std + seq_mean
        if return_auxiliary:
            return {
                "forecast": output,
                "drift_features": drift,
                "dynamic_weights": dynamic_weights,
                "channel_gate": channel_gate,
                "anchor_gate": anchor_gate,
                "anchor_delta": anchor_delta,
                "direct_forecast": self_forecast,
                "final_self_forecast": final_self_forecast,
                "selection": selection,
                "self_forecaster": self_details,
            }
        return output
