import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterCyclePersistentBlock(nn.Module):
    """Model shared same-phase trajectories across complete cycles."""

    def __init__(self, seq_len, pred_len, period, hidden_dim, dropout):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period = int(period)
        if self.period < 1:
            raise ValueError("period must be positive.")
        self.num_cycles = self.seq_len // self.period
        if self.num_cycles < 2:
            raise ValueError("seq_len // period must be at least 2.")
        self.complete_len = self.num_cycles * self.period
        self.num_future_cycles = math.ceil(self.pred_len / self.period)

        self.phase_encoder = nn.Sequential(
            nn.Linear(self.num_cycles, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.backcast_head = nn.Linear(hidden_dim, self.num_cycles)
        self.phase_mixer = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            groups=hidden_dim,
        )
        self.future_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_future_cycles),
        )

    def forward(self, residual):
        if residual.ndim != 3 or residual.size(-1) != self.seq_len:
            raise ValueError(f"Expected residual [B, N, {self.seq_len}].")
        batch_size, num_channels, _ = residual.shape
        cycles = residual[..., -self.complete_len :].reshape(
            batch_size,
            num_channels,
            self.num_cycles,
            self.period,
        )
        phase_trajectories = cycles.permute(0, 1, 3, 2)
        hidden = self.phase_encoder(phase_trajectories)

        backcast_cycles = self.backcast_head(hidden).permute(0, 1, 3, 2)
        complete_backcast = backcast_cycles.reshape(
            batch_size, num_channels, self.complete_len
        )
        backcast = F.pad(
            complete_backcast,
            (self.seq_len - self.complete_len, 0),
        )

        phase_features = hidden.permute(0, 1, 3, 2).reshape(
            batch_size * num_channels, hidden.size(-1), self.period
        )
        mixed_features = phase_features + self.phase_mixer(
            F.pad(phase_features, (2, 2), mode="circular")
        )
        mixed_hidden = mixed_features.reshape(
            batch_size, num_channels, hidden.size(-1), self.period
        ).permute(0, 1, 3, 2)

        latest = cycles[:, :, -1, :]
        trend = latest - cycles[:, :, -2, :]
        decoder_input = torch.cat(
            (
                mixed_hidden,
                latest.unsqueeze(-1),
                trend.unsqueeze(-1),
            ),
            dim=-1,
        )
        future_by_phase = self.future_decoder(decoder_input)
        future_cycles = future_by_phase.permute(0, 1, 3, 2)
        forecast = future_cycles.reshape(
            batch_size,
            num_channels,
            self.num_future_cycles * self.period,
        )[..., : self.pred_len]
        details = {
            "cycles": cycles,
            "phase_trajectories": phase_trajectories,
            "mixed_phase_hidden": mixed_hidden,
            "latest_cycle": latest,
            "cycle_trend": trend,
            "future_cycles": future_cycles,
        }
        return backcast, forecast, details


class IntraCycleTransientBlock(nn.Module):
    """Forecast recent high-frequency residual unexplained by persistence."""

    def __init__(
        self,
        seq_len,
        pred_len,
        period,
        hidden_dim,
        dropout,
        recent_cycles=2,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period = int(period)
        self.recent_cycles = int(recent_cycles)
        if self.recent_cycles < 1:
            raise ValueError("recent_cycles must be positive.")
        self.recent_len = min(self.seq_len, self.recent_cycles * self.period)
        self.smoothing_kernel = max(3, self.period // 8)
        if self.smoothing_kernel % 2 == 0:
            self.smoothing_kernel += 1

        self.feature_encoder = nn.Sequential(
            nn.Linear(2 * self.recent_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.backcast_head = nn.Linear(hidden_dim, self.recent_len)
        self.forecast_head = nn.Linear(hidden_dim, self.pred_len)
        nn.init.zeros_(self.forecast_head.weight)
        nn.init.zeros_(self.forecast_head.bias)

    def forward(self, residual):
        if residual.ndim != 3 or residual.size(-1) != self.seq_len:
            raise ValueError(f"Expected residual [B, N, {self.seq_len}].")
        recent = residual[..., -self.recent_len :]
        smoothed = F.avg_pool1d(
            recent.reshape(-1, 1, self.recent_len),
            kernel_size=self.smoothing_kernel,
            stride=1,
            padding=self.smoothing_kernel // 2,
        ).reshape_as(recent)
        high_frequency = recent - smoothed
        hidden = self.feature_encoder(torch.cat((recent, high_frequency), dim=-1))
        recent_backcast = self.backcast_head(hidden)
        backcast = F.pad(
            recent_backcast,
            (self.seq_len - self.recent_len, 0),
        )
        forecast = self.forecast_head(hidden)
        details = {
            "recent": recent,
            "smoothed": smoothed,
            "high_frequency": high_frequency,
        }
        return backcast, forecast, details


class CycleAlignedPersistentTransientForecaster(nn.Module):
    """Cycle-aligned persistent baseline with gated transient revision."""

    def __init__(
        self,
        seq_len,
        pred_len,
        period,
        hidden_dim,
        dropout,
        recent_cycles=2,
        transient_gate_bias=-2.0,
        use_transient_branch=True,
        eps=1e-5,
    ):
        super().__init__()
        if seq_len // period < 2:
            raise ValueError("seq_len // period must be at least 2.")
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period = int(period)
        self.eps = float(eps)
        self.use_transient_branch = bool(use_transient_branch)
        self.persistent_block = InterCyclePersistentBlock(
            seq_len, pred_len, period, hidden_dim, dropout
        )
        self.transient_block = IntraCycleTransientBlock(
            seq_len,
            pred_len,
            period,
            hidden_dim,
            dropout,
            recent_cycles,
        )
        gate_hidden = max(8, hidden_dim // 4)
        self.transient_gate = nn.Sequential(
            nn.Linear(5, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.normal_(self.transient_gate[-1].weight, std=1e-3)
        nn.init.constant_(self.transient_gate[-1].bias, transient_gate_bias)

    def _persistence_features(self, cycles, drift_features):
        latest = cycles[:, :, -1]
        historical_mean = cycles[:, :, :-1].mean(dim=2)
        transient_distance = (latest - historical_mean).abs().mean(dim=-1)

        latest_centered = latest - latest.mean(dim=-1, keepdim=True)
        history_centered = historical_mean - historical_mean.mean(dim=-1, keepdim=True)
        persistence = (
            (latest_centered * history_centered).mean(dim=-1)
            / torch.sqrt(
                latest_centered.square().mean(dim=-1)
                * history_centered.square().mean(dim=-1)
                + self.eps
            )
        ).clamp(-1.0, 1.0)
        cycle_variation = cycles.var(dim=2, unbiased=False).mean(dim=-1)

        if drift_features is None:
            trend_shift = torch.zeros_like(transient_distance)
            instability = torch.zeros_like(transient_distance)
        else:
            if drift_features.shape != (*cycles.shape[:2], 5):
                raise ValueError("drift_features must have shape [B, N, 5].")
            trend_shift = drift_features[..., 3]
            instability = drift_features[..., 4]
        return torch.stack(
            (
                transient_distance,
                persistence,
                cycle_variation,
                trend_shift,
                instability,
            ),
            dim=-1,
        )

    def forward(
        self,
        residual,
        drift_features=None,
        return_components=False,
    ):
        persistent_backcast, persistent_forecast, persistent_details = (
            self.persistent_block(residual)
        )
        remaining = residual - persistent_backcast
        gate_features = self._persistence_features(
            persistent_details["cycles"], drift_features
        )
        if self.use_transient_branch:
            transient_backcast, transient_forecast, transient_details = (
                self.transient_block(remaining)
            )
            transient_gate = torch.sigmoid(self.transient_gate(gate_features))
        else:
            transient_backcast = torch.zeros_like(persistent_backcast)
            transient_forecast = torch.zeros_like(persistent_forecast)
            transient_gate = gate_features.new_zeros(*gate_features.shape[:-1], 1)
            recent = remaining[..., -self.transient_block.recent_len :]
            transient_details = {
                "recent": recent,
                "high_frequency": torch.zeros_like(recent),
            }
        forecast = persistent_forecast + transient_gate * transient_forecast
        if not return_components:
            return forecast
        return {
            "forecast": forecast,
            "persistent_forecast": persistent_forecast,
            "transient_forecast": transient_forecast,
            "persistent_backcast": persistent_backcast,
            "transient_backcast": transient_backcast,
            "remaining_residual": remaining - transient_backcast,
            "transient_gate": transient_gate,
            "persistence_features": gate_features,
            "cycles": persistent_details["cycles"],
            "phase_trajectories": persistent_details["phase_trajectories"],
            "transient_recent": transient_details["recent"],
            "transient_high_frequency": transient_details["high_frequency"],
        }
