from module.predict_head.cycle_aligned import CycleAlignedPersistentTransientForecaster


def build_local_predictor(predict_type, configs, num_local_channels):
    if predict_type.lower() != "cyclealigned":
        raise ValueError("Only CycleAligned is supported.")
    local_period = int(
        getattr(configs, "local_cycle", None) or getattr(configs, "cycle", 24)
    )
    return CycleAlignedPersistentTransientForecaster(
        seq_len=configs.seq_len,
        pred_len=configs.pred_len,
        period=local_period,
        hidden_dim=getattr(configs, "cycle_forecaster_hidden", configs.d_model),
        dropout=configs.dropout,
        recent_cycles=getattr(configs, "transient_recent_cycles", 2),
        transient_gate_bias=getattr(configs, "transient_gate_bias", -2.0),
        use_transient_branch=bool(getattr(configs, "use_transient_branch", 1)),
        eps=float(getattr(configs, "similarity_eps", 1e-05)),
    )
