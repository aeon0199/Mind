"""Frozen M3R-2 replication and predictor preregistration."""

M3R2_CALIBRATION_PROTOCOL = "observer-m3r2-basin-replication-v1"
M3R2_ANALYSIS_PROTOCOL = "observer-m3r2-prompt-held-out-analysis-v1"
M3R2_PREDICTOR_L2_REGULARIZATION = 0.05
M3R2_AUROC_THRESHOLD = 0.70
M3R2_MINIMUM_VALID_PROMPT_SPLITS = 5

M3R2_PREDICTOR_FEATURES = (
    "clean_top1_margin",
    "clean_hidden_norm",
    "normalized_position",
    "context_sequence_length",
    "prefix_repetition_score",
    "prefix_unique_token_ratio",
    "consumed_token.length",
    "consumed_token.leading_space",
    "clean_diagnostics.local_prediction_error",
    "clean_diagnostics.logit_entropy",
    "clean_diagnostics.hidden_velocity",
    "clean_diagnostics.hidden_acceleration",
    "clean_diagnostics.spectral.spectral_entropy",
    "clean_diagnostics.spectral.permutation_change",
    "clean_diagnostics.svd.effective_rank",
    "clean_diagnostics.svd.top1_energy_frac",
    "clean_diagnostics.layer_stiffness.27.stiffness",
    "clean_diagnostics.layer_stiffness.27.stiffness_trend",
    "prompt_class.code",
    "prompt_class.creative",
    "prompt_class.factual",
    "prompt_class.procedural",
    "prompt_class.reasoning",
)

