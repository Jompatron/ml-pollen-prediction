"""TFT model definition and configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

try:
    # pytorch-forecasting 1.x ships with standalone `lightning`, not pytorch_lightning
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError:
    PYTORCH_FORECASTING_AVAILABLE = False


def build_model(dataset: "TimeSeriesDataSet") -> "TemporalFusionTransformer":
    """
    Build TFT model from dataset metadata.
    The dataset defines which features exist; the model is built to match.
    """
    if not PYTORCH_FORECASTING_AVAILABLE:
        raise ImportError("pytorch-forecasting not installed")

    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=settings.learning_rate,
        hidden_size=settings.hidden_size,
        lstm_layers=settings.lstm_layers,
        attention_head_size=settings.attention_head_size,
        dropout=settings.dropout,
        hidden_continuous_size=settings.hidden_size // 2,
        output_size=len(settings.quantiles),  # P10, P25, P50, P75, P90
        loss=QuantileLoss(quantiles=settings.quantiles),
        reduce_on_plateau_patience=4,
        log_interval=10,
        log_val_interval=1,
    )
