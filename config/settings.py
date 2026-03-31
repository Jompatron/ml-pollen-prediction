from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Settings:
    # === Location ===
    primary_location: str = "Stockholm"
    primary_lat: float = 59.3293
    primary_lng: float = 18.0686
    timezone: str = "Europe/Stockholm"

    reference_locations: dict = field(default_factory=lambda: {
        "Uppsala": (59.8586, 17.6389),
        "Norrköping": (58.5942, 16.1826),
        "Västerås": (59.6099, 16.5448),
        "Linköping": (58.4108, 15.6214),
    })

    # === Species ===
    pollen_species: list = field(default_factory=lambda: [
        "birch_pollen", "grass_pollen", "alder_pollen", "mugwort_pollen"
    ])
    primary_target: str = "birch_pollen"

    # === Data Collection ===
    open_meteo_air_quality_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    open_meteo_weather_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_historical_url: str = "https://archive-api.open-meteo.com/v1/archive"
    max_past_days: int = 92
    forecast_days: int = 5

    weather_features: list = field(default_factory=lambda: [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "precipitation", "rain", "surface_pressure",
        "shortwave_radiation", "direct_radiation",
        "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
    ])

    air_quality_features: list = field(default_factory=lambda: [
        "pm10", "pm2_5", "european_aqi",
    ])

    # === Feature Engineering ===
    encoder_length: int = 168    # 7 days * 24 hours
    prediction_length: int = 120  # 5 days * 24 hours

    temporal_features: list = field(default_factory=lambda: [
        "hour_of_day", "day_of_week", "day_of_year", "month",
        "is_weekend", "days_since_season_start",
    ])

    # === Model ===
    hidden_size: int = 64
    lstm_layers: int = 2
    attention_head_size: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    quantiles: list = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    n_ensemble_members: int = 5

    # === Drift Detection ===
    psi_threshold: float = 0.2
    performance_decay_threshold: float = 0.15
    min_samples_for_drift: int = 168

    # === Paths ===
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    artifacts_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)

    def __post_init__(self):
        self.artifacts_dir = self.project_root / "artifacts"
        self.data_dir = self.artifacts_dir / "data"
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"
        for d in [self.data_dir, self.models_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
