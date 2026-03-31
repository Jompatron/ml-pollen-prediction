# Pollen Forecast Stockholm — ML Backend

Temporal Fusion Transformer ensemble that produces probabilistic 5-day pollen forecasts for Stockholm, served via FastAPI.

## Architecture

```
Open-Meteo CAMS (pollen)  ─┐
Open-Meteo ERA5 (weather)  ─┤─► Feature Engineering ─► TFT Ensemble ─► FastAPI ─► Next.js
                            │                            (5 models ×              frontend
                            └── Drift Detector ──────► 4 species)
```

## Quick Start

```bash
cd ml

# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Backfill historical data (run once, takes ~5 min)
python -m data.historical_backfill

# 4. Run baseline evaluation
python -c "
from data.storage import ParquetStorage
from data.feature_engineering import PollenFeatureEngineer
from models.baseline import evaluate_baselines
df = PollenFeatureEngineer().transform(ParquetStorage().load_all('merged'))
print(evaluate_baselines(df))
"

# 5. Train models (takes 30-120 min on CPU)
python -m models.train

# 6. Start API server
uvicorn api.main:app --reload --port 8000

# API docs: http://localhost:8000/docs
```

## Execution Order (first-time setup)

1. `python -m data.historical_backfill` — download ~2 years of pollen + weather data
2. `python -m models.train` — train TFT ensemble (5 models × 4 species)
3. `uvicorn api.main:app --port 8000` — serve predictions
4. Set `NEXT_PUBLIC_ML_API_URL=http://localhost:8000/api/v1` in the Next.js `.env.local`

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/v1/forecast` | Probabilistic 5-day forecast (P10/P50/P90 per hour per species) |
| `GET /api/v1/current` | Current pollen levels from latest observation |
| `GET /api/v1/health` | Model status, version, drift status |
| `GET /api/v1/metrics` | MAE, RMSE, calibration from training + live performance |
| `GET /api/v1/history` | Historical observations (last N days) |

## Scheduled Jobs

**GitHub Actions (recommended — free)**
- `.github/workflows/daily-collect.yml` — runs at 06:00 UTC daily
- `.github/workflows/weekly-retrain.yml` — runs at 02:00 UTC every Sunday

**Local runner (alternative)**
```bash
python -m jobs.scheduler
```

## Running Tests

```bash
cd ml
pytest tests/ -v
```

## Key Design Decisions

| Decision | Why |
|---|---|
| Log1p target transform | Pollen is heavily right-skewed; log1p compresses scale while preserving zero |
| Cyclical encoding | Hour 23 and hour 0 are adjacent — sine/cosine keeps them adjacent |
| Growing Degree Days | Birch tracks cumulative warmth, not calendar dates |
| Walk-forward validation | Prevents future information leaking into training |
| 5-model ensemble | Reduces variance; median of 5 is more stable than any single model |
| Parquet storage | No database needed; monthly partitions make incremental updates cheap |
| PSI drift detection | Quantifies distributional shift without labels |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_ML_API_URL` | `http://localhost:8000/api/v1` | ML API base URL for Next.js frontend |
