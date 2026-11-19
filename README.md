# Delhi AQI 48-hour forecaster

Forecasts hourly PM2.5 and the Indian AQI category for five grid points in Delhi, up to 48 hours ahead, and checks honestly when the model beats "same as now" (persistence) and when it does not.

> **Data honesty.** Air quality here is *modelled* data from Open-Meteo (Copernicus CAMS), on a coarse grid. It is not CPCB station data, it is smoother than real sensors and can miss the worst spikes. This is an educational project, not an official forecast, and must not be used for health decisions.

## Results

Data: Open-Meteo modelled air quality (CAMS), five Delhi grid points, 2024-01 to 2026-11. Errors in ug/m3.

**Hourly PM2.5 error, 12-month walk-forward (Sep 2025 to Aug 2026, retrained monthly)**

| Horizon (h) | Persistence MAE | Seasonal naive MAE | XGBoost MAE | Skill vs persistence |
|---|---|---|---|---|
| 1 | 17.7 | 34.2 | 6.6 | 0.63 |
| 3 | 26.2 | 33.8 | 13.9 | 0.47 |
| 6 | 35.4 | 35.0 | 21.6 | 0.39 |
| 12 | 43.1 | 35.0 | 24.1 | 0.44 |
| 24 | 35.1 | 35.1 | 25.3 | 0.28 |
| 36 | 47.7 | 40.9 | 27.1 | 0.43 |
| 48 | 41.0 | 41.0 | 28.1 | 0.32 |

At 24 and 48 hours persistence and seasonal naive are the same forecast (same hour yesterday).

**AQI category of the 24-hour mean, final holdout (1 Sep to 12 Nov 2026, 1,440 forecasts per horizon)**

| Horizon (h) | Accuracy: persistence | seasonal naive | XGBoost | Poor-or-worse recall: persistence | seasonal naive | XGBoost |
|---|---|---|---|---|---|---|
| 1 | 0.993 | 0.967 | 0.990 | 1.000 | 0.989 | 1.000 |
| 6 | 0.918 | 0.839 | 0.950 | 0.913 | 0.890 | 0.950 |
| 12 | 0.746 | 0.706 | 0.852 | 0.777 | 0.818 | 0.872 |
| 24 | 0.469 | 0.540 | 0.656 | 0.603 | 0.655 | 0.723 |
| 36 | 0.388 | 0.471 | 0.533 | 0.574 | 0.603 | 0.605 |
| 48 | 0.361 | 0.401 | 0.490 | 0.542 | 0.546 | 0.558 |

At 24 h the model is within one category 98.5% of the time (persistence 91.6%); at 48 h, 96.5% (persistence 83.6%).

**80% prediction intervals, final holdout.** Raw quantile models covered 57% to 69%. After recalibrating the width using walk-forward residuals, coverage is 76% to 83% across horizons. Mean width is about 15 ug/m3 at 1 h, 66 at 24 h and 75 at 48 h. Horizons that were not evaluated individually use interpolated width factors.

**What to take from this**
- The model clearly beats persistence and seasonal naive on PM2.5 error up to 48 h, and on AQI category up to about 24 h.
- Beyond about a day, the ability to flag Poor-or-worse air is no better than the baselines. Recall was measured without false-alarm rates.
- Without forecast-weather inputs, 24 h MAE was about 9% worse (27.1 vs 24.8 on every second fold).

**Caveats.** Training and validation use actual ERA5 weather in place of weather forecasts, so these results are optimistic for real use. The holdout covers only about 10 weeks. The five locations are strongly correlated (0.80 to 0.88), so treat results as city-level. SARIMAX was evaluated on one location over low-pollution months, 184 origins, and was not consistently better than persistence. Prophet was not run in this environment.

## Quick start (real data, about 1 hour)

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11 to 3.13
make install
make all        # download, clean, tune, evaluate, slow-models, train
make report     # writes docs/model_comparison.md
make test
make api        # http://localhost:8000/docs
make dashboard  # http://localhost:8501
```

Prophet is the most fragile install. If it fails, remove it from `requirements.txt`; the slow-model step then skips Prophet and everything else still works. Docker/Linux avoids most install problems: `docker compose up --build`.

Try it offline with synthetic data: `make demo-data && make tune evaluate train`. Synthetic runs are labelled `SYNTHETIC` in the API, dashboard and report, and their numbers mean nothing about Delhi.

## How it works

```
Open-Meteo (air quality + ERA5 weather + weather forecast)
  -> fetch.py     yearly chunks, cached Parquet, resumable, retries
  -> clean.py     full hourly grid, gaps <= 3h interpolated and flagged, long gaps left missing
  -> features.py  lags, rolling stats, weather, calendar, Diwali / stubble flags, forecast weather
  -> validate.py  monthly expanding-window walk-forward + untouched holdout
  -> models       persistence, seasonal naive, SARIMAX, Prophet, XGBoost per horizon, LightGBM quantiles
  -> forecast.py  48 hourly values with 80% interval and AQI category
  -> FastAPI + Streamlit, packaged with Docker
```

Key decisions:

- **Direct multi-horizon.** One XGBoost model per horizon (1, 3, 6, 12, 24, 36, 48 h), interpolated for hours in between, so errors do not compound as they would when feeding predictions back.
- **Time zones.** UTC everywhere internally; IST only for display and calendar features.
- **AQI category** uses the 24-hour rolling mean of PM2.5 and CPCB breakpoints (top band capped at 380 ug/m3 for 500; check against CPCB's table). For forecasts, the 24-hour mean at the target time mixes observed and predicted hours.
- **Walk-forward.** Last 12 months scored month by month with retraining, origins every 6 hours, final 3 months held out until the end. Training rows whose target falls after the cut-off are excluded.
- **Intervals.** LightGBM 10th/90th percentile models, widened per horizon using walk-forward residuals so coverage reaches about 80%. Check coverage in the report; it is measured, not assumed.
- **SARIMAX and Prophet** are evaluated on one location with daily origins because they refit slowly. SARIMAX refits monthly and re-filters recent data per origin; Prophet is fitted once per month and does not use recent observations.
- At 24 and 48 hours, persistence and seasonal naive are the same forecast, so skill versus persistence is also skill versus seasonal naive there.

## Known limitations

- **Optimistic weather.** Training and validation use actual ERA5 weather in place of weather forecasts, so measured accuracy at longer horizons is better than a real deployment would see. Ablation `no forecast weather` shows how much the model leans on it.
- Modelled data, five nearby grid points that are strongly correlated (see the audit); treat results as city-level.
- Festival dates in `config/festivals.csv` were entered by hand; verify them.
- Boundary-layer height was not used because it is not in the history download.
- Notebooks are thin and unexecuted; run them after `make all`. EDA figures come from `python -m aqi.eda`.

## Layout

See `src/aqi/` (pipeline code), `app/dashboard.py`, `tests/` (leakage, splitter, AQI conversion, caching, API schema), `docs/`, `reports/`.

Data: Open-Meteo, Copernicus Atmosphere Monitoring Service (CAMS) and ERA5 (Copernicus Climate Change Service). Please cite them when reusing.


## Windows without make

`make` is not installed on Windows by default. From the project folder with the virtual environment active, run the commands directly:

```powershell
$env:PYTHONPATH = "$PWD\src"
pip install -r requirements.txt
python -m aqi.pipeline download
python -m aqi.pipeline clean
python -m aqi.tune
python -m aqi.evaluate
python -m aqi.slow_eval --folds 6      # add --no-prophet if Prophet is not installed
python -m aqi.train_final
python -m aqi.ablation
python -m aqi.eda
python -m aqi.report
$env:AQI_LIVE = "1"                     # refresh live data when the dashboard loads
streamlit run app/dashboard.py
```
