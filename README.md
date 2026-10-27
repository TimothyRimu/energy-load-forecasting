# Day-Ahead Electricity Load Forecasting: Energy Cooperative (Aveiro, Portugal)

Hourly day-ahead forecast of electricity use for a local energy cooperative, built from the public **Loureiro dataset** (172 smart meters, 15-minute readings, May 2022 to Sep 2023) joined with Aveiro weather data. Completed as part of my Certified Data Scientist training (IABAC / AFRICDSA); this write-up and code were produced with AI assistance and I reviewed the results.

## Result (walk-forward, Dec 2022 to May 2023, 4,296 hourly test points)

| Model | MAE | MAPE |
|---|---|---|
| Same hour yesterday (baseline) | 0.00927 | 9.6% |
| Same hour last week (baseline) | 0.00952 | 9.8% |
| Ridge regression | 0.00762 | 8.1% |
| **Gradient boosting** | **0.00722** | **7.5%** |

Gradient boosting cuts error by about **22% against the best naive baseline**. It was better than the baseline in every test month. See `forecast_results.png` and `results.json`.

**Weather helps, but modestly.** Removing the weather features raises gradient boosting's MAE from 0.00722 to 0.00767 (about 6% worse); Ridge is unchanged (`results_no_weather.json`).

## Method
- **Target:** mean energy per *reporting* meter, per hour. The number of meters reporting changes a lot over time (about 150 in winter, 40 to 110 from June 2023), so a plain sum across meters would show fake drops. Rows with fewer than 100 reporting meters were dropped.
- **Window:** Oct 2022 to May 2023, the period with stable meter coverage.
- **Features (all known at forecast time):** consumption 24h, 48h and 168h earlier, mean of the previous 24h, hour of day (sine/cosine), day of week, weekend flag, temperature, humidity, wind speed, solar radiation, precipitation.
- **Validation:** expanding-window walk-forward, one fold per month (Dec to May), retraining before each fold. No shuffling, no future data in training.

## Limitations (please read)
- Weather features are **observed** weather, not forecasts. A real deployment would use forecast weather, so real-world accuracy would be somewhat worse and the weather benefit smaller.
- One cooperative, one winter-to-spring period. No full-year seasonality is tested (summer 2023 is excluded because meter coverage collapses).
- Energy units are as given in the dataset; the source files do not document them, so errors are shown in raw units and as MAPE.
- Gains over the baseline are real but moderate; this is a clean, honest baseline study, not a production system.

## Run it
```
pip install pandas numpy scikit-learn matplotlib
python forecast.py path/to/LoureiroDataset
```
Data: Loureiro et al., electricity consumption dataset of a local energy cooperative (public). The raw CSVs are not included here (36 MB); download them from the original source.
