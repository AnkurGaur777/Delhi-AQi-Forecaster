import json
import logging
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from aqi import clean, config, evaluate, features, tracking, validate
from aqi.models import boosting

log = logging.getLogger("aqi.train_final")
BUNDLE_NAME = "bundle.joblib"


def fit_bundle(
    tables: dict[int, pd.DataFrame], train_end: pd.Timestamp, params: dict, scales: dict[int, float]
) -> dict:
    fitted = {}
    for h, table in tables.items():
        cols = features.feature_columns(table)
        fold = validate.Fold("final", train_end, train_end, train_end)
        train = evaluate.training_rows(table, fold, h)
        y = train["target"].to_numpy(dtype=float)
        fitted[h] = {
            "columns": cols,
            "point": boosting.fit_xgb(train[cols], y, params),
            "lo": boosting.fit_quantile(train[cols], y, 0.1),
            "hi": boosting.fit_quantile(train[cols], y, 0.9),
            "interval_scale": scales.get(h, 1.0),
        }
        log.info("fitted horizon %d on %d rows", h, len(train))
    return fitted


def save_shap_summary(fitted: dict, tables: dict[int, pd.DataFrame], horizon: int = 24) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    cols = fitted[horizon]["columns"]
    sample = tables[horizon][cols].dropna(thresh=len(cols) // 2).sample(2000, random_state=0)
    explainer = shap.TreeExplainer(fitted[horizon]["point"].model)
    values = explainer.shap_values(sample)
    shap.summary_plot(values, sample, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "shap_h24.png", dpi=130)
    plt.close()
    importance = pd.Series(np.abs(values).mean(axis=0), index=cols).sort_values(ascending=False)
    importance.head(20).rename("mean_abs_shap").to_csv(config.REPORTS_DIR / "shap_h24_top20.csv")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    hourly = clean.load_processed()
    data_end = hourly.index.max()
    _, holdout = validate.walk_forward_folds(data_end)
    params = evaluate.load_best_params()
    tables = evaluate.horizon_tables(hourly, config.SERVING_HORIZONS)

    with tracking.run("final_holdout", {"holdout_start": holdout.test_start, "xgb_params": params}) as mlf:
        walk_forward = pd.read_parquet(config.REPORTS_DIR / "walk_forward_predictions.parquet")
        scales = evaluate.interval_scales(walk_forward)
        known = sorted(scales)
        scales = {h: round(float(np.interp(h, known, [scales[k] for k in known])), 2) for h in config.SERVING_HORIZONS}
        log.info("interval scales from walk-forward residuals: %s", scales)
        preds = evaluate.run_walk_forward(hourly, tables, [holdout], True, params, label="holdout")
        preds = evaluate.apply_scales(preds, scales)
        results = evaluate.save_reports(preds, hourly, "holdout")
        for _, row in results["horizon"].iterrows():
            tracking.log_metrics(mlf, {f"holdout_mae_{row['model']}_h{int(row['horizon'])}": row["mae"]})
        print(results["horizon"].round(3).to_string())

        fitted = fit_bundle(tables, data_end + pd.Timedelta(hours=1), params, scales)
        meta = {
            "trained_until": data_end.isoformat(),
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "horizons": config.SERVING_HORIZONS,
            "interval_scales": {str(k): v for k, v in scales.items()},
            "model_version": datetime.now(UTC).strftime("%Y%m%d"),
            "data_source": json.loads((config.RAW_DIR / "metadata.json").read_text())["source"],
        }
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"fitted": fitted, "meta": meta}, config.MODEL_DIR / BUNDLE_NAME)
        (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
        save_shap_summary(fitted, tables)


if __name__ == "__main__":
    main()
