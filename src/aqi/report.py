import json
import logging

import pandas as pd

from aqi import config

log = logging.getLogger(__name__)
DOCS_DIR = config.ROOT / "docs"


def read(name: str) -> pd.DataFrame | None:
    path = config.REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def to_markdown(table: pd.DataFrame, digits: int = 3) -> str:
    table = table.round(digits)
    for column in ("horizon", "n"):
        if column in table:
            table[column] = table[column].astype(int)
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    divider = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, divider, *rows])


def pivot_mae(table: pd.DataFrame) -> pd.DataFrame:
    wide = table.pivot(index="horizon", columns="model", values="mae")
    wide = wide.reset_index()
    wide["horizon"] = wide["horizon"].astype(int)
    return wide


def section(title: str, table: pd.DataFrame | None, note: str = "") -> str:
    if table is None or table.empty:
        return f"## {title}\n\nNot available. Run the matching make target first.\n"
    return f"## {title}\n\n{note}\n\n{to_markdown(table)}\n"


def build() -> str:
    meta_path = config.RAW_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    banner = ""
    if meta.get("source") == "SYNTHETIC":
        banner = "> **These numbers come from SYNTHETIC demo data and say nothing about real Delhi air quality.**\n\n"

    parts = [
        "# Model comparison\n",
        banner,
        f"Data: {meta.get('source', 'unknown')}, {meta.get('start', '?')} to {meta.get('end', '?')}.\n",
        "All errors are MAE in ug/m3 unless noted. Skill is 1 minus the model MAE divided by the persistence MAE.\n",
    ]
    wf = read("walk_forward_horizon.csv")
    parts.append(
        section(
            "Walk-forward validation: MAE by horizon (all locations)",
            pivot_mae(wf) if wf is not None else None,
            "Monthly expanding-window folds, forecast origins every 6 hours.",
        )
    )
    parts.append(
        section(
            "Skill versus persistence, walk-forward",
            wf[wf["model"] != "persistence"][["model", "horizon", "mae", "rmse", "skill_vs_persistence"]]
            if wf is not None
            else None,
        )
    )
    parts.append(section("Walk-forward by season", read("walk_forward_season.csv")))
    parts.append(
        section(
            "AQI category accuracy, walk-forward",
            read("walk_forward_category.csv"),
            "Category of the 24-hour mean PM2.5 at the target time.",
        )
    )
    parts.append(
        section(
            "Prediction interval (80%) coverage, walk-forward",
            read("walk_forward_interval.csv"),
            "Target coverage is 0.80.",
        )
    )
    parts.append(
        section(
            "Final holdout: MAE by horizon",
            pivot_mae(read("holdout_horizon.csv")) if read("holdout_horizon.csv") is not None else None,
            "Scored once, after model selection.",
        )
    )
    parts.append(section("Final holdout: AQI category accuracy", read("holdout_category.csv")))
    parts.append(section("Final holdout: interval coverage", read("holdout_interval.csv")))
    parts.append(
        section(
            "SARIMAX and Prophet (one location, daily origins)",
            read("slow_models_horizon.csv"),
            "Evaluated on one location because both models are slow to refit.",
        )
    )
    parts.append(section("XGBoost ablations at 24 hours", read("ablation_h24.csv")))
    importance = read("shap_h24_top20.csv")
    parts.append(
        section("Top features by mean absolute SHAP, 24 h", importance.head(10) if importance is not None else None)
    )
    return "\n".join(p for p in parts if p)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "model_comparison.md").write_text(build())
    log.info("wrote docs/model_comparison.md")


if __name__ == "__main__":
    main()
