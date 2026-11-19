import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from aqi import aqi_index, clean, config

log = logging.getLogger(__name__)


def city_series(hourly: pd.DataFrame) -> pd.DataFrame:
    return hourly.groupby(level=0)[
        ["pm2_5", "wind_speed_10m", "relative_humidity_2m", "temperature_2m", "precipitation"]
    ].mean()


def save(fig, name: str) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / f"eda_{name}.png", dpi=120)
    plt.close(fig)


def seasonality_plots(city: pd.DataFrame) -> dict[str, str]:
    ist = city.index.tz_convert("Asia/Kolkata")
    monthly = city.groupby(ist.month)["pm2_5"].mean()
    hourly = city.groupby(ist.hour)["pm2_5"].mean()
    weekday = city.groupby(ist.day_of_week >= 5)["pm2_5"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    monthly.plot.bar(ax=axes[0], color="#555")
    axes[0].set(title="Mean PM2.5 by month", xlabel="month", ylabel="ug/m3")
    hourly.plot(ax=axes[1], color="#555")
    axes[1].set(title="Mean PM2.5 by hour of day (IST)", xlabel="hour")
    weekday.rename({False: "weekday", True: "weekend"}).plot.bar(ax=axes[2], color="#555")
    axes[2].set(title="Weekday vs weekend")
    save(fig, "seasonality")
    return {
        "monthly": f"Highest month {monthly.idxmax()} ({monthly.max():.0f}), lowest month {monthly.idxmin()} ({monthly.min():.0f}) ug/m3.",
        "hourly": f"Daily peak at {hourly.idxmax()}:00 IST ({hourly.max():.0f}), trough at {hourly.idxmin()}:00 IST ({hourly.min():.0f}) ug/m3.",
        "weekend": f"Weekend mean {weekday[True]:.0f} vs weekday {weekday[False]:.0f} ug/m3.",
    }


def category_by_month(city: pd.DataFrame) -> str:
    rolled = aqi_index.rolling_24h_mean(city["pm2_5"]).dropna()
    codes = pd.Series(aqi_index.category_code(rolled.to_numpy()), index=rolled.index)
    ist_month = codes.index.tz_convert("Asia/Kolkata").month
    share = pd.crosstab(ist_month, codes, normalize="index")
    share = share.reindex(columns=range(6), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    share.plot.bar(
        stacked=True, ax=ax, color=["#00B050", "#92D050", "#FFD700", "#FF9900", "#E8212B", "#8B0000"], width=0.85
    )
    ax.set(title="Share of hours in each AQI category, by month", xlabel="month", ylabel="share")
    ax.legend(aqi_index.CATEGORIES, fontsize=7, ncol=3)
    save(fig, "category_by_month")
    worst = (share[[3, 4, 5]].sum(axis=1)).idxmax()
    return f"Poor-or-worse share peaks in month {worst} at {share[[3, 4, 5]].sum(axis=1).max():.0%} of hours."


def diwali_effect(city: pd.DataFrame) -> str:
    festivals = config.load_festivals()
    dates = festivals.loc[festivals["name"] == "Diwali", "date"]
    ist_days = city.index.tz_convert("Asia/Kolkata").normalize().tz_localize(None)
    daily = city["pm2_5"].groupby(ist_days).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    summary = []
    for date in dates:
        window = daily.loc[date - pd.Timedelta(days=7) : date + pd.Timedelta(days=7)]
        if len(window) < 15:
            continue
        ax.plot(range(-7, 8), window.to_numpy(), label=str(date.year))
        summary.append(
            (
                date.year,
                window.loc[date : date + pd.Timedelta(days=2)].mean()
                / window.loc[: date - pd.Timedelta(days=3)].mean(),
            )
        )
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Daily mean PM2.5 around Diwali", xlabel="days from Diwali", ylabel="ug/m3")
    ax.legend()
    save(fig, "diwali")
    if not summary:
        return "No complete Diwali window in the data range."
    parts = ", ".join(f"{y}: x{r:.2f}" for y, r in summary)
    return f"Mean PM2.5 on Diwali and the two days after, relative to the days before it: {parts}."


def weather_relationships(city: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    out = []
    for ax, column in zip(axes, ["wind_speed_10m", "relative_humidity_2m", "temperature_2m", "precipitation"]):
        sample = city[[column, "pm2_5"]].dropna().sample(min(4000, len(city)), random_state=0)
        ax.scatter(sample[column], sample["pm2_5"], s=3, alpha=0.3, color="#444")
        ax.set(xlabel=column, ylabel="PM2.5")
        out.append(f"{column} {city[column].corr(city['pm2_5'], method='spearman'):+.2f}")
    save(fig, "weather")
    return "Spearman correlation with PM2.5: " + ", ".join(out) + "."


def autocorrelation(city: pd.DataFrame) -> str:
    series = np.log1p(city["pm2_5"].dropna())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    plot_acf(series, lags=96, ax=axes[0])
    plot_pacf(series, lags=48, ax=axes[1], method="ywm")
    save(fig, "acf_pacf")
    acf24 = series.autocorr(24)
    return f"Autocorrelation of log PM2.5 at 24 hours is {acf24:.2f}, at 168 hours {series.autocorr(168):.2f}."


def stationarity_and_decomposition(city: pd.DataFrame) -> str:
    series = np.log1p(city["pm2_5"].interpolate(limit=3).dropna())
    adf_p = adfuller(series.iloc[-24 * 365 :], autolag="AIC")[1]
    recent = series.iloc[-24 * 120 :]
    parts = seasonal_decompose(recent, period=24, model="additive")
    fig = parts.plot()
    fig.set_size_inches(10, 7)
    save(fig, "decomposition")
    strength = 1 - parts.resid.var() / (parts.resid + parts.seasonal).var()
    return f"ADF p-value on the last year of log PM2.5: {adf_p:.3g}. Daily seasonal strength in the last 120 days: {strength:.2f}."


def run_all() -> dict[str, str]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    hourly = clean.load_processed()
    city = city_series(hourly)
    findings = {}
    findings.update(seasonality_plots(city))
    findings["categories"] = category_by_month(city)
    findings["diwali"] = diwali_effect(city)
    findings["weather"] = weather_relationships(city)
    findings["autocorrelation"] = autocorrelation(city)
    findings["stationarity"] = stationarity_and_decomposition(city)
    audit = clean.audit(hourly)
    corr = audit["correlation"].to_numpy()
    findings["location_similarity"] = (
        f"Lowest correlation between two locations' PM2.5 is {corr.min():.2f}, mean {corr[np.triu_indices_from(corr, 1)].mean():.2f}."
    )
    for key, text in findings.items():
        log.info("%s: %s", key, text)
    pd.Series(findings).to_csv(config.REPORTS_DIR / "eda_findings.csv", header=["finding"])
    return findings


if __name__ == "__main__":
    run_all()
