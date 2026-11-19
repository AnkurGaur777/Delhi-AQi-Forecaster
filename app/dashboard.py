import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from aqi import aqi_index, config, forecast

API_URL = os.getenv("AQI_API_URL")
COLOURS = ["#00B050", "#92D050", "#FFD700", "#FF9900", "#E8212B", "#8B0000"]
BANDS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 250), (250, 400)]
DISCLAIMER = (
    "Educational project. Values are modelled air quality from Open-Meteo (Copernicus CAMS), not CPCB sensor readings. "
    "This is not an official forecast and must not be used for health decisions."
)

st.set_page_config(page_title="Delhi AQI 48-hour forecast", layout="wide")


def to_ist(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_convert("Asia/Kolkata")


@st.cache_data(ttl=900, show_spinner="Computing forecast")
def get_forecast(slug: str) -> dict:
    if API_URL:
        response = requests.get(f"{API_URL}/forecast", params={"location": slug}, timeout=60)
        response.raise_for_status()
        return response.json()
    return forecast.forecast_location(slug)


@st.cache_data(ttl=3600)
def load_report(name: str) -> pd.DataFrame | None:
    path = Path(config.REPORTS_DIR) / name
    return pd.read_csv(path) if path.exists() else None


def forecast_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    top = max(frame["pm25_hi"].max() * 1.1, 130)
    for (low, high), colour in zip(BANDS, COLOURS):
        if low < top:
            fig.add_hrect(y0=low, y1=min(high, top), fillcolor=colour, opacity=0.12, line_width=0)
    fig.add_trace(go.Scatter(x=frame["ist"], y=frame["pm25_hi"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=frame["ist"],
            y=frame["pm25_lo"],
            fill="tonexty",
            fillcolor="rgba(70,70,70,0.18)",
            line={"width": 0},
            name="80% interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["ist"],
            y=frame["pm25"],
            mode="lines+markers",
            line={"color": "#333", "width": 2},
            marker={
                "size": 7,
                "color": [COLOURS[c] for c in aqi_index.category_code(frame["pm25"].to_numpy())],
                "line": {"width": 0.5, "color": "#333"},
            },
            customdata=frame[["aqi", "category"]],
            hovertemplate="%{x|%a %H:%M}<br>PM2.5 %{y:.0f} ug/m3<br>24h-mean AQI %{customdata[0]} (%{customdata[1]})<extra></extra>",
            name="PM2.5 forecast",
        )
    )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis={"title": "PM2.5 (ug/m3)", "range": [0, top]},
        xaxis={"title": "Time (IST)"},
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def performance_tab() -> None:
    horizon = load_report("holdout_horizon.csv")
    source = "final holdout"
    if horizon is None:
        horizon = load_report("walk_forward_horizon.csv")
        source = "walk-forward validation"
    if horizon is None:
        st.info("No validation results found. Run `make evaluate` and `make train` first.")
        return
    st.caption(f"Results from {source}. Lower MAE is better; skill is the gain over persistence ('same as now').")
    fig = go.Figure()
    for model, part in horizon.groupby("model"):
        fig.add_trace(go.Scatter(x=part["horizon"], y=part["mae"], mode="lines+markers", name=model))
    fig.update_layout(height=360, xaxis_title="Forecast horizon (hours)", yaxis_title="MAE (ug/m3)", margin={"t": 10})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(horizon.round(3), width="stretch", hide_index=True)
    for title, name in [("AQI category accuracy", "category"), ("Prediction interval coverage", "interval")]:
        table = load_report(f"{'holdout' if source == 'final holdout' else 'walk_forward'}_{name}.csv")
        if table is not None:
            st.subheader(title)
            st.dataframe(table.round(3), width="stretch", hide_index=True)


def main() -> None:
    locations = config.load_locations()
    names = dict(zip(locations["name"].map(config.location_slug), locations["name"]))
    st.title("Delhi air quality: 48-hour PM2.5 and AQI forecast")

    slug = st.sidebar.selectbox(
        "Location", list(names), format_func=lambda s: names[s].title().replace("Ito", "ITO").replace("Rk", "RK")
    )
    st.sidebar.caption("Grid points near these areas, not CPCB stations.")
    st.sidebar.warning(DISCLAIMER)

    forecast_tab, performance, about = st.tabs(["Forecast", "Model performance", "About the data"])

    with forecast_tab:
        try:
            result = get_forecast(slug)
        except Exception as error:
            st.error(f"Could not produce a forecast: {error}")
            return
        if result["data_source"] == "SYNTHETIC":
            st.error("Demo mode: this forecast is built from SYNTHETIC data, not real air quality.")
        if result["stale"]:
            st.warning(f"Latest data is {result['data_age_hours']:.0f} hours old. The forecast starts from that time.")

        frame = pd.DataFrame(result["forecast"])
        frame["ist"] = pd.to_datetime(frame["time"]).dt.tz_convert("Asia/Kolkata")
        extremes = forecast.worst_and_best(result["forecast"])
        window = forecast.best_window(result["forecast"])
        current, worst = result["current"], extremes["worst"]

        cols = st.columns(4)
        cols[0].metric("Now (24h mean)", f"AQI {current['aqi']}", current["category"])
        cols[1].metric(
            "Worst 24h-avg AQI, next 48 h",
            worst["category"],
            f"AQI {worst['aqi']} at {to_ist(worst['time']):%a %H:%M}",
            delta_color="off",
        )
        if window:
            cols[2].metric(
                "Cleanest 3 hours (hourly PM2.5)",
                f"{to_ist(window['start']):%a %H:%M}",
                f"mean PM2.5 {window['mean_pm25']:.0f} ug/m3",
                delta_color="off",
            )
        cols[3].metric(
            "Latest data",
            f"{to_ist(result['latest_data_time']):%d %b %H:%M} IST",
            f"model {result['model_version']}",
            delta_color="off",
        )

        st.plotly_chart(forecast_chart(frame), width="stretch")
        st.info(forecast.advice_for(worst["category_code"]))
        legend = "  ".join(
            f":{'green' if i < 2 else 'orange' if i < 4 else 'red'}[{aqi_index.CATEGORIES[i]}]" for i in range(6)
        )
        st.caption(f"Categories use the 24-hour mean PM2.5 (CPCB): {legend}")

    with performance:
        performance_tab()

    with about:
        meta = forecast.data_metadata()
        st.markdown(
            f"""
**Source:** Open-Meteo Air Quality API (Copernicus CAMS) and Historical Weather API (ERA5). Please cite Open-Meteo, CAMS and ERA5 when you reuse this.

**Data range:** {meta.get("start", "unknown")} to {meta.get("end", "unknown")}, downloaded {meta.get("downloaded_at", "unknown")}.

**What this is:** a forecast of a modelled air-quality series on a coarse grid. It is smoother than real station readings and may miss Delhi's worst spikes.

**Forecast weather caveat:** models were trained with actual weather standing in for weather forecasts, so measured accuracy is somewhat optimistic.

{DISCLAIMER}
"""
        )


main()
