import numpy as np
import pytest

from aqi import aqi_index


@pytest.mark.parametrize(
    ("pm25", "expected"),
    [(0, 0), (30, 50), (60, 100), (90, 200), (120, 300), (250, 400), (380, 500), (45, 75), (105, 250)],
)
def test_sub_index_hand_values(pm25, expected):
    assert aqi_index.pm25_to_aqi(pm25) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("pm25", "code"),
    [(10, 0), (30, 0), (31, 1), (60, 1), (75, 2), (91, 3), (120, 3), (121, 4), (250, 4), (251, 5), (600, 5)],
)
def test_category_boundaries(pm25, code):
    assert aqi_index.category_code(pm25) == code


def test_nan_has_no_category():
    assert aqi_index.category_code(np.nan) == -1


def test_sub_index_is_capped_at_500():
    assert aqi_index.pm25_to_aqi(900) == 500


def test_rolling_mean_needs_16_hours():
    import pandas as pd

    series = pd.Series(np.arange(30, dtype=float))
    series.iloc[:10] = np.nan
    rolled = aqi_index.rolling_24h_mean(series)
    assert rolled.iloc[26:].notna().all()
    assert rolled.iloc[:25].isna().all()
