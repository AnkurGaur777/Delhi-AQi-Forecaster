import numpy as np
import pandas as pd

from aqi import validate


def test_folds_never_test_before_training_end():
    end = pd.Timestamp("2026-11-12 23:00", tz="UTC")
    folds, holdout = validate.walk_forward_folds(end)
    for fold in folds:
        assert fold.test_start >= fold.train_end
        assert fold.test_end > fold.test_start
    starts = [f.test_start for f in folds]
    assert starts == sorted(starts)
    assert folds[-1].test_end <= holdout.test_start
    assert holdout.test_end > end


def test_train_mask_respects_the_horizon_embargo():
    index = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    cutoff = index[100]
    mask = validate.train_mask(index, cutoff, 24)
    assert index[mask].max() + pd.Timedelta(hours=24) <= cutoff


def test_train_and_test_origins_do_not_overlap():
    end = pd.Timestamp("2026-11-12 23:00", tz="UTC")
    folds, _ = validate.walk_forward_folds(end)
    index = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC"), end, freq="h")
    fold = folds[3]
    train = validate.train_mask(index, fold.train_end, 48)
    test = validate.test_mask(index, fold)
    assert not (train & test).any()
    assert index[test].min() >= fold.train_end


def test_point_metrics_and_skill():
    assert validate.mae([1, 2, 3], [2, 2, 5]) == 1.0
    assert validate.rmse([0, 0], [3, 4]) == np.sqrt(12.5)
    assert validate.skill_score(5.0, 10.0) == 0.5


def test_interval_coverage():
    result = validate.interval_metrics([1, 5, 9], [0, 4, 10], [2, 6, 12])
    assert result["coverage"] == 2 / 3


def test_category_recall_counts_poor_or_worse_only():
    true = np.array([0, 3, 4, 5, 1])
    pred = np.array([0, 3, 2, 5, 1])
    result = validate.category_metrics(true, pred)
    assert result["poor_or_worse_recall"] == 2 / 3


def test_predicted_24h_mean_uses_observed_then_forecast():
    observed = np.full(24, 100.0)
    means = validate.predicted_24h_means(observed, [1, 24], np.array([100.0, 100.0]))
    assert np.allclose(means, 100.0)
    low = validate.predicted_24h_means(observed, [24], np.array([0.0]))
    assert low[0] < 100.0
