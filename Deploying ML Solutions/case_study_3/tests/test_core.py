"""Small checks that do not need the 789 MB extract."""

from __future__ import annotations

import numpy as np

from pipeline.features import engineer_sensor, source_flags
from pipeline.rules import classify_rules
from pipeline.sequence_models import numerical_gradient_ok
from pipeline.shap_analysis import mean_abs_by_class


def test_source_flags_cover_the_survey_strings():
    assert source_flags("Grid w/Solar & Generator Backup") == (1, 1, 1, 3)
    assert source_flags("Solar") == (0, 1, 0, 1)
    assert source_flags("Generator") == (0, 0, 1, 1)
    assert source_flags("Solar & Generator") == (0, 1, 1, 2)


def test_trailing_mad_on_a_flat_series_is_zero():
    time_s = np.arange(12, dtype=np.int64) * 120
    voltage = np.full(12, 230.0)
    frequency = np.full(12, 50.0)
    keys = (time_s // 120) * 120
    medians = np.full(len(keys), 50.05, np.float32)
    counts = np.full(len(keys), 4, np.float32)
    features, centered, site_median = engineer_sensor(
        time_s,
        voltage,
        frequency,
        np.ones(12),
        np.ones(12),
        np.ones(12),
        keys,
        medians,
        counts,
    )
    assert features.shape == (12, 24)
    assert features[-1, 8] == 0  # trail_mad_freq
    assert centered[-1] == 0
    assert abs(float(site_median[-1]) - 50.05) < 1e-5


def test_single_source_solar_is_not_relabelled_by_frequency():
    # Frequency is outside the solar band, but the room has only solar installed.
    features = np.zeros((1, 24), np.float32)
    features[0, 1] = 47.0  # frequency
    features[0, 18] = 0  # has_grid
    features[0, 19] = 1  # has_solar
    features[0, 20] = 0  # has_generator
    features[0, 21] = 1  # n_sources
    labels = classify_rules(features, mad=np.array([0.5]), site_median=np.array([50.0]), is_pcmh=False, is_ola=False)
    assert int(labels[0]) == 2  # solar


def test_multi_source_grid_rule_uses_the_0_11_hz_band():
    features = np.zeros((1, 24), np.float32)
    features[0, 1] = 50.0
    features[0, 12] = 0.05  # freq_dev_site_median
    features[0, 13] = 3  # n_sensors
    features[0, 18] = 1
    features[0, 19] = 1
    features[0, 20] = 1
    features[0, 21] = 3
    labels = classify_rules(
        features, mad=np.array([0.0]), site_median=np.array([50.0]), is_pcmh=False, is_ola=False
    )
    assert int(labels[0]) == 1  # grid


def test_lstm_and_rnn_gradients_match_a_finite_difference():
    assert numerical_gradient_ok("rnn")
    assert numerical_gradient_ok("lstm")


def test_mean_abs_shap_ranks_the_feature_that_moves():
    values = np.zeros((4, 24, 3), np.float64)
    values[:, 0, 0] = 2.0
    values[:, 1, 1] = -3.0
    ranking = mean_abs_by_class(values)
    assert next(iter(ranking["generator"])) == "voltage"
    assert next(iter(ranking["grid"])) == "frequency"
    assert ranking["generator"]["voltage"] == 2.0
