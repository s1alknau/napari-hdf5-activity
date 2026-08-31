"""Tests for the temperature control analysis."""

import numpy as np
import pytest

from napari_hdf5_activity import _temperature as T
from napari_hdf5_activity._batch import make_composite_id

BIN = 600.0                       # 10 min bins
N = int(120 * 3600 / BIN)         # 120 h recording
TIMES = np.arange(N) * BIN
CIRCADIAN = 0.5 + 0.3 * np.cos(2 * np.pi * TIMES / (24 * 3600))


def series(values):
    return [(float(a), float(b)) for a, b in zip(TIMES, values)]


@pytest.fixture
def flat_temperature():
    """A well-controlled incubator: no rhythm, a few hundredths of a degree."""
    rng = np.random.default_rng(0)
    return 20.0 + rng.normal(0, 0.05, N)


@pytest.fixture
def cycling_temperature():
    """A rig whose temperature swings daily — the confounded case."""
    rng = np.random.default_rng(1)
    return (
        20.0
        + 2.0 * np.cos(2 * np.pi * TIMES / (24 * 3600))
        + rng.normal(0, 0.05, N)
    )


def env_for(values):
    return {1: {"times": TIMES.tolist(), "temperature": np.asarray(values).tolist()}}


def endogenous_activity(n_roi=4, seed=2):
    rng = np.random.default_rng(seed)
    return {
        r: series(CIRCADIAN + rng.normal(0, 0.1, N)) for r in range(1, n_roi + 1)
    }


# ------------------------------------------------------------------ statistics

def test_temperature_statistics(flat_temperature):
    stats = T.temperature_statistics(flat_temperature)
    assert abs(stats["mean"] - 20.0) < 0.05
    assert stats["sd"] < 0.1
    assert stats["range"] < 0.6
    assert stats["robust_range"] <= stats["range"]
    assert stats["n_samples"] == N


def test_temperature_statistics_handles_empty():
    assert "error" in T.temperature_statistics([])


def test_robust_range_ignores_a_sensor_glitch(flat_temperature):
    spiked = flat_temperature.copy()
    spiked[100] = 80.0
    stats = T.temperature_statistics(spiked)
    assert stats["range"] > 50
    assert stats["robust_range"] < 1.0


# --------------------------------------------------------------- rhythmicity

def test_flat_temperature_is_not_rhythmic(flat_temperature):
    result = T.temperature_rhythmicity(flat_temperature, BIN, 18, 30)
    assert result["is_significant"] is False


def test_cycling_temperature_is_detected(cycling_temperature):
    result = T.temperature_rhythmicity(cycling_temperature, BIN, 18, 30)
    assert result["is_significant"] is True
    assert 22 < result["dominant_period"] < 26


# ------------------------------------------------------------------ alignment

def test_resample_to_interpolates_onto_activity_times():
    src_t = np.arange(0, 1000, 10, dtype=float)
    src_v = src_t * 2.0
    out = T.resample_to(src_t, src_v, np.array([0.0, 55.0, 990.0]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(110.0)
    assert out[2] == pytest.approx(1980.0)


def test_resample_marks_uncovered_span_as_nan():
    """Samples outside the temperature record must not be invented."""
    out = T.resample_to([0.0, 100.0], [20.0, 21.0], [-50.0, 50.0, 500.0])
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(20.5)
    assert np.isnan(out[2])


# --------------------------------------------------------------------- lag

def test_crosscorrelation_recovers_an_injected_lag(cycling_temperature):
    lag_bins = 6                                     # 1 h at 10 min bins
    activity = np.roll(cycling_temperature, lag_bins)
    result = T.crosscorrelate_with_lag(
        activity, cycling_temperature, BIN, max_lag_hours=6
    )
    assert result["best_lag_hours"] == pytest.approx(1.0, abs=0.2)
    assert result["best_r"] > 0.95


def test_crosscorrelation_reports_strong_negative_association():
    rng = np.random.default_rng(4)
    temp = 20.0 + np.cos(2 * np.pi * TIMES / (24 * 3600))
    activity = -2.0 * temp + rng.normal(0, 0.01, N)
    result = T.crosscorrelate_with_lag(activity, temp, BIN, max_lag_hours=2)
    assert result["zero_lag_r"] < -0.95
    assert result["zero_lag_r_squared"] > 0.9


def test_zero_lag_is_the_concurrent_correlation(cycling_temperature):
    """The headline number must be the genuine simultaneous correlation."""
    activity = 0.5 + 0.15 * (cycling_temperature - 20.0)
    result = T.crosscorrelate_with_lag(activity, cycling_temperature, BIN)
    direct = np.corrcoef(activity, cycling_temperature)[0, 1]
    assert result["zero_lag_r"] == pytest.approx(direct, abs=1e-9)


def test_search_window_excludes_the_antiphase_artifact(cycling_temperature):
    """A large negative r near half a period must not become the reported result.

    Two 24 h-periodic signals always correlate strongly at some lag; ranking by
    |r| over a whole cycle manufactures a dramatic number that says nothing.
    """
    rng = np.random.default_rng(9)
    # Endogenous rhythm, in phase with temperature, not caused by it
    activity = 0.5 + 0.3 * np.cos(2 * np.pi * TIMES / (24 * 3600)) \
        + rng.normal(0, 0.05, N)
    result = T.crosscorrelate_with_lag(
        activity, cycling_temperature, BIN, max_lag_hours=6
    )
    # Reported best lag stays inside the plausible window, temperature leading
    assert 0.0 <= result["best_lag_hours"] <= 6.0
    assert result["search_window_hours"] == (0.0, 6.0)
    # The full curve does reach strongly negative values — but not as the result
    assert np.nanmin(result["correlations"]) < -0.5
    assert result["best_r"] > 0


def test_full_curve_spans_the_requested_window(cycling_temperature):
    activity = 0.5 + 0.3 * np.cos(2 * np.pi * TIMES / (24 * 3600))
    result = T.crosscorrelate_with_lag(
        activity, cycling_temperature, BIN, max_lag_hours=6, curve_span_hours=24
    )
    lags = result["lags_hours"]
    assert lags.min() == pytest.approx(-24.0, abs=0.2)
    assert lags.max() == pytest.approx(24.0, abs=0.2)
    assert len(lags) == len(result["correlations"])


# ------------------------------------------------------------------- Q10 bound

@pytest.mark.parametrize(
    "delta_t, q10, expected_ratio",
    [(10.0, 2.0, 2.0), (10.0, 3.0, 3.0), (0.0, 3.0, 1.0), (20.0, 2.0, 4.0)],
)
def test_q10_bound_matches_the_definition(delta_t, q10, expected_ratio):
    result = T.q10_amplitude_bound(delta_t, q10)
    assert result["rate_ratio"] == pytest.approx(expected_ratio)
    assert result["max_modulation"] == pytest.approx(expected_ratio - 1.0)


def test_q10_bound_on_a_small_swing():
    """A few tenths of a degree cannot produce a large behavioural rhythm."""
    result = T.q10_amplitude_bound(0.53, q10=3.0)
    assert result["max_modulation"] < 0.07          # under 7 %


def test_compare_to_q10_bound_flags_an_unexplainable_rhythm():
    # 40 % amplitude = 80 % peak-to-trough, against a 0.53 °C swing
    result = T.compare_to_q10_bound(0.40, 0.53)
    assert result["temperature_sufficient"] is False
    assert result["min_exceedance"] > 10
    # The most generous Q10 gives the smallest exceedance
    exceedances = {q: b["exceedance"] for q, b in result["bounds"].items()}
    assert exceedances[max(exceedances)] == pytest.approx(result["min_exceedance"])


def test_compare_to_q10_bound_accepts_a_sufficient_swing():
    """A large temperature cycle with a small rhythm is NOT flagged."""
    result = T.compare_to_q10_bound(0.02, 5.0)      # 4 % ptp vs 5 °C swing
    assert result["temperature_sufficient"] is True
    assert result["min_exceedance"] < 1.0


# ------------------------------------------------------- inter-individual spread

def test_interindividual_spread_quantifies_scatter():
    result = T.interindividual_spread(
        amplitudes=[0.166, 0.170, 0.084, 0.114],
        mesors=[0.326, 0.313, 0.368, 0.373],
        peak_times=[11.0, 13.4, 13.0, 15.7],
        roi_ids=[2, 3, 4, 5],
    )
    assert result["n"] == 4
    assert result["roi_ids"] == [2, 3, 4, 5]
    assert result["amplitude_ratio"] == pytest.approx(2.38, abs=0.05)
    assert result["phase_range_hours"] == pytest.approx(4.7, abs=0.1)
    assert 1.0 < result["phase_circular_sd_hours"] < 3.0


def test_identical_individuals_show_no_spread():
    """A perfectly shared driver would look like this — the null case."""
    result = T.interindividual_spread(
        amplitudes=[0.2] * 4, mesors=[0.5] * 4, peak_times=[12.0] * 4,
        roi_ids=[1, 2, 3, 4],
    )
    assert result["amplitude_ratio"] == pytest.approx(1.0)
    assert result["phase_range_hours"] == pytest.approx(0.0)
    assert result["phase_circular_sd_hours"] == pytest.approx(0.0, abs=1e-9)
    assert result["resultant_length"] == pytest.approx(1.0)


def test_spread_keeps_ids_aligned_when_an_individual_drops_out():
    """A failed cosinor must not shift the ROI/point pairing."""
    result = T.interindividual_spread(
        amplitudes=[0.2, np.nan, 0.1],
        mesors=[0.5, 0.5, 0.5],
        peak_times=[10.0, 12.0, 14.0],
        roi_ids=[7, 8, 9],
    )
    assert result["roi_ids"] == [7, 9]
    assert result["peak_times"] == [10.0, 14.0]


def test_spread_needs_two_individuals():
    assert "error" in T.interindividual_spread([0.2], [0.5], [12.0])


# -------------------------------------------------------------- regression

def test_regression_removes_temperature_variance(cycling_temperature):
    activity = 0.5 + 0.15 * (cycling_temperature - 20.0)
    result = T.regress_out_temperature(activity, cycling_temperature)
    assert result["r_squared"] > 0.99
    assert np.std(result["residual"]) < 0.01


def test_regression_on_constant_temperature_is_a_noop():
    activity = CIRCADIAN.copy()
    result = T.regress_out_temperature(activity, np.full(N, 20.0))
    assert result["r_squared"] == 0.0
    assert "note" in result


# ------------------------------------------------------- the three scenarios

def test_endogenous_rhythm_survives_flat_temperature(flat_temperature):
    """The result the user is trying to demonstrate."""
    results = T.temperature_control_analysis(
        endogenous_activity(), env_for(flat_temperature),
        sampling_interval=BIN, min_period_hours=18, max_period_hours=30,
    )
    summary = results["summary"]
    assert summary["n_rhythmic_before"] == 4
    assert summary["n_rhythm_survives"] == 4
    assert summary["mean_variance_explained"] < 0.05
    assert summary["residual_test_conclusive"] is True


def test_temperature_driven_activity_does_not_survive(cycling_temperature):
    rng = np.random.default_rng(5)
    activity = {
        r: series(0.5 + 0.15 * (cycling_temperature - 20.0)
                  + rng.normal(0, 0.02, N))
        for r in range(1, 5)
    }
    results = T.temperature_control_analysis(
        activity, env_for(cycling_temperature),
        sampling_interval=BIN, min_period_hours=18, max_period_hours=30,
    )
    summary = results["summary"]
    assert summary["mean_variance_explained"] > 0.9
    assert summary["n_rhythm_survives"] == 0


def test_collinear_case_refuses_to_conclude(cycling_temperature):
    """A real rhythm plus a rhythmic incubator must NOT read as a negative.

    Activity and temperature are collinear at 24 h, so regression cannot
    separate them — reporting "rhythm lost" would be a false negative.
    """
    results = T.temperature_control_analysis(
        endogenous_activity(), env_for(cycling_temperature),
        sampling_interval=BIN, min_period_hours=18, max_period_hours=30,
    )
    summary = results["summary"]
    assert summary["n_confounded"] == 4
    assert summary["residual_test_conclusive"] is False

    results["temperature_rhythm"] = T.temperature_rhythmicity(
        cycling_temperature, BIN, 18, 30
    )
    text = T.generate_temperature_summary(results)
    assert "NOT CONCLUSIVE" in text
    assert "NOT explained by temperature" not in text
    assert "Free-run under constant temperature" in text


# -------------------------------------------------------------- robustness

def test_sampling_interval_is_taken_from_the_timestamps():
    """A wrong declared bin size must not rescale the reported periods.

    The core bin spinbox is capped at 300 s, so the caller's declared interval
    can disagree with the data's real spacing.
    """
    rng = np.random.default_rng(6)
    activity = series(CIRCADIAN + rng.normal(0, 0.05, N))
    temp = 20.0 + rng.normal(0, 0.05, N)
    result = T.analyze_roi_temperature_control(
        activity, TIMES.tolist(), temp.tolist(),
        sampling_interval=BIN / 4,          # deliberately wrong
        min_period_hours=18, max_period_hours=30,
    )
    assert 22 < result["activity_rhythm"]["dominant_period"] < 26


def test_missing_temperature_is_reported_per_roi():
    results = T.temperature_control_analysis(
        endogenous_activity(), {}, sampling_interval=BIN
    )
    assert all("error" in v for v in results["roi_results"].values())
    assert results["summary"]["n_valid"] == 0


def test_non_overlapping_temperature_is_rejected(flat_temperature):
    """A temperature record from a different time span must not be used."""
    far_times = (TIMES + 10 * 24 * 3600).tolist()
    result = T.analyze_roi_temperature_control(
        series(CIRCADIAN), far_times, flat_temperature.tolist(),
        sampling_interval=BIN,
    )
    assert "error" in result
    assert "overlap" in result["error"].lower()


# ------------------------------------------------------------------- pooled

def test_each_pooled_roi_uses_its_own_dataset_temperature(
    flat_temperature, cycling_temperature
):
    rng = np.random.default_rng(8)
    activity = {}
    for r in range(1, 3):
        activity[make_composite_id(1, r)] = series(
            CIRCADIAN + rng.normal(0, 0.1, N)
        )
        activity[make_composite_id(2, r)] = series(
            0.5 + 0.15 * (cycling_temperature - 20.0) + rng.normal(0, 0.02, N)
        )

    env = {
        1: {"times": TIMES.tolist(), "temperature": flat_temperature.tolist()},
        2: {"times": TIMES.tolist(), "temperature": cycling_temperature.tolist()},
    }
    results = T.temperature_control_analysis(
        activity, env, sampling_interval=BIN,
        min_period_hours=18, max_period_hours=30,
    )
    roi_results = results["roi_results"]

    # Dataset 1 sat in a flat incubator: almost no shared variance
    assert roi_results[1]["regression"]["r_squared"] < 0.05
    # Dataset 2's activity was constructed from its own temperature
    assert roi_results[make_composite_id(2, 1)]["regression"]["r_squared"] > 0.9


def test_summary_labels_pooled_rois(flat_temperature):
    activity = {
        make_composite_id(1, 1): series(CIRCADIAN),
        make_composite_id(2, 1): series(CIRCADIAN),
    }
    env = {
        1: {"times": TIMES.tolist(), "temperature": flat_temperature.tolist()},
        2: {"times": TIMES.tolist(), "temperature": flat_temperature.tolist()},
    }
    results = T.temperature_control_analysis(
        activity, env, sampling_interval=BIN,
        min_period_hours=18, max_period_hours=30,
    )
    text = T.generate_temperature_summary(results)
    assert "ROI 1" in text
    assert "ROI1_2" in text
