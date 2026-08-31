"""
_temperature.py - Temperature control analysis

Tests whether the detected activity rhythm could be an artefact of ambient
temperature variation, rather than an endogenous circadian rhythm.

A plain activity-vs-temperature correlation is not sufficient evidence on its
own: if the incubator temperature happens to drift on a 24 h cycle, activity and
temperature correlate strongly whether or not temperature drives the behaviour.

Which test carries the argument depends entirely on whether the temperature is
itself rhythmic at the period in question:

**Temperature arrhythmic** — the easy case. ``temperature_rhythmicity`` settles
it immediately: no 24 h component in the temperature means no possible 24 h
drive. ``regress_out_temperature`` plus a periodogram of the residuals then
confirms it directly.

**Temperature rhythmic at the same period** — the hard and, in practice, common
case. Activity and temperature are then *collinear* at that frequency, and no
regression can separate them: removing temperature also removes a genuine
endogenous rhythm. The residual test cannot answer the question and is flagged
``confounded`` rather than reported as a negative result. Two arguments still
work, and they are the ones to put in a figure:

``compare_to_q10_bound``
    A physiological ceiling, not a statistical test. Metabolic rate scales as
    ``Q10 ** (dT / 10)``, so a measured temperature swing sets a hard upper limit
    on the modulation it could produce. If the observed rhythm exceeds that limit
    several-fold, temperature cannot be the cause regardless of correlation.

``interindividual_spread``
    Individuals in one dish share the *identical* temperature trace — its
    between-individual variance is exactly zero. A common driver therefore
    predicts near-identical rhythms. Differing gain could scatter amplitudes, but
    a linear response cannot shift phase, so scattered acrophases argue for
    independent internal clocks.

A note on the correlation: both signals are periodic, which makes a naive
"strongest r over all lags" statistic close to tautological — see
``crosscorrelate_with_lag``, which reports the concurrent value instead.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Timeseries names used for ambient temperature across recording schema versions
TEMPERATURE_KEYS = (
    "temperature",
    "temperature_celsius",
    "temp",
    "temp_celsius",
    "ambient_temperature",
)

HUMIDITY_KEYS = ("humidity", "humidity_percent")


# ---------------------------------------------------------------------------
# Reading the temperature record
# ---------------------------------------------------------------------------

def extract_environment_from_file(
    file_path: str, frame_interval: float = None, log=None
) -> Optional[Dict[str, Any]]:
    """Read the ambient temperature timeseries from a raw recording.

    Temperature is stored once per frame under ``timeseries/`` in the raw HDF5,
    the same place the Telemetry tab reads it from. Timestamps are reconstructed
    from the frame interval, matching how the activity time base is built.

    Returns ``None`` when the file carries no recognised temperature dataset.
    """
    from ._io_abstraction import open_file_reader

    def _log(msg):
        if log is not None:
            log(msg)

    if not file_path or file_path.lower().endswith((".avi", ".mp4")):
        return None

    try:
        with open_file_reader(file_path) as reader:
            if "timeseries" not in reader.keys("/"):
                return None
            ts_keys = reader.keys("timeseries")

            temperature = None
            temp_source = None
            for name in TEMPERATURE_KEYS:
                if name in ts_keys:
                    temperature = reader.read_all(f"timeseries/{name}").astype(float)
                    temp_source = name
                    break
            if temperature is None:
                _log("  No temperature timeseries found in recording")
                return None

            humidity = None
            humidity_source = None
            for name in HUMIDITY_KEYS:
                if name in ts_keys:
                    humidity = reader.read_all(f"timeseries/{name}").astype(float)
                    humidity_source = name
                    break

            interval = frame_interval
            if not interval:
                root_attrs = reader.get_attrs("/")
                for attr in ("frame_interval", "interval"):
                    if attr in root_attrs:
                        interval = float(root_attrs[attr])
                        break
                if not interval and "fps" in root_attrs:
                    fps = float(root_attrs["fps"])
                    interval = 1.0 / fps if fps > 0 else None
            interval = float(interval or 1.0)

            times = np.arange(len(temperature), dtype=float) * interval

            _log(
                f"  ✓ Temperature: {len(temperature)} samples from "
                f"'{temp_source}' at {interval:g} s intervals"
            )

            env = {
                "times": times.tolist(),
                "temperature": temperature.tolist(),
                "temperature_source": temp_source,
                "units": "celsius",
                "frame_interval": interval,
            }
            if humidity is not None:
                env["humidity"] = humidity.tolist()
                env["humidity_source"] = humidity_source
            return env

    except Exception as exc:
        _log(f"  ⚠️ Could not read temperature data: {exc}")
        return None


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def resample_to(
    source_times: Sequence[float],
    source_values: Sequence[float],
    target_times: Sequence[float],
) -> np.ndarray:
    """Interpolate a temperature trace onto an ROI's own timestamps.

    Temperature is recorded once per frame while activity is binned, so the two
    must be put on a common time base before anything is correlated. Samples
    outside the temperature record become ``NaN`` rather than being clamped to
    the nearest edge value, which would invent a flat stretch.
    """
    src_t = np.asarray(source_times, dtype=float)
    src_v = np.asarray(source_values, dtype=float)
    tgt_t = np.asarray(target_times, dtype=float)

    if src_t.size == 0 or tgt_t.size == 0:
        return np.full(tgt_t.shape, np.nan)

    order = np.argsort(src_t)
    src_t, src_v = src_t[order], src_v[order]

    finite = np.isfinite(src_v)
    if finite.sum() < 2:
        return np.full(tgt_t.shape, np.nan)

    return np.interp(
        tgt_t, src_t[finite], src_v[finite], left=np.nan, right=np.nan
    )


def _finite_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


# ---------------------------------------------------------------------------
# 1. How much does temperature vary at all?
# ---------------------------------------------------------------------------

def temperature_statistics(values: Sequence[float]) -> Dict[str, Any]:
    """Basic descriptive statistics of the temperature trace.

    A small peak-to-trough range is itself strong evidence: a few tenths of a
    degree cannot plausibly drive a large behavioural rhythm.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"error": "No finite temperature samples"}

    p_low, p_high = np.percentile(v, [1, 99])
    return {
        "n_samples": int(v.size),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "range": float(np.max(v) - np.min(v)),
        # Robust range ignores single-sample sensor glitches
        "robust_range": float(p_high - p_low),
    }


# ---------------------------------------------------------------------------
# 2. Is the temperature trace itself rhythmic?
# ---------------------------------------------------------------------------

def temperature_rhythmicity(
    values: Sequence[float],
    sampling_interval: float,
    min_period_hours: float = 18.0,
    max_period_hours: float = 30.0,
    significance_level: float = 0.05,
    method: str = "chi2",
    n_permutations: int = 200,
) -> Dict[str, Any]:
    """Run the standard periodogram on the temperature trace.

    If temperature carries no significant component in the period band where
    the activity rhythm sits, the causal explanation fails immediately — this
    is the cleanest evidence available and costs one periodogram.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return {"error": "Temperature series too short for periodogram"}

    if method == "fft":
        from ._circadian_fft import fft_periodogram

        result = fft_periodogram(
            v,
            sampling_interval=sampling_interval,
            min_period_hours=min_period_hours,
            max_period_hours=max_period_hours,
            significance_level=significance_level,
            n_permutations=n_permutations,
        )
        result["method"] = "fft"
        result["is_significant"] = bool(result.get("is_significant", False))
    else:
        from ._fisher_analysis import fisher_z_periodogram

        result = fisher_z_periodogram(
            v,
            sampling_interval=sampling_interval,
            min_period_hours=min_period_hours,
            max_period_hours=max_period_hours,
            significance_level=significance_level,
        )
        result["method"] = "chi2"
        result["is_significant"] = bool(result.get("is_significant", False))

    return result


# ---------------------------------------------------------------------------
# 3. Cross-correlation with lag
# ---------------------------------------------------------------------------

def crosscorrelate_with_lag(
    activity: Sequence[float],
    temperature: Sequence[float],
    sampling_interval: float,
    max_lag_hours: float = 6.0,
    curve_span_hours: float = 24.0,
) -> Dict[str, Any]:
    """Pearson r between activity and temperature, honestly reported.

    Both signals are periodic, which makes a naive "strongest correlation over
    all lags" statistic close to tautological: any two 24 h-periodic signals
    correlate strongly at *some* lag, and near half a period that correlation is
    large and negative. Ranking by |r| over a full cycle therefore manufactures a
    dramatic number that carries no information about causation.

    This function reports instead:

    ``zero_lag_r``
        The concurrent correlation, with its sign. This is the headline number.
    ``best_r`` / ``best_lag_hours``
        The strongest correlation within a *physiologically plausible* window —
        by default 0 to +6 h with temperature leading, the only direction in
        which temperature could drive behaviour.
    ``lags_hours`` / ``correlations``
        The full r(lag) curve over ``curve_span_hours``, for display only. Seeing
        it oscillate is what stops a reader mistaking the periodicity for a
        result.
    """
    a = np.asarray(activity, dtype=float)
    t = np.asarray(temperature, dtype=float)
    n = min(a.size, t.size)
    a, t = a[:n], t[:n]

    a, t = _finite_pair(a, t)
    if a.size < 10:
        return {"error": "Too few overlapping samples for cross-correlation"}

    def _r_at(k: int) -> float:
        if k > 0:                       # temperature leads activity
            x, y = t[:-k], a[k:]
        elif k < 0:                     # activity leads temperature
            x, y = t[-k:], a[:k]
        else:
            x, y = t, a
        if x.size < 10 or np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    limit = (a.size - 1) // 2

    # Full curve, symmetric, for display
    span = int(curve_span_hours * 3600.0 / sampling_interval)
    span = max(0, min(span, limit))
    curve_lags = np.arange(-span, span + 1)
    curve = np.array([_r_at(int(k)) for k in curve_lags])

    zero_lag_r = _r_at(0)
    if not np.isfinite(zero_lag_r):
        return {"error": "Correlation undefined (constant signal)"}

    # Plausible window: temperature leading, 0 .. max_lag_hours
    win = int(max_lag_hours * 3600.0 / sampling_interval)
    win = max(0, min(win, limit))
    win_lags = np.arange(0, win + 1)
    win_r = np.array([_r_at(int(k)) for k in win_lags])

    if np.any(np.isfinite(win_r)):
        best_idx = int(np.nanargmax(np.abs(win_r)))
        best_r = float(win_r[best_idx])
        best_lag_samples = int(win_lags[best_idx])
    else:
        best_r, best_lag_samples = zero_lag_r, 0

    n_eff = a.size - abs(best_lag_samples)

    from ._circadian_similarity import correlation_significance_test

    sig_zero = correlation_significance_test(zero_lag_r, a.size)
    sig_best = correlation_significance_test(best_r, n_eff)

    return {
        # headline
        "zero_lag_r": zero_lag_r,
        "zero_lag_r_squared": float(zero_lag_r**2),
        "zero_lag_p_value": sig_zero["p_value"],
        "zero_lag_is_significant": bool(sig_zero["is_significant"]),
        # plausible window
        "best_r": best_r,
        "best_lag_hours": best_lag_samples * sampling_interval / 3600.0,
        "r_squared": float(best_r**2),
        "p_value": sig_best["p_value"],
        "is_significant": bool(sig_best["is_significant"]),
        "search_window_hours": (0.0, float(max_lag_hours)),
        # display only
        "lags_hours": curve_lags * sampling_interval / 3600.0,
        "correlations": curve,
        "n_samples": int(n_eff),
    }


# ---------------------------------------------------------------------------
# 4. Coherence at the target period
# ---------------------------------------------------------------------------

def coherence_at_period(
    activity: Sequence[float],
    temperature: Sequence[float],
    sampling_interval: float,
    target_period_hours: float = 24.0,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Magnitude-squared coherence between activity and temperature.

    Catches a consistent phase relationship at the circadian frequency that a
    single broadband r can miss.
    """
    from ._circadian_coherence import calculate_coherence

    a = np.asarray(activity, dtype=float)
    t = np.asarray(temperature, dtype=float)
    n = min(a.size, t.size)
    a, t = _finite_pair(a[:n], t[:n])
    if a.size < 10:
        return {"error": "Too few overlapping samples for coherence"}

    return calculate_coherence(
        a,
        t,
        sampling_interval=sampling_interval,
        target_period_hours=target_period_hours,
        significance_level=significance_level,
    )


# ---------------------------------------------------------------------------
# 5. Regress temperature out, then re-test the rhythm
# ---------------------------------------------------------------------------

def regress_out_temperature(
    activity: Sequence[float],
    temperature: Sequence[float],
    lag_samples: int = 0,
) -> Dict[str, Any]:
    """Remove all linear temperature-explained variance from an activity trace.

    Fits ``activity = a + b * temperature(t - lag)`` by ordinary least squares
    and returns the residual. Whatever rhythm survives in that residual cannot
    be attributed to a linear temperature effect at that lag.

    Returns a dict with ``residual`` (same length as the overlapping samples),
    the fit coefficients, and ``r_squared`` — the fraction of activity variance
    temperature accounts for.
    """
    a = np.asarray(activity, dtype=float)
    t = np.asarray(temperature, dtype=float)
    n = min(a.size, t.size)
    a, t = a[:n], t[:n]

    if lag_samples > 0:
        t, a = t[:-lag_samples], a[lag_samples:]
    elif lag_samples < 0:
        t, a = t[-lag_samples:], a[:lag_samples]

    a, t = _finite_pair(a, t)
    if a.size < 10:
        return {"error": "Too few overlapping samples for regression"}
    if np.std(t) == 0:
        # Constant temperature explains nothing; the residual is the signal.
        return {
            "residual": a - np.mean(a),
            "slope": 0.0,
            "intercept": float(np.mean(a)),
            "r_squared": 0.0,
            "n_samples": int(a.size),
            "note": "Temperature is constant — nothing to regress out",
        }

    slope, intercept = np.polyfit(t, a, 1)
    fitted = slope * t + intercept
    residual = a - fitted

    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    ss_res = float(np.sum(residual**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "residual": residual,
        "fitted": fitted,
        "activity_aligned": a,
        "temperature_aligned": t,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "n_samples": int(a.size),
    }


# ---------------------------------------------------------------------------
# 6. Q10 bound — a physiological ceiling, immune to collinearity
# ---------------------------------------------------------------------------

def q10_amplitude_bound(
    temperature_peak_to_trough: float, q10: float = 3.0
) -> Dict[str, Any]:
    """Largest activity modulation the measured temperature swing could cause.

    Metabolic rate scales as ``Q10 ** (dT / 10)``. For a temperature cycle of
    ``dT`` peak-to-trough, that sets a hard ceiling on the peak-to-trough
    modulation of any metabolically driven behaviour — no matter how the
    statistics come out.

    This is the one argument that survives the collinearity that defeats the
    residual test: it is a physiological limit, not a statistical inference.
    """
    dt = float(temperature_peak_to_trough)
    if dt < 0 or not np.isfinite(dt):
        return {"error": "Invalid temperature range"}
    ratio = float(q10) ** (dt / 10.0)
    return {
        "q10": float(q10),
        "temperature_peak_to_trough": dt,
        "rate_ratio": ratio,
        # peak-to-trough modulation as a fraction of the mean
        "max_modulation": ratio - 1.0,
    }


def compare_to_q10_bound(
    observed_relative_amplitude: float,
    temperature_peak_to_trough: float,
    q10_values: Sequence[float] = (2.0, 2.5, 3.0),
) -> Dict[str, Any]:
    """Compare the observed rhythm amplitude against the Q10 ceiling.

    Args:
        observed_relative_amplitude: Cosinor amplitude / MESOR (not
            peak-to-trough — this function doubles it).
        temperature_peak_to_trough: From the temperature record, in °C.
        q10_values: Ceilings to report. 2–3 covers most ectotherm metabolism.

    Returns a dict whose ``exceedance`` says how many times larger the observed
    rhythm is than the most generous temperature explanation.
    """
    observed_ptp = 2.0 * float(observed_relative_amplitude)
    bounds = {}
    for q10 in q10_values:
        bound = q10_amplitude_bound(temperature_peak_to_trough, q10)
        if "error" in bound:
            continue
        bound["observed_peak_to_trough"] = observed_ptp
        bound["exceedance"] = (
            observed_ptp / bound["max_modulation"]
            if bound["max_modulation"] > 0
            else float("inf")
        )
        bounds[float(q10)] = bound

    if not bounds:
        return {"error": "Could not compute Q10 bound"}

    # The most generous ceiling is the highest Q10 — quote that one
    most_generous = bounds[max(bounds)]
    return {
        "bounds": bounds,
        "observed_peak_to_trough": observed_ptp,
        "temperature_peak_to_trough": float(temperature_peak_to_trough),
        "min_exceedance": float(most_generous["exceedance"]),
        "temperature_sufficient": bool(most_generous["exceedance"] <= 1.0),
    }


# ---------------------------------------------------------------------------
# 7. Inter-individual spread — a test at the level of rhythm parameters
# ---------------------------------------------------------------------------

def interindividual_spread(
    amplitudes: Sequence[float],
    mesors: Sequence[float],
    peak_times: Sequence[float],
    period_hours: float = 24.0,
    roi_ids: Sequence[int] = None,
) -> Dict[str, Any]:
    """Spread of rhythm parameters across individuals sharing one environment.

    Every ROI in a dish sees the *identical* temperature trace — its variance
    between individuals is exactly zero. A common deterministic driver therefore
    predicts near-identical rhythms. Amplitude may still differ if individuals
    respond with different gain, but a linear response cannot shift the *phase*:
    scattered acrophases require independent internal timing.

    This tests the claim at the level of rhythm parameters rather than of the
    raw time series, which is where collinearity does not bite.
    """
    amp = np.asarray(amplitudes, dtype=float)
    mes = np.asarray(mesors, dtype=float)
    ph = np.asarray(peak_times, dtype=float)
    ids = np.asarray(
        roi_ids if roi_ids is not None else np.arange(amp.size), dtype=int
    )
    ok = np.isfinite(amp) & np.isfinite(mes) & np.isfinite(ph) & (mes != 0)
    # Carry the ids through the same mask, so callers can never mis-pair a
    # point with an ROI colour or label when an individual drops out.
    amp, mes, ph, ids = amp[ok], mes[ok], ph[ok], ids[: ok.size][ok]

    if amp.size < 2:
        return {"error": "Need at least 2 individuals for a spread estimate"}

    rel = amp / mes
    angles = ph * 2.0 * np.pi / float(period_hours)
    resultant = np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    mean_phase = (mean_angle % (2 * np.pi)) * period_hours / (2 * np.pi)
    # Circular SD, converted from radians to hours
    circular_sd = (
        float(np.sqrt(-2.0 * np.log(resultant)) * period_hours / (2 * np.pi))
        if 0 < resultant < 1
        else 0.0
    )

    return {
        "n": int(amp.size),
        "roi_ids": ids.tolist(),
        "relative_amplitudes": rel.tolist(),
        "amplitude_min": float(rel.min()),
        "amplitude_max": float(rel.max()),
        "amplitude_ratio": float(rel.max() / rel.min()) if rel.min() > 0 else float("inf"),
        "amplitude_cv": float(amp.std(ddof=1) / amp.mean()) if amp.mean() else float("nan"),
        "peak_times": ph.tolist(),
        "phase_range_hours": float(ph.max() - ph.min()),
        "mean_phase_hours": float(mean_phase),
        "resultant_length": float(resultant),
        "phase_circular_sd_hours": circular_sd,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze_roi_temperature_control(
    activity_series: Sequence[Tuple[float, float]],
    temp_times: Sequence[float],
    temp_values: Sequence[float],
    sampling_interval: float,
    min_period_hours: float = 18.0,
    max_period_hours: float = 30.0,
    target_period_hours: float = 24.0,
    max_lag_hours: float = 12.0,
    significance_level: float = 0.05,
    method: str = "chi2",
    n_permutations: int = 200,
) -> Dict[str, Any]:
    """Full temperature control analysis for one ROI.

    Runs the rhythm test on the raw activity, on the temperature, and on the
    temperature-regressed residual, so the three can be compared directly.
    """
    if not activity_series or len(activity_series) < 10:
        return {"error": "Activity series too short"}

    times = np.array([t for t, _ in activity_series], dtype=float)
    activity = np.array([v for _, v in activity_series], dtype=float)
    temperature = resample_to(temp_times, temp_values, times)

    # Derive the sampling interval from the timestamps rather than trusting the
    # caller's declared bin size. The two can disagree — the core bin spinbox is
    # capped at 300 s while re-binning happens at the analysis bin size — and a
    # wrong interval rescales every period the periodograms report.
    if times.size > 1:
        observed = float(np.median(np.diff(times)))
        if observed > 0:
            sampling_interval = observed

    n_overlap = int(np.sum(np.isfinite(temperature) & np.isfinite(activity)))
    if n_overlap < 10:
        return {
            "error": "Temperature record does not overlap this ROI's recording",
            "n_overlap": n_overlap,
        }

    def _periodogram(values):
        return temperature_rhythmicity(
            values,
            sampling_interval=sampling_interval,
            min_period_hours=min_period_hours,
            max_period_hours=max_period_hours,
            significance_level=significance_level,
            method=method,
            n_permutations=n_permutations,
        )

    activity_rhythm = _periodogram(activity)
    # Periodogram of the temperature *as the activity analysis sees it* — same
    # grid, same sampling interval — so the two are directly comparable when
    # testing for collinearity below.
    temp_rhythm = _periodogram(temperature[np.isfinite(temperature)])
    crosscorr = crosscorrelate_with_lag(
        activity, temperature, sampling_interval, max_lag_hours=max_lag_hours,
        curve_span_hours=target_period_hours,
    )

    # Cosinor at the target period supplies the amplitude and acrophase that
    # the Q10 bound and the inter-individual spread are built on.
    from ._cosinor_analysis import single_cosinor_analysis

    cosinor = single_cosinor_analysis(
        activity, period_hours=target_period_hours,
        sampling_interval=sampling_interval,
    )
    coherence = coherence_at_period(
        activity,
        temperature,
        sampling_interval,
        target_period_hours=target_period_hours,
        significance_level=significance_level,
    )

    lag_samples = 0
    if "error" not in crosscorr:
        lag_samples = int(
            round(crosscorr["best_lag_hours"] * 3600.0 / sampling_interval)
        )
    regression = regress_out_temperature(activity, temperature, lag_samples)

    residual_rhythm = (
        _periodogram(regression["residual"])
        if "error" not in regression
        else {"error": regression["error"]}
    )

    return {
        "activity_rhythm": activity_rhythm,
        "temperature_rhythm": temp_rhythm,
        "crosscorrelation": crosscorr,
        "coherence": coherence,
        "regression": regression,
        "residual_rhythm": residual_rhythm,
        "cosinor": cosinor,
        "temperature_stats": temperature_statistics(temperature),
        "n_overlap": n_overlap,
        "verdict": _roi_verdict(
            activity_rhythm, residual_rhythm, regression, temp_rhythm
        ),
    }


def _period_of(rhythm: Dict[str, Any]) -> Optional[float]:
    if not rhythm or "error" in rhythm:
        return None
    value = rhythm.get("dominant_period")
    return float(value) if value else None


def _roi_verdict(
    activity_rhythm: Dict[str, Any],
    residual_rhythm: Dict[str, Any],
    regression: Dict[str, Any],
    temperature_rhythm: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Does the rhythm survive removal of the temperature-explained variance?

    Carries an important caveat. If the temperature trace is *itself* rhythmic
    at essentially the same period as the activity, the two are collinear at
    that frequency and no regression can separate them: removing temperature
    also removes the endogenous rhythm, so a "does not survive" result is not
    evidence of a temperature cause. That case is flagged as ``confounded`` and
    must not be read as a negative result — see ``confound_note``.
    """
    before = _period_of(activity_rhythm)
    after = _period_of(residual_rhythm)
    sig_before = bool(activity_rhythm.get("is_significant")) if activity_rhythm else False
    sig_after = bool(residual_rhythm.get("is_significant")) if residual_rhythm else False
    var_explained = float(regression.get("r_squared", 0.0)) if regression else 0.0

    period_shift = (
        abs(after - before) if (before is not None and after is not None) else None
    )
    survives = bool(
        sig_after and period_shift is not None and period_shift <= max(1.0, 0.1 * before)
    )

    # Collinearity check: is the temperature rhythmic at the same period?
    temp_period = _period_of(temperature_rhythm)
    temp_is_rhythmic = (
        bool(temperature_rhythm.get("is_significant"))
        if temperature_rhythm
        else False
    )
    confounded = False
    if temp_is_rhythmic and temp_period is not None and before is not None:
        # Within 10 % of each other counts as the same frequency band
        confounded = abs(temp_period - before) <= max(1.0, 0.1 * before)

    return {
        "period_before": before,
        "period_after": after,
        "period_shift_hours": period_shift,
        "significant_before": sig_before,
        "significant_after": sig_after,
        "variance_explained_by_temperature": var_explained,
        "rhythm_survives": survives,
        "temperature_period": temp_period,
        "temperature_is_rhythmic": temp_is_rhythmic,
        "confounded": confounded,
        "confound_note": (
            "Temperature is itself rhythmic at the same period — regression "
            "cannot separate the two, so the residual test is not conclusive."
            if confounded
            else ""
        ),
    }


def temperature_control_analysis(
    activity_data: Dict[int, Sequence[Tuple[float, float]]],
    temperature_by_dataset: Dict[int, Dict[str, Sequence[float]]],
    sampling_interval: float,
    **kwargs,
) -> Dict[str, Any]:
    """Run the temperature control analysis across all (pooled) ROIs.

    Args:
        activity_data: ``{roi_id: [(time_seconds, value), ...]}``. May use pooled
            composite ROI ids from :mod:`._batch`.
        temperature_by_dataset: ``{dataset_idx: {"times": [...], "temperature": [...]}}``
            already shifted onto the same time base as the activity data. Each
            ROI is tested against its own dataset's temperature record.
        sampling_interval: Effective seconds between activity samples.

    Returns:
        ``{"roi_results": {...}, "summary": {...}}``
    """
    from ._batch import dataset_index

    # Summary-only options must not leak into the per-ROI analysis
    q10_values = kwargs.pop("q10_values", (2.0, 2.5, 3.0))
    period_hours = kwargs.get("target_period_hours", 24.0)

    roi_results: Dict[int, Dict[str, Any]] = {}

    for roi_id, series in activity_data.items():
        ds_idx = dataset_index(roi_id)
        env = temperature_by_dataset.get(ds_idx)
        if not env or not env.get("temperature"):
            roi_results[roi_id] = {
                "error": f"No temperature record for dataset {ds_idx}"
            }
            continue

        roi_results[roi_id] = analyze_roi_temperature_control(
            series,
            env.get("times", []),
            env.get("temperature", []),
            sampling_interval=sampling_interval,
            **kwargs,
        )

    return {
        "roi_results": roi_results,
        "summary": summarize_temperature_control(
            roi_results,
            temperature_by_dataset,
            period_hours=period_hours,
            q10_values=q10_values,
        ),
    }


def summarize_temperature_control(
    roi_results: Dict[int, Dict[str, Any]],
    temperature_by_dataset: Dict[int, Dict[str, Sequence[float]]],
    period_hours: float = 24.0,
    q10_values: Sequence[float] = (2.0, 2.5, 3.0),
) -> Dict[str, Any]:
    """Aggregate the per-ROI outcomes into population-level numbers."""
    valid = {k: v for k, v in roi_results.items() if "error" not in v}

    r_values = [
        v["crosscorrelation"]["best_r"]
        for v in valid.values()
        if "error" not in v.get("crosscorrelation", {"error": 1})
    ]
    zero_lag_values = [
        v["crosscorrelation"]["zero_lag_r"]
        for v in valid.values()
        if "error" not in v.get("crosscorrelation", {"error": 1})
    ]
    # Upper end of the plausible response window actually used, for reporting
    search_window = 6.0
    for v in valid.values():
        cc = v.get("crosscorrelation", {})
        if "search_window_hours" in cc:
            search_window = float(cc["search_window_hours"][1])
            break
    var_explained = [
        v["regression"]["r_squared"]
        for v in valid.values()
        if "error" not in v.get("regression", {"error": 1})
    ]
    survivors = [v for v in valid.values() if v["verdict"]["rhythm_survives"]]
    rhythmic_before = [
        v for v in valid.values() if v["verdict"]["significant_before"]
    ]
    confounded = [v for v in valid.values() if v["verdict"]["confounded"]]

    # Rhythmicity of the temperature traces themselves, one per dataset
    temp_rhythms = {}
    for ds_idx, env in temperature_by_dataset.items():
        values = env.get("temperature")
        if values:
            temp_rhythms[ds_idx] = {
                "stats": temperature_statistics(values),
            }

    # --- Rhythm parameters across individuals -----------------------------
    amps, mesors, phases, spread_ids = [], [], [], []
    for roi_id in sorted(valid):
        cos = valid[roi_id].get("cosinor") or {}
        amp, mes, peak = (
            cos.get("amplitude"),
            cos.get("mesor"),
            cos.get("peak_time"),
        )
        if amp is not None and mes and np.isfinite(amp) and np.isfinite(mes):
            amps.append(float(amp))
            mesors.append(float(mes))
            phases.append(float(peak) if peak is not None else np.nan)
            spread_ids.append(int(roi_id))
    spread = (
        interindividual_spread(
            amps, mesors, phases, period_hours=period_hours, roi_ids=spread_ids
        )
        if len(amps) >= 2
        else {"error": "Need at least 2 ROIs"}
    )

    # --- Q10 ceiling ------------------------------------------------------
    # Uses the 24 h component of the temperature, not the raw min/max, so a
    # one-off sensor glitch or the initial equilibration cannot inflate it.
    q10_result = {"error": "No temperature amplitude available"}
    if temp_rhythms and "error" not in spread:
        first_env = temperature_by_dataset[min(temperature_by_dataset)]
        values = np.asarray(first_env.get("temperature", []), dtype=float)
        times = np.asarray(first_env.get("times", []), dtype=float)
        if values.size > 10 and times.size == values.size:
            from ._cosinor_analysis import single_cosinor_analysis

            interval = float(np.median(np.diff(times))) if times.size > 1 else 60.0
            temp_cos = single_cosinor_analysis(
                values, period_hours=period_hours, sampling_interval=interval
            )
            temp_amp = temp_cos.get("amplitude")
            if temp_amp is not None and np.isfinite(temp_amp):
                mean_rel = float(np.mean(spread["relative_amplitudes"]))
                q10_result = compare_to_q10_bound(
                    mean_rel, 2.0 * float(temp_amp), q10_values=q10_values
                )
                q10_result["temperature_amplitude"] = float(temp_amp)

    return {
        "interindividual_spread": spread,
        "q10": q10_result,
        "max_lag_hours": search_window,
        "mean_zero_lag_r": (
            float(np.mean(zero_lag_values)) if zero_lag_values else None
        ),
        "max_abs_zero_lag_r": (
            float(np.max(np.abs(zero_lag_values))) if zero_lag_values else None
        ),
        "n_rois": len(roi_results),
        "n_valid": len(valid),
        "n_rhythmic_before": len(rhythmic_before),
        "n_rhythm_survives": len(survivors),
        "n_confounded": len(confounded),
        # When temperature shares the activity's period the residual test cannot
        # discriminate, and reporting it as a negative result would be wrong.
        "residual_test_conclusive": len(confounded) == 0,
        "mean_abs_r": float(np.mean(np.abs(r_values))) if r_values else None,
        "max_abs_r": float(np.max(np.abs(r_values))) if r_values else None,
        "mean_variance_explained": (
            float(np.mean(var_explained)) if var_explained else None
        ),
        "max_variance_explained": (
            float(np.max(var_explained)) if var_explained else None
        ),
        "temperature_by_dataset": temp_rhythms,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_temperature_summary(results: Dict[str, Any]) -> str:
    """Human-readable report of the temperature control analysis."""
    from ._batch import roi_label

    roi_results = results.get("roi_results", {})
    summary = results.get("summary", {})
    temp_rhythm = results.get("temperature_rhythm", {})

    lines = [
        "=" * 70,
        "TEMPERATURE CONTROL - Is the rhythm driven by temperature?",
        "=" * 70,
        "",
        "Tests whether ambient temperature variation could account for the",
        "activity rhythm. The decisive test is the residual periodogram: the",
        "temperature-explained variance is regressed out of each ROI and the",
        "rhythm is re-tested on what remains.",
        "",
    ]

    # --- 1. Temperature magnitude -----------------------------------------
    lines.append("─── 1. Temperature variation ──────────────────────────")
    per_dataset = summary.get("temperature_by_dataset", {})
    if not per_dataset:
        lines.append("  No temperature record available.")
    for ds_idx in sorted(per_dataset):
        stats = per_dataset[ds_idx].get("stats", {})
        if "error" in stats:
            lines.append(f"  Dataset {ds_idx}: {stats['error']}")
            continue
        lines.append(
            f"  Dataset {ds_idx}: mean {stats['mean']:.2f} °C, "
            f"SD {stats['sd']:.3f} °C, "
            f"range {stats['range']:.2f} °C "
            f"(1–99 %: {stats['robust_range']:.2f} °C)"
        )
    lines.append("")

    # --- 2. Is the temperature itself rhythmic? ---------------------------
    lines.append("─── 2. Is the temperature trace itself rhythmic? ──────")
    if temp_rhythm and "error" not in temp_rhythm:
        period = temp_rhythm.get("dominant_period")
        significant = temp_rhythm.get("is_significant")
        period_str = f"{period:.1f} h" if period else "n/a"
        lines.append(
            f"  Dominant period: {period_str} — "
            f"{'SIGNIFICANT' if significant else 'not significant'}"
        )
        if not significant:
            lines.append(
                "  → Temperature carries no significant rhythm in this band, so it"
            )
            lines.append(
                "    cannot be the source of a rhythm in the activity data."
            )
    elif temp_rhythm:
        lines.append(f"  {temp_rhythm.get('error', 'not available')}")
    lines.append("")

    # --- 3. Correlation ----------------------------------------------------
    lines.append("─── 3. Activity vs temperature correlation ────────────")
    if summary.get("mean_zero_lag_r") is not None:
        mean_zero = summary["mean_zero_lag_r"]
        lines.append(
            f"  Concurrent correlation (lag 0): r = {mean_zero:+.3f} on average, "
            f"R² = {100 * mean_zero**2:.1f} %"
        )
        lines.append(
            f"  Strongest within the plausible 0–{summary.get('max_lag_hours', 6):.0f} h "
            f"window: |r| = {summary.get('mean_abs_r', float('nan')):.3f} on average"
        )
        lines.append(
            "  Note: the r(lag) curve oscillates because both signals are"
        )
        lines.append(
            "  periodic. A large |r| exists at SOME lag by construction, so only"
        )
        lines.append(
            "  the concurrent value and the plausible window are interpretable."
        )
    else:
        lines.append("  No valid correlations computed.")
    lines.append("")

    # --- 3b. Q10 ceiling ---------------------------------------------------
    lines.append("─── 3b. Q10 ceiling (works despite collinearity) ──────")
    q10 = summary.get("q10", {})
    if q10 and "error" not in q10:
        lines.append(
            f"  Temperature 24 h component: {q10['temperature_peak_to_trough']:.3f} °C "
            f"peak-to-trough"
        )
        for q in sorted(q10["bounds"]):
            bound = q10["bounds"][q]
            lines.append(
                f"    Q10 = {q:g}  →  at most "
                f"{100 * bound['max_modulation']:.2f} % modulation"
            )
        lines.append(
            f"  Observed rhythm: "
            f"{100 * q10['observed_peak_to_trough']:.1f} % peak-to-trough"
        )
        if q10["temperature_sufficient"]:
            lines.append(
                "  ⚠ The temperature swing is large enough to account for the rhythm."
            )
        else:
            lines.append(
                f"  → The rhythm is {q10['min_exceedance']:.0f}× larger than even the"
            )
            lines.append(
                "    most generous temperature explanation allows. This is a"
            )
            lines.append(
                "    physiological limit, not a statistical test, so it holds even"
            )
            lines.append(
                "    when temperature shares the activity's period."
            )
    else:
        lines.append("  Not available.")
    lines.append("")

    # --- 3c. Rhythm parameters across individuals -------------------------
    lines.append("─── 3c. Do individuals differ? ────────────────────────")
    spread = summary.get("interindividual_spread", {})
    if spread and "error" not in spread:
        lines.append(
            f"  n = {spread['n']} individuals, all exposed to the IDENTICAL"
        )
        lines.append("  temperature trace (between-individual variance = 0).")
        lines.append(
            f"  Relative amplitude: {100 * spread['amplitude_min']:.1f} % – "
            f"{100 * spread['amplitude_max']:.1f} %  "
            f"(ratio {spread['amplitude_ratio']:.2f}×, CV {100 * spread['amplitude_cv']:.1f} %)"
        )
        lines.append(
            f"  Acrophase spread: {spread['phase_range_hours']:.1f} h "
            f"(circular SD {spread['phase_circular_sd_hours']:.2f} h, "
            f"mean ZT {spread['mean_phase_hours']:.1f} h)"
        )
        lines.append(
            "  → A shared driver predicts near-identical rhythms. Differing gain"
        )
        lines.append(
            "    could scatter amplitudes, but a linear response cannot shift"
        )
        lines.append(
            "    phase — scattered acrophases argue for independent clocks."
        )
        if spread["n"] < 5:
            lines.append(
                f"  ⚠ Only {spread['n']} individuals — suggestive, not decisive."
            )
    else:
        lines.append("  Needs at least 2 ROIs.")
    lines.append("")

    # --- 4. The decisive test ---------------------------------------------
    lines.append("─── 4. Does the rhythm survive? (decisive) ────────────")
    n_before = summary.get("n_rhythmic_before", 0)
    n_after = summary.get("n_rhythm_survives", 0)
    n_confounded = summary.get("n_confounded", 0)
    conclusive = summary.get("residual_test_conclusive", True)

    lines.append(f"  Rhythmic ROIs before regression : {n_before}")
    lines.append(f"  Still rhythmic after removing temperature: {n_after}")

    if not conclusive:
        # Temperature and activity are collinear at this period. Removing one
        # removes the other, so a low survival count says nothing about cause.
        lines.append("")
        lines.append(f"  ⚠ NOT CONCLUSIVE for {n_confounded} ROI(s).")
        lines.append(
            "    The temperature is itself rhythmic at the same period as the"
        )
        lines.append(
            "    activity. The two are collinear at that frequency, so regression"
        )
        lines.append(
            "    cannot separate them — removing temperature also removes an"
        )
        lines.append(
            "    endogenous rhythm. A low survival count here is NOT evidence"
        )
        lines.append("    that temperature drives the behaviour.")
        lines.append("")
        lines.append("    What does discriminate, in order of strength:")
        lines.append(
            "      1. Amplitude argument — see section 1. A temperature swing of"
        )
        lines.append(
            "         a few tenths of a degree cannot produce a large behavioural"
        )
        lines.append(
            "         rhythm (Q10 of 2–3 gives only a few percent metabolic change)."
        )
        lines.append(
            "      2. Free-run under constant temperature — the rhythm persisting"
        )
        lines.append("         with tau != 24 h is decisive."
        )
        lines.append(
            "      3. Phase relationship — a rhythm peaking at a phase the"
        )
        lines.append(
            "         temperature cycle cannot explain, or leading it (section 3)."
        )
        lines.append(
            "      4. Fix the rig so temperature no longer cycles, and re-record."
        )
    elif n_before:
        pct = 100.0 * n_after / n_before
        lines.append(f"  → {pct:.0f} % of rhythmic ROIs retain their rhythm")
        if pct >= 90:
            lines.append(
                "  → The rhythm is NOT explained by temperature variation."
            )
        elif pct >= 50:
            lines.append(
                "  → Mostly retained, but check the ROIs that lost significance."
            )
        else:
            lines.append(
                "  ⚠ Most rhythms disappear once temperature is removed — "
                "temperature may be a driver here."
            )
    lines.append("")

    # --- Per-ROI table -----------------------------------------------------
    lines.append("─── Per-ROI detail ────────────────────────────────────")
    lines.append(
        f"  {'ROI':<10} {'r':>7} {'lag(h)':>7} {'R²(%)':>7} "
        f"{'τ before':>9} {'τ after':>9}  survives"
    )
    for roi_id in sorted(roi_results):
        result = roi_results[roi_id]
        label = roi_label(roi_id)
        if "error" in result:
            lines.append(f"  {label:<10} {result['error']}")
            continue
        cc = result.get("crosscorrelation", {})
        verdict = result["verdict"]
        r_str = f"{cc['best_r']:+.3f}" if "error" not in cc else "  n/a"
        lag_str = f"{cc['best_lag_hours']:+.1f}" if "error" not in cc else " n/a"
        var_str = f"{100 * verdict['variance_explained_by_temperature']:.1f}"
        before = verdict["period_before"]
        after = verdict["period_after"]
        before_str = f"{before:.1f}" if before else "  n/a"
        after_str = f"{after:.1f}" if after else "  n/a"
        if verdict["confounded"]:
            mark = "n/c"          # not conclusive — collinear with temperature
        else:
            mark = "yes" if verdict["rhythm_survives"] else "NO"
        lines.append(
            f"  {label:<10} {r_str:>7} {lag_str:>7} {var_str:>7} "
            f"{before_str:>9} {after_str:>9}  {mark}"
        )

    lines.extend([
        "",
        "survives: yes = rhythm persists after temperature is removed",
        "          NO  = rhythm lost",
        "          n/c = not conclusive, temperature shares the same period",
        "",
        "Note: a positive lag means temperature leads activity — the direction",
        "expected if temperature were driving behaviour. A best correlation at a",
        "negative lag is evidence against a temperature cause.",
        "=" * 70,
    ])
    return "\n".join(lines)
