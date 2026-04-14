"""
_cosinor_analysis.py - Cosinor analysis for circadian rhythm quantification

This module implements single and population cosinor analysis, widely used
in chronobiology to quantify circadian rhythms. Cosinor analysis fits a
cosine curve to time series data and extracts key rhythmic parameters:

- MESOR (Midline Estimating Statistic of Rhythm): Mean activity level
- Amplitude: Half the difference between peak and trough
- Phase angle (φ): Phase offset of the fitted cosine (radians).
  NOTE: this is NOT the biological acrophase, which requires knowledge of
  the ZT reference and a confirmed circadian/ultradian signal. The "peak_time"
  output (hours from recording start to the first fitted peak) is also only
  a proxy for the biological acrophase.
- Period: Duration of one complete cycle

Reference:
Cornelissen, G. (2014). Cosinor-based rhythmometry.
Theoretical Biology and Medical Modelling, 11(1), 16.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats


def single_cosinor_analysis(
    time_series: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    period_hours: float = 24.0,
    sampling_interval: float = 60.0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform single-component cosinor analysis on time series data.

    Fits a cosine function: y(t) = MESOR + Amplitude * cos(2π*t/τ + φ)
    where τ is the period being tested and φ is the phase angle.

    Args:
        time_series: 1D array of activity values
        timestamps: Optional array of timestamps in seconds. If None, uses
                   regular intervals based on sampling_interval
        period_hours: Period to test (default: 24h for circadian rhythm)
        sampling_interval: Time between samples in seconds (default: 60s)
        alpha: Significance level for statistical tests (default: 0.05)

    Returns:
        Dictionary containing:
        - mesor: Midline Estimating Statistic of Rhythm (mean level)
        - amplitude: Amplitude of the rhythm
        - phase_angle_rad: Phase offset φ of the fitted cosine (radians, range −π to π)
        - peak_time: Time from recording start to first fitted cosine peak within
                     one period (hours). Only interpretable as biological acrophase
                     if ZT0 is known and the signal is a confirmed rhythm.
        - period: Tested period (hours)
        - p_value: Statistical significance of rhythm
        - r_squared: Goodness of fit
        - significant: Boolean indicating if rhythm is significant
        - fitted_curve: Fitted cosine curve values
        - ci_mesor: 95% confidence interval for MESOR
        - ci_amplitude: 95% confidence interval for amplitude
        - ci_peak_time: 95% confidence interval for peak_time (hours)
    """
    # Input validation
    if len(time_series) < 10:
        return {
            "mesor": np.nan,
            "amplitude": np.nan,
            "phase_angle_rad": np.nan,
            "peak_time": np.nan,
            "period": period_hours,
            "p_value": 1.0,
            "r_squared": 0.0,
            "significant": False,
            "fitted_curve": np.array([]),
            "error": "Time series too short for cosinor analysis (n < 10)",
        }

    # Create time array if not provided
    if timestamps is None:
        timestamps = np.arange(len(time_series)) * sampling_interval

    # Convert timestamps to hours
    time_hours = timestamps / 3600.0

    # Remove NaN values
    mask = ~np.isnan(time_series)
    if np.sum(mask) < 10:
        return {
            "mesor": np.nan,
            "amplitude": np.nan,
            "phase_angle_rad": np.nan,
            "peak_time": np.nan,
            "period": period_hours,
            "p_value": 1.0,
            "r_squared": 0.0,
            "significant": False,
            "fitted_curve": np.array([]),
            "error": "Too many NaN values in time series",
        }

    time_clean = time_hours[mask]
    data_clean = time_series[mask]

    # Angular frequency (radians per hour)
    omega = 2 * np.pi / period_hours

    # Create design matrix for least squares regression
    # y = M + A*cos(ωt) + B*sin(ωt)
    # where A = Amplitude*cos(φ), B = -Amplitude*sin(φ)
    # This can be rewritten as: y = β0 + β1*cos(ωt) + β2*sin(ωt)

    X = np.column_stack(
        [
            np.ones(len(time_clean)),  # Intercept (MESOR)
            np.cos(omega * time_clean),  # Cosine component
            np.sin(omega * time_clean),  # Sine component
        ]
    )

    # Perform least squares regression
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, data_clean, rcond=None)

        # Extract parameters
        mesor = beta[0]   # β0
        beta_cos = beta[1]  # β1
        beta_sin = beta[2]  # β2

        # Calculate amplitude and phase angle φ
        # Model: y = MESOR + β1·cos(ωt) + β2·sin(ωt)
        #           = MESOR + A·cos(ωt + φ)
        # where β1 = A·cos(φ), β2 = −A·sin(φ)  → φ = arctan2(−β2, β1)
        amplitude = np.sqrt(beta_cos**2 + beta_sin**2)
        phase_angle_rad = np.arctan2(-beta_sin, beta_cos)  # φ in radians, range (−π, π]

        # Time of fitted cosine peak within one period
        # Peak of cos(ωt + φ) occurs at ωt + φ = 0  →  t_peak = −φ/ω
        peak_time = (-phase_angle_rad / omega) % period_hours

        # Generate fitted curve for full time range
        fitted_curve = (
            mesor
            + beta_cos * np.cos(omega * time_hours)
            + beta_sin * np.sin(omega * time_hours)
        )

        # Calculate R-squared
        ss_res = np.sum(
            (
                data_clean
                - (
                    mesor
                    + beta_cos * np.cos(omega * time_clean)
                    + beta_sin * np.sin(omega * time_clean)
                )
            )
            ** 2
        )
        ss_tot = np.sum((data_clean - np.mean(data_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate F-statistic for significance testing
        n = len(data_clean)
        k = 2  # Number of predictors (cos and sin components)
        df_model = k
        df_residual = n - k - 1

        if df_residual > 0 and ss_tot > 0:
            mse_residual = ss_res / df_residual
            mse_model = (ss_tot - ss_res) / df_model
            f_statistic = mse_model / mse_residual if mse_residual > 0 else 0
            p_value = 1 - stats.f.cdf(f_statistic, df_model, df_residual)
        else:
            f_statistic = 0
            p_value = 1.0

        significant = p_value < alpha

        # Calculate confidence intervals (approximate)
        if df_residual > 0:
            mse = ss_res / df_residual
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                var_beta = mse * np.diag(XtX_inv)
                se_beta = np.sqrt(var_beta)

                # CI for MESOR
                t_crit = stats.t.ppf(1 - alpha / 2, df_residual)
                ci_mesor = (mesor - t_crit * se_beta[0], mesor + t_crit * se_beta[0])

                # CI for amplitude (delta method — approximate)
                # SE(A) ≈ sqrt[(β1·SE(β1))² + (β2·SE(β2))²] / A
                if amplitude > 0:
                    se_amp = (
                        np.sqrt(
                            (beta_cos * se_beta[1]) ** 2 + (beta_sin * se_beta[2]) ** 2
                        )
                        / amplitude
                    )
                    ci_amplitude = (
                        max(0, amplitude - t_crit * se_amp),
                        amplitude + t_crit * se_amp,
                    )
                else:
                    ci_amplitude = (0, 0)

                # CI for peak_time via delta method
                # φ = arctan2(−β2, β1)  →  ∂φ/∂β1 = β2/A²,  ∂φ/∂β2 = −β1/A²
                # se(φ)² = (β2²·var(β1) + β1²·var(β2)) / A⁴
                # t_peak = −φ/ω  →  se(t_peak) = se(φ)/ω
                A_sq = beta_cos**2 + beta_sin**2
                if A_sq > 0:
                    se_phi = np.sqrt(
                        beta_sin**2 * var_beta[1] + beta_cos**2 * var_beta[2]
                    ) / A_sq
                    se_peak_time = se_phi / omega
                    ci_peak_time = (
                        (peak_time - t_crit * se_peak_time) % period_hours,
                        (peak_time + t_crit * se_peak_time) % period_hours,
                    )
                else:
                    ci_peak_time = (0, 0)
            except np.linalg.LinAlgError:
                ci_mesor = (np.nan, np.nan)
                ci_amplitude = (np.nan, np.nan)
                ci_peak_time = (np.nan, np.nan)
        else:
            ci_mesor = (np.nan, np.nan)
            ci_amplitude = (np.nan, np.nan)
            ci_peak_time = (np.nan, np.nan)

        return {
            "mesor": float(mesor),
            "amplitude": float(amplitude),
            "beta_cos": float(beta_cos),   # cosine coefficient — needed for population F-test
            "beta_sin": float(beta_sin),   # sine coefficient  — needed for population F-test
            "phase_angle_rad": float(phase_angle_rad),  # φ — phase offset of the cosine
            "peak_time": float(peak_time),              # −φ/ω mod T — time of fitted peak (h)
            "period": float(period_hours),
            "p_value": float(p_value),
            "f_statistic": float(f_statistic),
            "r_squared": float(r_squared),
            "significant": bool(significant),
            "fitted_curve": fitted_curve,
            "ci_mesor": ci_mesor,
            "ci_amplitude": ci_amplitude,
            "ci_peak_time": ci_peak_time,
            "n_observations": int(n),
            "df_residual": int(df_residual),
        }

    except Exception as e:
        return {
            "mesor": np.nan,
            "amplitude": np.nan,
            "phase_angle_rad": np.nan,
            "peak_time": np.nan,
            "period": period_hours,
            "p_value": 1.0,
            "r_squared": 0.0,
            "significant": False,
            "fitted_curve": np.array([]),
            "error": f"Cosinor analysis failed: {str(e)}",
        }


def multi_period_cosinor(
    time_series: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    test_periods: Optional[List[float]] = None,
    sampling_interval: float = 60.0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test multiple periods with cosinor analysis to find best-fitting rhythm.

    Args:
        time_series: 1D array of activity values
        timestamps: Optional array of timestamps in seconds
        test_periods: List of periods to test (hours). If None, tests
                     common circadian/ultradian periods: [12, 18, 24, 30, 36]
        sampling_interval: Time between samples in seconds
        alpha: Significance level for statistical tests

    Returns:
        Dictionary containing:
        - all_results: List of results for each tested period
        - best_period: Period with highest R-squared
        - best_result: Full cosinor result for best period
        - tested_periods: Array of all tested periods
    """
    if test_periods is None:
        # Default: test circadian and common ultradian/infradian periods
        test_periods = [12.0, 18.0, 24.0, 30.0, 36.0]

    all_results = []
    for period in test_periods:
        result = single_cosinor_analysis(
            time_series=time_series,
            timestamps=timestamps,
            period_hours=period,
            sampling_interval=sampling_interval,
            alpha=alpha,
        )
        result["test_period"] = period
        all_results.append(result)

    # Find best fitting period (highest R-squared among significant results)
    significant_results = [r for r in all_results if r.get("significant", False)]

    if significant_results:
        best_result = max(significant_results, key=lambda x: x.get("r_squared", 0))
        best_period = best_result["period"]
    else:
        # If no significant results, take highest R-squared overall
        best_result = max(all_results, key=lambda x: x.get("r_squared", 0))
        best_period = best_result["period"]

    return {
        "all_results": all_results,
        "best_period": float(best_period),
        "best_result": best_result,
        "tested_periods": np.array(test_periods),
    }


def population_cosinor(
    time_series_list: List[np.ndarray],
    timestamps_list: Optional[List[np.ndarray]] = None,
    period_hours: float = 24.0,
    sampling_interval: float = 60.0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform population-mean cosinor analysis on multiple individuals.

    This method tests whether a common rhythm exists across a population
    and estimates population-level parameters.

    Args:
        time_series_list: List of time series arrays (one per individual)
        timestamps_list: Optional list of timestamp arrays
        period_hours: Period to test (default: 24h)
        sampling_interval: Time between samples in seconds
        alpha: Significance level

    Returns:
        Dictionary containing:
        - population_mesor: Population-level MESOR
        - population_amplitude: Population-level amplitude
        - population_peak_time: Circular mean of individual peak_time values (hours).
                                Not a biological acrophase without ZT reference.
        - p_value: Test for zero amplitude (no rhythm)
        - significant: Whether population rhythm is significant
        - individual_results: List of individual cosinor results
        - n_individuals: Number of individuals analyzed
        - n_significant: Number of individuals with significant rhythms
    """
    n_individuals = len(time_series_list)

    if n_individuals == 0:
        return {
            "error": "No time series provided",
            "n_individuals": 0,
        }

    # Analyze each individual
    individual_results = []
    for i, ts in enumerate(time_series_list):
        timestamps = timestamps_list[i] if timestamps_list is not None else None
        result = single_cosinor_analysis(
            time_series=ts,
            timestamps=timestamps,
            period_hours=period_hours,
            sampling_interval=sampling_interval,
            alpha=alpha,
        )
        individual_results.append(result)

    # Extract individual amplitudes and peak times
    valid_results = [
        r for r in individual_results if not np.isnan(r.get("amplitude", np.nan))
    ]

    if not valid_results:
        return {
            "error": "No valid individual results",
            "n_individuals": n_individuals,
            "individual_results": individual_results,
        }

    # Convert peak_time (hours) to phase angles for circular averaging
    # peak_time ∈ [0, T)  →  angle = peak_time * 2π / T  ∈ [0, 2π)
    amplitudes = np.array([r["amplitude"] for r in valid_results])
    peak_time_angles = np.array(
        [r["peak_time"] * 2 * np.pi / period_hours for r in valid_results]
    )
    mesors = np.array([r["mesor"] for r in valid_results])

    # Population MESOR (simple mean)
    pop_mesor = np.mean(mesors)

    # Population amplitude and peak_time using vector averaging
    x_coords = amplitudes * np.cos(peak_time_angles)
    y_coords = amplitudes * np.sin(peak_time_angles)

    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)

    # Population amplitude (length of mean vector)
    pop_amplitude = np.sqrt(mean_x**2 + mean_y**2)

    # Population peak_time (angle of mean vector, converted back to hours)
    pop_peak_time_rad = np.arctan2(mean_y, mean_x)
    pop_peak_time = (pop_peak_time_rad * period_hours / (2 * np.pi)) % period_hours

    # Population cosinor F-test (Nelson et al. 1979 / Cornelissen 2014)
    # Tests H₀: mean β_cos = mean β_sin = 0  (no population rhythm)
    # F = [n(β̄²_cos + β̄²_sin) / 2] / [Σ(β_cos_i − β̄_cos)² + Σ(β_sin_i − β̄_sin)²) / (2(n−1))]
    # df1 = 2, df2 = 2(n − 1)
    n = len(valid_results)
    beta_cos_vals = np.array([r.get("beta_cos", r["amplitude"] * np.cos(-r["phase_angle_rad"])) for r in valid_results])
    beta_sin_vals = np.array([r.get("beta_sin", r["amplitude"] * np.sin(-r["phase_angle_rad"])) for r in valid_results])

    mean_bc = np.mean(beta_cos_vals)
    mean_bs = np.mean(beta_sin_vals)

    # Sum of squared deviations
    ss_between = n * (mean_bc ** 2 + mean_bs ** 2)
    ss_within = np.sum((beta_cos_vals - mean_bc) ** 2 + (beta_sin_vals - mean_bs) ** 2)

    if n > 1 and ss_within > 0:
        f_pop = (ss_between / 2.0) / (ss_within / (2 * (n - 1)))
        p_value = float(1.0 - stats.f.cdf(f_pop, dfn=2, dfd=2 * (n - 1)))
    else:
        f_pop = np.nan
        p_value = 1.0

    significant = p_value < alpha

    # Count significant individuals
    n_significant = sum(1 for r in individual_results if r.get("significant", False))

    return {
        "population_mesor": float(pop_mesor),
        "population_amplitude": float(pop_amplitude),
        "population_peak_time": float(pop_peak_time),  # circular mean, NOT biological acrophase
        "period": float(period_hours),
        "p_value": float(p_value),
        "f_statistic": float(f_pop) if not np.isnan(f_pop) else None,
        "significant": bool(significant),
        "individual_results": individual_results,
        "n_individuals": int(n_individuals),
        "n_significant": int(n_significant),
        "proportion_significant": float(n_significant / n_individuals),
    }
