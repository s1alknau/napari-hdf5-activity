"""
_cosinor_analysis.py - Cosinor analysis for circadian rhythm quantification

This module implements single and population cosinor analysis, widely used
in chronobiology to quantify circadian rhythms. Cosinor analysis fits a
cosine curve to time series data and extracts key rhythmic parameters:

- MESOR (Midline Estimating Statistic of Rhythm): Mean activity level
- Amplitude: Half the difference between peak and trough
- Acrophase: Time of peak activity (phase)
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

    Fits a cosine function: y(t) = MESOR + Amplitude * cos(2π*t/τ + Acrophase)
    where τ is the period being tested.

    Args:
        time_series: 1D array of activity values (e.g., fraction movement)
        timestamps: Optional array of timestamps in seconds. If None, uses
                   regular intervals based on sampling_interval
        period_hours: Period to test (default: 24h for circadian rhythm)
        sampling_interval: Time between samples in seconds (default: 60s)
        alpha: Significance level for statistical tests (default: 0.05)

    Returns:
        Dictionary containing:
        - mesor: Midline Estimating Statistic of Rhythm (mean level)
        - amplitude: Amplitude of the rhythm
        - acrophase: Phase angle at peak (hours from start)
        - acrophase_time: Clock time of peak activity
        - period: Tested period (hours)
        - p_value: Statistical significance of rhythm
        - r_squared: Goodness of fit
        - significant: Boolean indicating if rhythm is significant
        - fitted_curve: Fitted cosine curve values
        - ci_mesor: 95% confidence interval for MESOR
        - ci_amplitude: 95% confidence interval for amplitude
        - ci_acrophase: 95% confidence interval for acrophase
    """
    # Input validation
    if len(time_series) < 10:
        return {
            "mesor": np.nan,
            "amplitude": np.nan,
            "acrophase": np.nan,
            "acrophase_time": np.nan,
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
            "acrophase": np.nan,
            "acrophase_time": np.nan,
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
        # Use numpy's least squares
        beta, residuals, rank, s = np.linalg.lstsq(X, data_clean, rcond=None)

        # Extract parameters
        mesor = beta[0]  # β0
        beta_cos = beta[1]  # β1
        beta_sin = beta[2]  # β2

        # Calculate amplitude and acrophase
        amplitude = np.sqrt(beta_cos**2 + beta_sin**2)
        acrophase_rad = np.arctan2(-beta_sin, beta_cos)  # Phase in radians

        # Convert acrophase to hours (0-24 range)
        acrophase_hours = (acrophase_rad / omega) % period_hours

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
        # Standard error estimation
        if df_residual > 0:
            mse = ss_res / df_residual
            # Variance-covariance matrix
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                var_beta = mse * np.diag(XtX_inv)
                se_beta = np.sqrt(var_beta)

                # CI for MESOR
                t_crit = stats.t.ppf(1 - alpha / 2, df_residual)
                ci_mesor = (mesor - t_crit * se_beta[0], mesor + t_crit * se_beta[0])

                # CI for amplitude (using delta method - approximate)
                # SE(amplitude) ≈ sqrt[(β1*SE(β1))^2 + (β2*SE(β2))^2] / amplitude
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

                # CI for acrophase (circular statistics - approximate)
                # This is a simplified approach
                if amplitude > 0:
                    se_acrophase_rad = np.sqrt(
                        (se_beta[1] / beta_cos) ** 2 + (se_beta[2] / beta_sin) ** 2
                    )
                    se_acrophase_hours = se_acrophase_rad / omega
                    ci_acrophase = (
                        (acrophase_hours - t_crit * se_acrophase_hours) % period_hours,
                        (acrophase_hours + t_crit * se_acrophase_hours) % period_hours,
                    )
                else:
                    ci_acrophase = (0, 0)
            except np.linalg.LinAlgError:
                ci_mesor = (np.nan, np.nan)
                ci_amplitude = (np.nan, np.nan)
                ci_acrophase = (np.nan, np.nan)
        else:
            ci_mesor = (np.nan, np.nan)
            ci_amplitude = (np.nan, np.nan)
            ci_acrophase = (np.nan, np.nan)

        return {
            "mesor": float(mesor),
            "amplitude": float(amplitude),
            "acrophase": float(acrophase_hours),
            "acrophase_time": float(acrophase_hours),
            "period": float(period_hours),
            "p_value": float(p_value),
            "f_statistic": float(f_statistic),
            "r_squared": float(r_squared),
            "significant": bool(significant),
            "fitted_curve": fitted_curve,
            "ci_mesor": ci_mesor,
            "ci_amplitude": ci_amplitude,
            "ci_acrophase": ci_acrophase,
            "n_observations": int(n),
            "df_residual": int(df_residual),
        }

    except Exception as e:
        return {
            "mesor": np.nan,
            "amplitude": np.nan,
            "acrophase": np.nan,
            "acrophase_time": np.nan,
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
        - population_acrophase: Population-level acrophase
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

    # Extract individual amplitudes and acrophases
    valid_results = [
        r for r in individual_results if not np.isnan(r.get("amplitude", np.nan))
    ]

    if not valid_results:
        return {
            "error": "No valid individual results",
            "n_individuals": n_individuals,
            "individual_results": individual_results,
        }

    # Convert to polar coordinates for population averaging
    amplitudes = np.array([r["amplitude"] for r in valid_results])
    acrophases_rad = np.array(
        [r["acrophase"] * 2 * np.pi / period_hours for r in valid_results]
    )
    mesors = np.array([r["mesor"] for r in valid_results])

    # Population MESOR (simple mean)
    pop_mesor = np.mean(mesors)

    # Population amplitude and acrophase using vector averaging
    # Convert to Cartesian coordinates
    x_coords = amplitudes * np.cos(acrophases_rad)
    y_coords = amplitudes * np.sin(acrophases_rad)

    # Mean vector
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)

    # Population amplitude (length of mean vector)
    pop_amplitude = np.sqrt(mean_x**2 + mean_y**2)

    # Population acrophase (angle of mean vector)
    pop_acrophase_rad = np.arctan2(mean_y, mean_x)
    pop_acrophase_hours = (
        pop_acrophase_rad * period_hours / (2 * np.pi)
    ) % period_hours

    # Test for zero amplitude (Rayleigh test for uniformity)
    n = len(valid_results)
    R = n * pop_amplitude  # Resultant vector length
    z = R**2 / n  # Rayleigh's R²/n
    p_value = np.exp(-z) * (
        1
        + (2 * z - z**2) / (4 * n)
        - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)
    )

    significant = p_value < alpha

    # Count significant individuals
    n_significant = sum(1 for r in individual_results if r.get("significant", False))

    return {
        "population_mesor": float(pop_mesor),
        "population_amplitude": float(pop_amplitude),
        "population_acrophase": float(pop_acrophase_hours),
        "period": float(period_hours),
        "p_value": float(p_value),
        "significant": bool(significant),
        "individual_results": individual_results,
        "n_individuals": int(n_individuals),
        "n_significant": int(n_significant),
        "proportion_significant": float(n_significant / n_individuals),
    }
