# Circadian Rhythm Analysis with Fischer Z-Transformation

## Overview

The Extended Analysis tab implements **Fischer Z-transformation** for detecting periodic patterns in biological activity data. This statistical method is widely used in chronobiology research to identify circadian rhythms, ultradian rhythms (< 24h), and infradian rhythms (> 24h) in timeseries data.

## What is a Periodogram?

A **periodogram** is a frequency-domain representation of timeseries data that reveals periodic (repeating) patterns. It transforms temporal data into the frequency space to identify dominant cycles.

### Key Concepts

- **Period (T)**: The length of one complete cycle (e.g., 24 hours for circadian rhythms)
- **Frequency (f)**: The reciprocal of period (f = 1/T)
- **Amplitude**: The strength or power of a periodic component
- **Phase**: The timing offset of a rhythm relative to a reference point

## Fischer Z-Transformation

### Mathematical Background

The Fischer Z-transformation tests whether a timeseries contains significant periodic components. For a given period T:

1. **Decompose the signal** into sine and cosine components:
   ```
   x(t) ≈ A·cos(2πt/T) + B·sin(2πt/T) + C
   ```

2. **Calculate the periodogram value** (normalized power):
   ```
   P(T) = (A² + B²) / variance(x)
   ```

3. **Compute the Z-score**:
   ```
   Z = √(2N·P)
   ```
   where N is the number of data points

4. **Statistical significance**:
   - Under the null hypothesis (random noise), Z² follows a chi-square distribution with 2 degrees of freedom
   - Critical value for α=0.05: Z_crit ≈ 2.45
   - If Z > Z_crit, the period is statistically significant

### Why Fischer Z-Transformation?

Traditional periodogram analysis (e.g., Fourier transform) can produce spurious peaks in noisy biological data. Fischer's method provides:

- **Statistical rigor**: Built-in significance testing
- **Robustness**: Less sensitive to irregular sampling or missing data
- **Biological relevance**: Focuses on a specified period range (e.g., 12-36h)
- **Multiple testing correction**: Accounts for testing multiple periods

## Implementation Details

### Algorithm Steps

1. **Data Preparation**
   - Extract movement activity data from main analysis
   - Sampling interval: Defined by frame interval (default: 5 seconds)
   - Data type: Fraction of time active in each time bin

2. **Period Range Configuration**
   - Minimum period: 0-100 hours (default: 12h)
   - Maximum period: 0-100 hours (default: 36h)
   - Resolution: 0.1 hours
   - Rationale: Covers circadian (24h), ultradian (<24h), and infradian (>24h) rhythms

3. **For Each ROI**:
   - Detrend the timeseries (remove linear trend)
   - Normalize to zero mean, unit variance
   - For each period in range:
     - Fit sine/cosine components
     - Calculate Z-score
   - Identify dominant period (maximum Z-score)
   - Test significance against critical Z value

4. **Sleep/Wake Phase Detection**
   - Use dominant period to fit a cosine function
   - Phase threshold (default: 0.5): Values above = wake, below = sleep
   - Output: Predicted sleep and wake phases throughout recording

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Minimum Period** | 12.0 h | 0-100 h | Lower bound of period search range |
| **Maximum Period** | 36.0 h | 0-100 h | Upper bound of period search range |
| **Significance Level (α)** | 0.05 | 0.001-0.1 | False positive rate for significance testing |
| **Phase Threshold** | 0.5 | 0.0-1.0 | Threshold for classifying sleep vs wake |

### Output

**Text Results:**
```
ROI 1: SIGNIFICANT circadian rhythm detected
  Dominant Period: 24.2 ± 0.8 hours
  Z-score: 8.45 (p < 0.001)
  Predicted Sleep Phases:
    - Phase 1: 0.0 - 12.1 hours (12.1h duration)
    - Phase 2: 24.2 - 36.3 hours (12.1h duration)
  Predicted Wake Phases:
    - Phase 1: 12.1 - 24.2 hours (12.1h duration)
```

**Periodogram Plot:**
- X-axis: Period (hours)
- Y-axis: Z-score
- Blue line: Z-scores for all tested periods
- Red dashed line: Significance threshold
- Red marker: Dominant period (if significant)
- ROI title color: Green = significant, Black = not significant

## Biological Interpretation

### Circadian Rhythms (≈24h)

**Strong 24h peak:**
- Organism is entrained to light/dark cycle
- Clear day/night activity pattern
- Functional circadian clock

**Weak or absent 24h peak:**
- Arrhythmic behavior
- Clock dysfunction
- Constant environmental conditions

### Ultradian Rhythms (<24h)

**Peak at 12h:**
- Bimodal activity (e.g., morning and evening peaks)
- Common in crepuscular species
- May reflect feeding or tidal rhythms

**Peak at 8h or shorter:**
- High-frequency activity bouts
- May indicate stress or abnormal behavior

### Infradian Rhythms (>24h)

**Peak at 48h or longer:**
- Multi-day cycles
- Lunar or tidal influences
- Developmental rhythms

### No Significant Peaks

- Random or irregular activity
- Exploratory behavior
- Response to unpredictable environmental cues
- Technical issues (low temporal resolution, short recording duration)

## Best Practices

### Recording Duration

- **Minimum**: 3-4 cycles of the expected period
  - For 24h rhythm: At least 3-4 days
  - For 12h rhythm: At least 2 days
- **Recommended**: 5-7 days for robust circadian analysis
- **Longer is better**: Improves statistical power and phase estimation

### Sampling Interval

- **Frame interval**: 5 seconds (default) is sufficient for most organisms
- **Nyquist criterion**: Sample at least twice per cycle
  - For 12h rhythm: Sample at least every 6 hours
  - For 24h rhythm: Sample at least every 12 hours
- **Trade-off**: Higher resolution = larger file size and longer processing

### Environmental Control

- **Light/Dark cycles**: Ensure stable, programmed lighting
- **Temperature**: Maintain constant temperature (±1°C)
- **Feeding**: Regular feeding schedule or fast during recording
- **Isolation**: Minimize external disturbances

### Data Quality

- **Sufficient activity**: ROI must show measurable movement
- **Consistent tracking**: ROI should not drift or be lost
- **No gaps**: Continuous recording without interruptions
- **Baseline period**: Include enough baseline data for threshold calculation

## Troubleshooting

### No Significant Rhythms Detected

**Possible causes:**
1. **Recording too short**: Extend recording duration to 5-7 days
2. **Activity too low**: Check ROI detection and baseline thresholds
3. **Period range wrong**: Adjust min/max period to expected range
4. **Organism arrhythmic**: Some individuals naturally lack strong rhythms
5. **Environmental noise**: Improve isolation and temperature control

**Solutions:**
- Increase recording duration
- Verify ROI detection quality
- Try different period ranges (e.g., 6-30h for ultradian)
- Reduce significance level (e.g., α = 0.1) for exploratory analysis

### Multiple Significant Peaks

**Interpretation:**
- **Harmonics**: Peaks at 24h and 12h (harmonic relationship)
  - Primary rhythm is at longer period (24h)
  - Shorter peak (12h) is a harmonic
- **Multiple rhythms**: Independent biological rhythms
  - Example: Circadian (24h) + tidal (12.4h)
- **Spurious peaks**: Statistical artifacts
  - Check if peaks are narrow (< 2h width) = likely artifact

**Solutions:**
- Focus on the dominant (highest Z-score) peak
- Check biological plausibility
- Compare across multiple individuals/ROIs

### Inconsistent Results Across ROIs

**Possible causes:**
1. **Individual variation**: Natural differences in rhythm strength
2. **ROI size mismatch**: Different ROI sizes affect signal-to-noise
3. **Partial organism**: ROI doesn't capture full organism
4. **Edge effects**: ROI partially out of frame

**Solutions:**
- Report population statistics (mean ± SEM)
- Exclude outlier ROIs with abnormal detection
- Ensure ROIs are properly centered and sized

## References

### Key Papers

1. **Fischer, R. A. (1929)** - "Tests of Significance in Harmonic Analysis"
   - Original description of the Z-transformation method

2. **Sokolove, P. G., & Bushell, W. N. (1978)** - "The chi square periodogram: Its application to the analysis of circadian rhythms"
   - Chronobiology application of Fischer's method
   - Widely cited in circadian research

3. **Levine, J. D., et al. (2002)** - "Signal analysis of behavioral and molecular cycles"
   - Modern applications in Drosophila circadian biology

### Online Resources

- **Circadian rhythm analysis**: https://www.circadian.org/
- **Chronobiology tools**: https://www.euclock.org/
- **Statistical methods**: https://www.chronobiology.com/

## Citation

If you use the Extended Analysis feature in your research, please cite:

```bibtex
@software{napari_hdf5_activity_extended,
  author = {s1alknau},
  title = {napari-hdf5-activity: Extended Analysis with Fischer Z-transformation},
  year = {2025},
  url = {https://github.com/s1alknau/napari-hdf5-activity}
}
```

And cite the foundational work:

```bibtex
@article{sokolove1978chi,
  title={The chi square periodogram: its application to the analysis of circadian rhythms},
  author={Sokolove, PG and Bushell, WN},
  journal={Journal of theoretical biology},
  volume={72},
  number={1},
  pages={131--160},
  year={1978},
  publisher={Elsevier}
}
```

## See Also

- [Main README](../README.md) - Plugin overview and installation
- [User Guide](USER_GUIDE.md) - Step-by-step usage instructions
- [Technical Documentation](TECHNICAL.md) - Code architecture and API reference
