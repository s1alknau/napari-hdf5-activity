# Circadian Rhythm Analysis

## Overview

The Extended Analysis tab provides three complementary methods for detecting and
quantifying periodic patterns in biological activity data:

| Method | Best For | Requires |
|--------|----------|----------|
| **Chi² Periodogram** | Exploratory period detection, robust to non-sinusoidal signals | ≥ 3 cycles |
| **Cosinor Analysis** | Quantifying amplitude and acrophase of a known period | ≥ 7 cycles, sinusoidal signal |
| **Population Mean** | Cross-individual consistency, SEM across ROIs | ≥ 2 ROIs |

---

## Chi² Periodogram

### Principle

The Chi² periodogram (Sokolove & Bushell 1978) tests whether a timeseries contains
statistically significant periodic components. For each candidate period T:

1. Fold the timeseries into T-length epochs
2. Compute correlation coefficients with sine and cosine at period T
3. Derive the Chi² statistic: `Q = n × (r²_cos + r²_sin)`
4. Convert to Z-score for display

### Z-score

```
Z-score = f(Chi², n)

n = number of data points ∝ recording duration
```

**Important:** The Z-score is NOT a pure measure of rhythm strength. It depends on
both rhythm quality AND sample size:

- Longer recording → more data points (n↑) → higher Z-score for the same rhythm
- Z-scores from different time ranges or different recording durations are **not
  directly comparable**
- Use **Amplitude** (Cosinor) for comparing rhythm strength between experiments

### Significance threshold

Under the null hypothesis (white noise), the Chi² statistic follows a chi-square
distribution with 2 degrees of freedom. The critical Z-value at α = 0.05 is ~5.99
(Bonferroni-corrected for the number of tested periods).

### Period Range vs. Time Range

These are two fundamentally different settings:

| Setting | What it changes | Effect on Z-score |
|---------|----------------|-------------------|
| **Period Range** (min/max) | Which periods are searched (X-axis zoom) | None — same data, same Z-scores |
| **Time Range** (start/end) | Which data points are included | Yes — fewer points = lower Z-score |

→ Adjusting the period range is always safe.
→ Adjusting the time range changes the analysis fundamentally.

### Population Mean Panel

Shown automatically when ≥ 2 ROIs are analyzed:

- **Black line**: Mean Z-score across all ROIs at each tested period
- **Grey band**: ± SEM (Standard Error of the Mean)
- **SEM** = SD / √n_rois — describes uncertainty of the mean, not biological variability
- A wide SEM band indicates heterogeneous periods across individuals (e.g., two
  sub-groups with different tau)
- **Median peak** (blue dashed): median of individual dominant periods

---

## Cosinor Analysis

### Principle

Fits the model `y(t) = MESOR + Amplitude × cos(2πt/τ + φ)` to the timeseries.

| Parameter | Meaning |
|-----------|---------|
| **MESOR** | Midline Estimating Statistic of Rhythm — rhythm-adjusted mean level |
| **Amplitude** | Half the peak-to-trough difference — biological rhythm strength |
| **φ (phase)** | Phase offset of the fitted cosine |
| **Peak Time** | Time from recording start to first cosine peak |
| **R²** | Goodness of fit (proportion of variance explained by the cosine) |

### R² interpretation

| R² | Meaning |
|----|---------|
| > 0.30 | Strong rhythmic pattern |
| 0.10 – 0.30 | Moderate rhythm |
| < 0.10 | Weak or absent rhythm — cosine is a poor fit |

### Why Cosinor needs long recordings

The Cosinor fits 3 free parameters (MESOR, Amplitude, Phase) to noisy data.
Each additional cycle reduces estimation error by √n:

| Recording | Cycles (24h) | Expected R² | Phase CI |
|-----------|-------------|-------------|----------|
| 3 days | 3 | ~0.05–0.11 | ± 0.6 h |
| 7 days | 7 | ~0.20–0.35 | ± 0.3 h |
| 14 days | 14 | ~0.35–0.50 | ± 0.2 h |

**Minimum: 7 cycles (= 7 days for 24h rhythm)**

### p-values in Cosinor

With large datasets (thousands of frames), p-values will be extremely small
(p < 1e-300) even for biologically weak rhythms. Do not use p-value alone as
evidence of a strong rhythm — always report R² and Amplitude.

### Acrophase and ZT reference

The plugin outputs **Peak Time** = time from recording start to first cosine peak.
To convert to Zeitgeber Time (ZT):

```
Acrophase (ZT) = (Peak Time + ZT of recording start) mod 24
```

**Example:** Recording started at ZT7 (7 h after lights-on), Peak Time = 12.5 h:
```
Acrophase = (12.5 + 7) mod 24 = ZT 19.5  (7.5 h into dark phase)
```

If the recording started at ZT0 (lights-on = recording start): no correction needed.

> **Always document:** recording start clock time AND ZT0 clock time.

---

## Data Sources

### Fraction Movement

- Computed as: `active time (s) / bin size (s)` per time bin
- Time-based (not frame-based) → robust to irregular frame intervals
- Continuous signal in [0, 1]
- Threshold-dependent: the hysteresis threshold determines what counts as "active"
- **Standard for circadian biology** (comparable to running wheel activity counts)
- Recommended for: Chi² periodogram, actogram visualization

### Raw Intensity

- Per-frame pixel change values (frame differences), MinMax-normalized per ROI
- Continuous signal, preserves amplitude information
- Independent of movement threshold
- Recommended for: Cosinor analysis, detecting subtle rhythms below the
  movement threshold

### When to use which

| Analysis | Recommended source | Reason |
|----------|-------------------|--------|
| Chi² Periodogram | Fraction Movement | Literature standard, robust |
| Cosinor | Raw Intensity | Assumes sinusoidal, benefits from amplitude info |
| Actogram | Fraction Movement | Interpretable as % time active |

---

## Sleep Analysis

### Derivation chain

```
Raw signal → Movement detection → Fraction Movement → Quiescence → Sleep (≥ 8 min)
```

Sleep is derived from activity — the two periodograms are **not independent**.
Dominant periods will typically overlap between Activity and Sleep analyses.

| Data source | Definition | Use |
|-------------|-----------|-----|
| **Quiescence** | Movement fraction < threshold (per bin) | Direct complement of activity, same resolution |
| **Sleep (≥ 8 min)** | Sustained quiescence ≥ 8 consecutive minutes | Biologically strict, low-pass filtered |

The parallel Sleep periodogram is confirmatory, not independent:
it shows whether the rhythm also appears in consolidated rest behavior.

---

## Free-Running vs. Entrained Rhythms

### Free-running (constant conditions, DD or LL)

- Period τ ≠ 24 h (intrinsic clock period)
- Typical range in animals: 20–28 h
- Drifts relative to external time (no synchronization)
- Chi² periodogram: sharp peak at τ

### Entrained (under LD cycle)

- Period converges to exactly 24 h (or the Zeitgeber period)
- Stable acrophase relative to ZT
- Requires 2–5 transient cycles after LD onset before stable entrainment
- Chi² periodogram: peak at 24 h

### Transient phase

When animals are transferred from DD to LD (or vice versa), the first 2–5 cycles
show a gradual phase shift. Analyzing transient + entrained data together in a single
Cosinor fit degrades R² because the model assumes a single constant period.

**Solution:** Analyze segments separately using Time Range Selection:

```
Full recording:    [DD] ──── [LD transient] ──── [LD stable] ──── [DD post]
Time Range:         └── τ_1 ──┘               └── Acrophase ──┘  └── τ_2 ──┘
```

---

## Adaptive Illumination Baseline

When analyzing recordings with light/dark cycles (LD), the raw activity signal has
different baseline levels during the light and dark phases. The Adaptive Illumination
Baseline option:

1. Detects period boundaries from HDF5 LED data
2. Computes the resting floor (15th percentile) for each period
3. Shifts each period's signal to a common global reference level
4. Then computes a single global threshold on the equalized signal

This ensures the hysteresis detection works correctly across all illumination phases
without requiring per-period thresholds.

---

## Best Practices

### Recording duration

| Goal | Minimum | Recommended |
|------|---------|-------------|
| Chi² period detection | 3 cycles (3 days for τ=24h) | 5–7 days |
| Cosinor fit | 7 cycles | 10–14 days |
| Entrainment verification | 5 days LD + 5 days DD | 7 days LD + 7 days DD |

### Period range

- Set wider than expected: 16–36 h covers circadian rhythms safely
- Check for boundary warnings (⚠️) — if the peak is at the boundary, extend the range
- Extending the period range does not affect Z-scores

### Time range

- Use Full Recording by default
- Use segment analysis only when recording is long enough to split into
  transient and stable phases (≥ 7 days total)
- Minimum window for reliable analysis: 5 × expected period

### Environmental control

- **Temperature**: constant ± 0.5 °C (temperature is itself a Zeitgeber)
- **Feeding**: document time; regular feeding is a Zeitgeber
- **Vibration/sound**: minimize at fixed times (can mask rhythms)
- **ZT0 consistency**: use the same lights-on time across experiments

### Segment analysis for entrainment experiments

Do not fit Cosinor across the full LD recording if it includes the transient phase.
Use Time Range Selection:

- Chi² on full recording → overview, period visible in all phases
- Cosinor on stable phase only (e.g., days 4–7 of LD) → reliable acrophase

---

## Troubleshooting

### Peak at period boundary (⚠️)

**Cause:** True peak is outside the tested range.
**Fix:** Extend min/max period range (e.g., from 20–28 h → 16–36 h).
Does not change any Z-scores, only expands the search space.

### Low R² in Cosinor (< 0.10)

**Causes:**
- Too few cycles (< 7 days)
- Mixed transient + stable phases in the same fit
- Signal is non-sinusoidal (use Chi² instead)
- LD recording: baseline level differs between light/dark (enable Adaptive Illumination Baseline)

### Z-score drops when narrowing Time Range

**Expected behavior.** Z-score depends on number of data points (n).
Fewer data = lower Z-score even for identical rhythm quality.
Always compare Z-scores from the same time range.

### Periods ≠ 24 h under LD 12:12

**Possible causes:**
1. Animals still in transient phase (first 2–3 days of LD) → wait or extend recording
2. Animals not entrained → check light intensity, check temperature constancy
3. Masking without entrainment → verify with DD phase after LD

### Sleep and Activity show identical periods

Expected — sleep is derived from activity. The overlap confirms the same biological
clock drives both behaviors. Check Z-score and amplitude differences between the two
for additional insight.

---

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| Min Period | 12.0 h | Lower bound of period search range |
| Max Period | 36.0 h | Upper bound of period search range |
| Significance α | 0.05 | False positive rate |
| Data Source | Fraction Movement | Input signal for periodogram |
| Sleep Source | Sleep ≥ 8 min | Input for sleep phase periodogram |
| Time Range | Full Recording | Data window for analysis |
| Bin Size | 60 s | Time bin for fraction movement calculation |

---

## References

1. **Sokolove, P. G., & Bushell, W. N. (1978)**
   The chi square periodogram: its application to the analysis of circadian rhythms.
   *Journal of Theoretical Biology*, 72(1), 131–160.

2. **Nelson, W., et al. (1979)**
   Methods for cosinor rhythmometry.
   *Chronobiologia*, 6(4), 305–323.

3. **Pittendrigh, C. S., & Daan, S. (1976)**
   A functional analysis of circadian pacemakers in nocturnal rodents.
   *Journal of Comparative Physiology*, 106(3), 223–252.

4. **Aschoff, J. (1965)**
   Circadian Clocks. North-Holland Publishing, Amsterdam.

5. **Aguillon, R., et al. (2023)**
   Quantification of activity and rest in *Nematostella vectensis*.

---

## See Also

- [Entrainment Protocol](entrainment_protocol_EN.txt) — Step-by-step experimental design
- [Main README](../README.md) — Plugin overview and installation
