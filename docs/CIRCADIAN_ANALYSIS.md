# Circadian Rhythm Analysis

## Overview

The Extended Analysis tab provides three complementary methods for detecting and
quantifying periodic patterns in biological activity data:

| Method | Best For | Requires |
|--------|----------|----------|
| **Chi² Periodogram** | Exploratory period detection, robust to non-sinusoidal signals | ≥ 3 cycles |
| **Cosinor Analysis** | Quantifying amplitude and acrophase of a known period | ≥ 3 cycles, sinusoidal signal |
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

### Z-score (chi-squared statistic)

The plot y-axis is labeled **"Z-score"** in the UI, but the quantity displayed is the
chi-squared statistic:

```
Z(T) = n × (r²_cos + r²_sin)

n = number of data points ∝ recording duration
```

Under H₀ this follows χ²(df=2). The name "Z-score" is a UI label convention — Z(T)
is not a standard normal Z-score.

#### Symbols

| Symbol | Meaning | Unit |
|:------:|---------|------|
| *T* | candidate period currently being tested | h |
| Δ*t* | sampling interval of the binned timeseries | s (converted to h internally) |
| *n* | number of samples in the analysed segment | — |
| *t* | sample index, 0 … *n*−1 | samples |
| ω | angular frequency of the test wave, 2π / (*T* / Δ*t*) | rad per **sample** |
| *r*<sub>cos</sub>, *r*<sub>sin</sub> | Pearson correlation of the signal with cos(ω*t*) and sin(ω*t*) | — |
| *Z*(*T*) | chi-squared statistic for period *T* | — |
| *m* | number of tested periods (100) | — |
| α | significance level before correction (0.05) | — |

Note that ω is expressed **per sample**, not per hour: the period is divided by the
sampling interval first (`period_samples`), so *t* can stay a plain sample counter.
The Cosinor does it the other way round — there ω is per hour.

#### Step by step, with the code for each step

Every step below shows the calculation first and the lines that perform it
directly underneath. All fragments are from `fisher_z_periodogram()` in
[`_fisher_analysis.py`](https://github.com/s1alknau/napari-hdf5-activity/blob/main/src/napari_hdf5_activity/_fisher_analysis.py).

**Step 1 — the sampling interval in hours, and the grid of candidate periods**

```
Δt = sampling_interval / 3600            (s → h)
T_i = min_period + i·(max_period − min_period)/(m−1),   i = 0 … m−1,   m = 100
```

```python
sampling_hours = sampling_interval / 3600.0            # Δt in hours          (line 53)
periods = np.linspace(min_period, max_period, 100)     # the m = 100 T values (line 65)
n = len(time_series)                                   # n                    (line 67)
t = np.arange(n)                                       # t = 0, 1, … n−1      (line 71)
```

`t` is a plain sample counter, not a time in hours — that choice is what makes
step 2 divide by Δ*t*.

---

**Step 2 — angular frequency of the test wave, for one candidate period**

```
T_samples = T / Δt                       period expressed in samples
ω = 2π / T_samples                       rad per sample
```

```python
for idx, period_hours in enumerate(periods):
    period_samples = period_hours / sampling_hours     # T in samples         (line 74)
    omega = 2 * np.pi / period_samples                 # ω = 2π/T             (line 75)
```

---

**Step 3 — build the two reference waves at exactly that period**

```
c(t) = cos(ωt)          s(t) = sin(ωt)
```

```python
    cos_component = np.cos(omega * t)                  # cos(ωt)              (line 77)
    sin_component = np.sin(omega * t)                  # sin(ωt)              (line 78)
```

Their amplitude is irrelevant: step 4 uses Pearson's *r*, which is invariant to
scale and offset, so only the shape of the wave enters the result.

---

**Step 4 — correlate the signal with both waves**

```
r_cos = corr(y, cos(ωt))               r_sin = corr(y, sin(ωt))
```

```python
    # np.corrcoef returns the 2×2 correlation matrix; [0, 1] is the
    # off-diagonal element — r between signal and reference wave.
    r_cos = np.corrcoef(time_series, cos_component)[0, 1]                    # (line 80)
    r_sin = np.corrcoef(time_series, sin_component)[0, 1]                    # (line 81)

    # A constant segment has zero variance → r is NaN. Treated as "no
    # correlation" so one flat ROI cannot poison the whole periodogram.
    if np.isnan(r_cos):
        r_cos = 0.0                                                          # (line 84)
    if np.isnan(r_sin):
        r_sin = 0.0                                                          # (line 86)
```

Two coefficients are needed because a single one would depend on where the
rhythm happens to sit in its cycle. Step 5 combines them into a phase-independent
quantity.

---

**Step 5 — the chi-squared statistic**

```
Z(T) = n · (r²_cos + r²_sin)
```

```python
    # Squaring drops the sign, summing both components removes the phase
    # dependence: the rhythm is found whether it aligns with cos, sin, or
    # anything in between. The factor n is why Z grows with recording length.
    z_scores[idx] = n * (r_cos**2 + r_sin**2)                                # (line 92)
```

---

**Step 6 — significance threshold, Bonferroni-corrected**

```
α_corr = α / m                           m = 100 tested periods
Z_crit = χ²(1 − α_corr, df = 2)          ≈ 15.2 for α = 0.05, m = 100
```

```python
m = len(periods)                                       # m                    (line 95)
corrected_alpha = significance_level / m               # α/m                  (line 96)
chi2_crit = stats.chi2.ppf(1 - corrected_alpha, df=2)  # Z_crit               (line 97)
```

`ppf` is the inverse CDF (percent point function), i.e. the quantile — the Z
value that the χ²(2) distribution exceeds only with probability α/m.

---

**Step 7 — dominant period and its p-value**

```
T_dom = argmax_T Z(T)
p     = 1 − F_χ²(Z(T_dom); df = 2)
```

```python
significant_mask = z_scores > critical_z               # all periods above it (line 100)
max_z_idx = np.argmax(z_scores)                        # tallest peak         (line 103)
dominant_period = periods[max_z_idx]                   # T_dom                (line 104)
p_value = 1 - stats.chi2.cdf(max_z_score, df=2)        # p                    (line 107)
```

The module is named after Fisher's Z-transformation for historical reasons; the
statistic it computes today is the classical Sokolove & Bushell chi-squared
periodogram, as the comment at line 89 states.

### Significance threshold

Under the null hypothesis (white noise), the Chi² statistic follows a chi-square
distribution with 2 degrees of freedom. With Bonferroni correction for m = 100 tested
periods, the critical threshold at α = 0.05 is χ²(1 − α/m, df=2) ≈ 15.2, **not** 5.99.
The uncorrected α = 0.05 critical value (5.99) is far too lenient when testing 100 periods
simultaneously and would produce many false positives.

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
- **Horizontal dashed line**: The Bonferroni-corrected significance threshold (≈ 15.2 for
  α = 0.05, m = 100 tested periods)

To increase n, pool several recordings — see [Batch Datasets](#batch-datasets-pooling-several-recordings).

---

## Cosinor Analysis

### Principle

Fits the model `y(t) = MESOR + Amplitude × cos(2πt/τ + φ)` to the timeseries.

| Parameter | Symbol | Meaning |
|-----------|:------:|---------|
| MESOR | *M* | Midline Estimating Statistic of Rhythm — rhythm-adjusted mean level |
| Amplitude | *A* | Half the peak-to-trough difference — biological rhythm strength |
| Acrophase | *φ* | Phase offset of the fitted cosine (radians) |
| Peak Time | *t*<sub>peak</sub> | Clock time of the first cosine peak (*φ* converted to hours from recording start) |
| Goodness of fit | *R*² | Proportion of variance explained by the fitted cosine |

#### From a non-linear model to linear least squares

The model as written is non-linear in *φ*, which would need an iterative fit. The
standard trick is the angle-sum identity:

```
A·cos(ωt + φ) = A·cosφ · cos(ωt) − A·sinφ · sin(ωt)
              = β₁ · cos(ωt)     + β₂ · sin(ωt)

with β₁ = A·cosφ   and   β₂ = −A·sinφ
```

So the model becomes **linear** in the three unknowns β₀, β₁, β₂:

```
y(t) = β₀ + β₁·cos(ωt) + β₂·sin(ωt)
```

which one ordinary least-squares call solves exactly — no starting values, no
convergence problems. *A* and *φ* are recovered afterwards:

```
A = √(β₁² + β₂²)          φ = arctan2(−β₂, β₁)          t_peak = (−φ/ω) mod τ
```

`t_peak` follows from asking where the cosine is maximal: cos(ω*t* + φ) peaks when
its argument is zero, i.e. ω*t* + φ = 0 → *t* = −φ/ω. The modulo folds that into the
first cycle, which is why Peak Time is the peak of the **first** oscillation.

#### Symbols

| Symbol | Meaning | Unit |
|:------:|---------|------|
| τ | period the fit is performed at (fixed — from the periodogram or the period field) | h |
| ω | angular frequency, 2π/τ | rad per **hour** |
| *t* | time since recording start | h |
| β₀ | intercept — *is* the MESOR *M* | signal units |
| β₁, β₂ | cosine and sine coefficient | signal units |
| *A* | amplitude, √(β₁² + β₂²) | signal units |
| φ | phase angle, arctan2(−β₂, β₁), range (−π, π] | rad |
| *t*<sub>peak</sub> | Peak Time, (−φ/ω) mod τ | h after recording start |
| SS<sub>res</sub>, SS<sub>tot</sub> | residual and total sum of squares | signal units² |
| *R*² | 1 − SS<sub>res</sub>/SS<sub>tot</sub> | — |
| *k* | number of rhythm predictors (2: cos and sin) | — |
| *F* | test statistic, MS<sub>model</sub>/MS<sub>res</sub>, df = (2, *n*−3) | — |

#### Step by step, with the code for each step

All fragments are from `cosinor_analysis()` in
[`_cosinor_analysis.py`](https://github.com/s1alknau/napari-hdf5-activity/blob/main/src/napari_hdf5_activity/_cosinor_analysis.py).

**Step 1 — angular frequency at the fitted period**

```
ω = 2π / τ                               rad per hour
```

```python
omega = 2 * np.pi / period_hours                        # ω = 2π/τ            (line 107)
```

τ is *fixed*, not fitted: it comes from the periodogram or the period field.
Unlike the Chi² periodogram, the time axis here is in hours, so ω is per hour
and every time that follows is a real duration.

---

**Step 2 — the design matrix of the linearised model**

```
y(t) = β₀·1 + β₁·cos(ωt) + β₂·sin(ωt)

X = [ 1   cos(ωt₁)   sin(ωt₁) ]
    [ 1   cos(ωt₂)   sin(ωt₂) ]
    [ ⋮      ⋮          ⋮     ]
```

```python
# One row per time point, three columns — one per unknown. Column 0 is all
# ones, so its coefficient is the intercept β₀ = MESOR.
X = np.column_stack(                                                       # (line 114)
    [
        np.ones(len(time_clean)),      # → β₀ (MESOR)                      (line 116)
        np.cos(omega * time_clean),    # → β₁ (cosine coefficient)         (line 117)
        np.sin(omega * time_clean),    # → β₂ (sine coefficient)           (line 118)
    ]
)
```

---

**Step 3 — solve for the coefficients**

```
β̂ = argmin_β ‖Xβ − y‖²                 →   β̂ = (XᵀX)⁻¹Xᵀy
M = β₀
```

```python
# Ordinary least squares in one step. Because the model is linear in β, this
# is the exact optimum — no iteration, no starting guess, no convergence.
beta, residuals, rank, s = np.linalg.lstsq(X, data_clean, rcond=None)      # (line 124)

mesor = beta[0]      # β₀ — the MESOR, read straight off the intercept     (line 127)
beta_cos = beta[1]   # β₁                                                  (line 128)
beta_sin = beta[2]   # β₂                                                  (line 129)
```

---

**Step 4 — amplitude**

```
A = √(β₁² + β₂²)
```

```python
# β₁ = A·cosφ and β₂ = −A·sinφ, so the two coefficients are the Cartesian
# form of one vector: its length is the amplitude, its angle the phase.
amplitude = np.sqrt(beta_cos**2 + beta_sin**2)                             # (line 135)
```

---

**Step 5 — phase angle**

```
φ = arctan2(−β₂, β₁)                     range (−π, π]
```

```python
# arctan2 (not arctan) keeps the quadrant, so φ is unambiguous over (−π, π].
# The minus sign on β₂ undoes the sign in β₂ = −A·sinφ.
phase_angle_rad = np.arctan2(-beta_sin, beta_cos)                          # (line 136)
```

---

**Step 6 — Peak Time**

```
cos(ωt + φ) is maximal at  ωt + φ = 0   →   t_peak = (−φ/ω) mod τ
```

```python
# The modulo folds the result into the first cycle: Peak Time is the peak of
# the first fitted oscillation, in hours after recording start.
peak_time = (-phase_angle_rad / omega) % period_hours                      # (line 140)
```

---

**Step 7 — goodness of fit**

```
SS_res = Σ (yᵢ − ŷᵢ)²        with ŷ = M + β₁cos(ωt) + β₂sin(ωt)
SS_tot = Σ (yᵢ − ȳ)²
R²     = 1 − SS_res / SS_tot
```

```python
# SS_res: what the fitted cosine failed to explain ...
ss_res = np.sum((data_clean - (mesor                                       # (line 150)
                               + beta_cos * np.cos(omega * time_clean)
                               + beta_sin * np.sin(omega * time_clean))) ** 2)
# ... SS_tot: what a flat line at the mean would leave unexplained.
ss_tot = np.sum((data_clean - np.mean(data_clean)) ** 2)                   # (line 161)
r_squared = 1 - (ss_res / ss_tot)                                          # (line 162)
```

R² is therefore the share of variance the cosine captures *relative to the
flat-line model* — the same denominator for every tested period, which is what
makes R² comparable across the multi-period scan.

---

**Step 8 — is the rhythm significant?**

```
df_model = k = 2                         cos and sin
df_res   = n − k − 1 = n − 3             β₀ costs one degree of freedom too
MS_model = (SS_tot − SS_res) / df_model
MS_res   = SS_res / df_res
F        = MS_model / MS_res             ~ F(2, n−3) under H₀
p        = 1 − F_cdf(F; 2, n−3)
```

```python
n = len(data_clean)
k = 2                                       # cos + sin — the rhythm terms  (line 166)
df_model = k                                # dfn = 2                       (line 167)
df_residual = n - k - 1                     # dfd = n − 3                   (line 168)

mse_residual = ss_res / df_residual         # MS_res                        (line 171)
mse_model = (ss_tot - ss_res) / df_model    # MS_model                      (line 172)
f_statistic = mse_model / mse_residual      # F                             (line 173)
p_value = 1 - stats.f.cdf(f_statistic, df_model, df_residual)              # (line 174)
```

H₀ is "β₁ = β₂ = 0", i.e. a flat line explains the data as well as the cosine.
With thousands of frames *n* is large, `df_residual` follows, and *p* becomes
tiny even for weak rhythms — which is why R² and Amplitude carry the biological
statement, not *p*.

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

**Minimum: 3 cycles (= 3 days for a 24 h rhythm)** — the same as the Chi² periodogram. More cycles are not required; they simply tighten the confidence intervals and raise R² (see the table above).

### p-values in Cosinor

**Individual fit**: significance is assessed by an F(2, n−3) test (two sine/cosine
parameters vs. n observations).

**Population cosinor**: uses the Nelson et al. (1979) F-test — F(dfn=2, dfd=2(n−1)) —
which tests whether the mean β_cos and mean β_sin across all ROIs are simultaneously
zero (H₀: no population-level rhythm). The individual β coefficients feed the
numerator and denominator of this F-statistic.

**Multi-period scan**: tests a fine-grained grid (~20 steps across the period range,
with ≥ 1 h resolution) and selects the period with the best R² among those that pass
the individual F-test.

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

## Batch Datasets (pooling several recordings)

The **Batch Datasets** group at the top of the Extended Analysis tab pools the ROIs of
several saved analysis HDF5 files into one population, so every method runs on
n = ROIs × datasets instead of a single recording's worth.

### Workflow

1. **Add Dataset…** — pick one or more saved results files. The first row is the
   **main dataset**: its analysis parameters, ROI masks and time base are used for
   the whole batch.
2. Set the **ZT reference** for each additional dataset (see below).
3. **Load / Reload Pooled Data** — re-reads every file and pools the ROIs.
4. Run the rhythmic pattern analysis as usual.

A single row behaves exactly like the old "Load Results from HDF5" button — same
results, same ROI labels, same colours.

### ROI naming and colours

| Dataset | ROI keys | Labels | Colour |
|---------|----------|--------|--------|
| 1 (main) | 1, 2, 3, … | `ROI 1`, `ROI 2`, … | cycle position 1, 2, 3, … |
| 2 | 2001, 2002, … | `ROI1_2`, `ROI2_2`, … | same as `ROI 1`, `ROI 2`, … |
| 3 | 3001, 3002, … | `ROI1_3`, `ROI2_3`, … | same as `ROI 1`, `ROI 2`, … |

Colour encodes the **ROI number**, the suffix encodes the **dataset**. `ROI1_1`,
`ROI1_2` and `ROI1_3` therefore share one colour, and individual traces are drawn
with a per-dataset linestyle (solid / dashed / dotted) so they stay distinguishable.

CSV and Excel exports use `ROI_1`, `ROI_1_2`, … as column names and gain a
**BATCH DATASETS** and **ROI PROVENANCE** block mapping every pooled ROI back to its
source file and ZT shift.

### ZT reference

Each additional dataset is aligned by one of two modes:

**`ZT0 = own recording start`** (default)
Each recording's own first sample is ZT0, so all datasets share a time origin of
"hours since recording start". This is the correct choice when every recording was
started at the same point in the light cycle — the usual protocol of starting at
lights-on. It is what makes pooled cosinor acrophases comparable: peak time is
measured from the series' time origin, so a common origin means a common phase
reference.

**`ZT relative to main dataset`**
The dataset is shifted by an explicit offset (hours) so its first sample sits at
that ZT on the main dataset's clock. Use this when a recording started at a
different light-cycle phase — e.g. dataset 2 started 8 h later, so it is entered as
`ZT +8.00 h` and its samples line up with dataset 1's ZT8 onward.

> Declaring a recording ZT0 when it actually started at an arbitrary time of day
> smears the pooled acrophase and flattens the population mean by exactly the
> difference in start times. When in doubt, enter the real offset.

Modes can be mixed within one batch. The analysis output states the alignment
actually used for every dataset.

### What the mode does and does not affect

| Unaffected by the mode | Affected by the mode |
|------------------------|----------------------|
| Period estimates (Chi², FFT, Cosinor) | Cosinor acrophase / peak time |
| Amplitude, MESOR, R² | Population mean trace shape |
| Rhythmicity p-values | Coherence and similarity overlap |

Period and amplitude are computed per ROI in the frequency domain and do not
depend on the time origin, so **n improves for those regardless of the mode chosen**.

### Pairwise methods on pooled data

Similarity and Coherence align their two inputs by sample index, which equals time
alignment only inside one recording. On pooled data every ROI is first resampled
onto one shared absolute time grid, and each pair is then computed on the samples
both recordings actually cover. Pairs with too little overlap are skipped rather
than reported from a handful of points:

- **Similarity**: needs ≥ 2 × `max_lag_hours` of overlap
- **Coherence**: needs ≥ 2 × target period of overlap

Skipped pairs appear blank in the matrix and are counted in the summary text.

### Reading the pooled population panel

- The **Significant: k/n** box gains a per-dataset breakdown (`D1: 8/12`, `D2: 6/12`)
  so a result carried by one dataset alone is visible rather than hidden in the pooled n.
- When the recordings differ in length or ZT alignment, the number of contributing
  ROIs varies over time. The population mean plot then shows an **n(t)** step trace on
  a right-hand axis, and the mean is labelled with its range (`Mean (n=8–16)`).
  Treat thinly-supported stretches of the mean with the same caution as a small sample.

---

## Temperature Control

Temperature is itself a Zeitgeber, so a reviewer will ask whether the rhythm you
report is simply the animals tracking a temperature cycle in the rig. The
**Temperature Control** group in the Extended Analysis tab answers that question.

### Why correlation alone is not enough

If the incubator temperature happens to drift on a 24 h cycle, activity and
temperature correlate strongly **whether or not temperature drives the behaviour**
— both are 24 h periodic. A low correlation is not conclusive either, since a
temperature response may be lagged or nonlinear. The panel therefore runs four
tests, in increasing order of the weight they carry:

| # | Panel | What a clean result looks like |
|---|------|-------------------------------|
| 1 | Temperature variation | Small SD and range (e.g. ±0.05 °C, range 0.35 °C) |
| 2 | Is the temperature itself rhythmic? | No significant component in the period band |
| 3 | Correlation vs lag | Low r at lag 0, little variance explained |
| 4 | Rhythm after regressing temperature out | Rhythm survives, period unchanged |
| 5 | Q10 ceiling | Observed rhythm far exceeds what ΔT allows |
| 6 | Rhythms differ between individuals | Scattered amplitudes **and** phases |

**Test 2 decides which of the others can answer your question.** If temperature
carries no significant rhythm in the analysed band, the causal story fails
immediately and test 4 confirms it directly. If temperature *is* rhythmic at the
same period, tests 4 becomes uninterpretable and **tests 5 and 6 carry the
argument** — see the confounded case below.

**Test 4** regresses each ROI's activity on temperature at its optimal lag
(`activity = a + b × temperature(t − lag)`) and re-runs the periodogram on the
residual. A surviving peak at essentially the same period means temperature does
not explain the rhythm — *provided the two are not collinear*.

**Test 5, the Q10 ceiling,** is a physiological limit rather than a statistical
test, which is why it still works under collinearity. Metabolic rate scales as
`Q10^(ΔT/10)`, so the measured temperature swing sets a hard upper bound on the
modulation it could cause. A rhythm several-fold larger than that bound cannot be
thermally driven, whatever the correlation says.

**Test 6** exploits a fact the other tests ignore: every ROI in a dish sees the
*identical* temperature trace, so its between-individual variance is exactly zero.
A common driver predicts near-identical rhythms. Differing sensitivity could
scatter amplitudes, but a linear response cannot shift phase — scattered
acrophases argue for independent internal clocks.

### Reading the correlation panel

Panel 3 plots r as a function of lag rather than a single "strongest
correlation" number, and that is deliberate. **Both signals are periodic, so a
large |r| exists at some lag by construction** — near half a period it is large
and negative. Ranking by |r| over a full cycle therefore manufactures a dramatic
number that carries no information about causation.

What is interpretable:

- the **dot at lag 0** — the genuine concurrent correlation, with its sign
- the **shaded window** (0 to *Response window*, temperature leading) — the only
  direction in which temperature could drive behaviour
- the **oscillation of the curve itself** — visible proof that a big |r| further
  out is an artefact of periodicity, not a finding

### The confounded case — read this before quoting a negative result

If your temperature **is** rhythmic at the same period as the activity, the two
are collinear at that frequency and **no regression can separate them**. Removing
temperature also removes a genuine endogenous rhythm, so "rhythm lost" would be a
false negative.

The analysis detects this and refuses to conclude: the report prints
`⚠ NOT CONCLUSIVE`, per-ROI rows are marked `n/c`, and panel 4 of the figure is
outlined in red with the caveat in its title. It will never claim a clean result
in this situation.

When that happens, what actually discriminates, in order of strength:

1. **Amplitude argument** — from test 1. A swing of a few tenths of a degree
   cannot produce a large behavioural rhythm; a Q10 of 2–3 yields only a few
   percent metabolic change.
2. **Free-run under constant temperature** — the rhythm persisting with τ ≠ 24 h
   is decisive.
3. **Phase relationship** — from test 3. A rhythm that peaks at a phase the
   temperature cycle cannot explain, or that *leads* temperature, argues against a
   temperature cause. A positive lag means temperature leads activity, the
   direction expected if temperature were driving behaviour.
4. **Fix the rig** so temperature no longer cycles, and re-record.

### Where the temperature comes from

Temperature is read from `timeseries/temperature` (or `temperature_celsius`) in
the **raw recording** — the same source the Telemetry tab uses. Resolution order:

1. The results file, if it was saved after temperature storage was added.
2. Otherwise, the loaded raw recording.
3. Otherwise, the analysis reports what is missing and how to fix it.

Results files saved from now on include the temperature record, so the analysis
works from a results file alone. **Re-save older results files** to make them
self-contained — this matters for batch mode, where each pooled dataset needs its
own temperature record and only the main dataset's raw file is typically loaded.

### With pooled datasets

Every ROI is tested against **its own dataset's** temperature record, shifted onto
the pooled time base. Recordings made in different incubators or at different
times are therefore handled correctly rather than all being compared to dataset 1.

### Settings

| Setting | Meaning |
|---------|---------|
| **Rhythm test** | Periodogram used for all three rhythm tests (activity, temperature, residual). Chi² is the robust default. |
| **Response window** | Upper end of the physiologically plausible response delay. Only lags from 0 to this value are searched, with temperature *leading*. Default 6 h — deliberately less than a full period, for the reason given above. |
| **Q10 (max)** | Temperature coefficient for the amplitude ceiling. The figure reports 2.0, 2.5 and this value and quotes the most generous, so a **higher** Q10 is the conservative choice. Default 3.0. |

Period range and significance level are taken from the **Period Range Parameters**
below, so the control test uses exactly the settings your main analysis used.

---

## Data Sources

### Fraction Movement

- Computed as: `active time (s) / bin size (s)` per time bin
- Time-based (not frame-based) → robust to irregular frame intervals
- Continuous signal in [0, 1]
- Threshold-dependent: the hysteresis threshold determines what counts as "active"
- **Standard for circadian biology** (comparable to running wheel activity counts)
- Recommended for: Chi² periodogram, Cosinor, actogram visualization

### Raw Intensity

- Per-frame pixel change values (frame differences), MinMax-normalized per ROI
- Continuous signal, preserves amplitude information
- Independent of movement threshold
- Optional alternative when you want to preserve sub-threshold amplitude
  (e.g. very weak rhythms). The analyses in this project use Fraction
  Movement throughout, including Cosinor.

### When to use which

| Analysis | Recommended source | Reason |
|----------|-------------------|--------|
| Chi² Periodogram | Fraction Movement | Literature standard, robust |
| Cosinor | Fraction Movement | Continuous [0,1] activity; consistent with the other analyses |
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
| Cosinor fit | 3 cycles (3 days for τ=24h) | 7–14 days |
| Entrainment verification | 5 days LD + 5 days DD | 7 days LD + 7 days DD |

### Period range

- Set wider than expected: 16–36 h covers circadian rhythms safely
- Check for boundary warnings (⚠️) — if the peak is at the boundary, extend the range
- Extending the period range does not affect Z-scores

### Time range

- Use Full Recording by default
- Use segment analysis only when recording is long enough to split into
  transient and stable phases (≥ 7 days total)
- Minimum window for reliable analysis: 3 × expected period

### Environmental control

- **Temperature**: constant ± 0.5 °C (temperature is itself a Zeitgeber).
  Verify it with [Temperature Control](#temperature-control) rather than assuming
  it — and note that a temperature that *does* cycle at 24 h makes the residual
  test inconclusive, so it is worth fixing at the rig rather than in analysis.
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
