# Extended Analysis Documentation

## Table of Contents
1. [Overview](#overview)
2. [Fisher Z-Transformation Periodogram](#fisher-z-transformation-periodogram)
3. [FFT Power Spectrum](#fft-power-spectrum)
4. [ROI Similarity Matrix](#roi-similarity-matrix)
5. [Coherence Analysis](#coherence-analysis)
6. [Phase Clustering](#phase-clustering)
7. [Interpreting Results](#interpreting-results)
8. [Export Functionality](#export-functionality)
9. [Color Consistency](#color-consistency)
10. [Best Practices](#best-practices)

---

## Overview

The Extended Analysis tab provides five complementary methods for analyzing rhythmic patterns and synchronization in activity data. These methods are designed to detect circadian rhythms, ultradian cycles, and behavioral coordination across multiple ROIs.

### Motivation and Scientific Rationale

**Why Extended Analysis?**

Traditional movement analysis (tracking when animals move) answers the question "how much" but misses the critical question of "when" - the temporal organization of behavior. Many biological processes are inherently rhythmic:

1. **Circadian Clocks are Fundamental**
   - Nearly all organisms have internal ~24-hour clocks
   - Disrupted rhythms indicate disease, stress, or aging
   - Drug effects often manifest as rhythm changes before gross behavioral changes
   - Basic movement metrics (total activity) can be identical between rhythmic and arrhythmic animals

2. **Social Coordination Requires Temporal Analysis**
   - Two animals with identical total activity may be completely synchronized or completely independent
   - Dominance hierarchies manifest as sequential (lag-based) activity patterns
   - Competition appears as anti-phase relationships (one active while other rests)
   - Simple correlation misses these timing-dependent relationships

3. **Multiple Methods Provide Complementary Information**
   - No single method captures all aspects of rhythmic behavior
   - Statistical validation (Fisher) confirms what spectral analysis (FFT) reveals
   - Synchronization (Similarity, Coherence) explains relationships between rhythms
   - Phase analysis (Phase Clustering) quantifies precise timing and clock strength
   - Cross-validation between methods ensures robust, reproducible findings

**Real-World Example:**

Consider two experimental groups with identical mean activity levels (30% movement):

**Group A (Rhythmic):**
- Strong 24h circadian rhythm (Fisher: p < 0.0001)
- All animals synchronized (Similarity: r > 0.9)
- Robust circadian clock (Phase Clustering: high amplitude)
- **Interpretation**: Healthy, entrained animals with intact circadian systems

**Group B (Arrhythmic):**
- No significant rhythms (Fisher: p > 0.5)
- No synchronization (Similarity: r < 0.3)
- Weak/absent circadian clock (Phase Clustering: low amplitude)
- **Interpretation**: Disrupted circadian system (disease model, SCN lesion, or environmental stress)

**Standard movement analysis cannot distinguish these groups** - both show 30% activity. Extended Analysis reveals the critical difference: temporal organization.

### When to Use Extended Analysis

- **Circadian Research**: Detect 24-hour activity cycles and assess circadian clock function
- **Ultradian Rhythms**: Identify shorter cycles (e.g., feeding patterns, 3-12 hour rhythms)
- **Social Behavior**: Analyze synchronization, dominance hierarchies, and social coordination
- **Sleep/Wake Patterns**: Characterize activity phase relationships and sleep architecture
- **Drug Effects**: Compare rhythmic patterns before/after treatment (rhythm changes precede behavior changes)
- **Disease Models**: Assess circadian disruption in neurological disorders, aging, metabolic syndrome
- **Environmental Studies**: Measure entrainment to light/dark cycles, temperature, feeding schedules
- **Chronobiology**: Investigate free-running periods, phase shifts, and zeitgeber effects

### Analysis Methods Summary

| Method | Purpose | Best For | Output |
|--------|---------|----------|--------|
| Fisher Z-Transformation | Statistical period detection | Confirming significant rhythms | Z-scores, p-values |
| FFT Power Spectrum | Frequency-domain analysis | Identifying all periodic components | Power spectrum, dominant peaks |
| ROI Similarity | Cross-correlation analysis | Finding synchronized ROIs | Correlation matrix, clusters |
| Coherence Analysis | Frequency-specific synchronization | Identifying shared rhythms | Coherence heatmap |
| Phase Clustering | Timing relationships | Detecting activity phases | Phase plot, timing offsets |

### Method Comparison: Strengths and Limitations

| Feature | Fisher Z | FFT | ROI Similarity | Coherence | Phase Clustering |
|---------|----------|-----|----------------|-----------|------------------|
| **Primary Output** | p-values, significance | Power spectrum | Correlation matrix | Coherence values | Phase & amplitude |
| **Statistical Testing** | ✅ Yes (p-values) | ❌ No | ⚠️ Partial (correlation) | ❌ No | ❌ No |
| **Exploratory Analysis** | ⚠️ Limited | ✅ Excellent | ✅ Good | ⚠️ Moderate | ❌ Poor |
| **Computational Speed** | ⚠️ Slow | ✅ Very fast | ⚠️ Moderate | ⚠️ Slow | ✅ Fast |
| **Data Requirements** | ≥3 cycles | ≥2 cycles | ≥2 cycles | ≥5 segments | ≥2 cycles |
| **Finds Periods** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Phase Information** | ❌ No | ❌ No | ✅ Yes (via lag) | ❌ No | ✅ Yes |
| **Handles Mixed Periods** | ✅ Yes | ✅ Yes | ❌ Poor | ⚠️ Moderate | ❌ No |
| **Multiple Rhythms** | ⚠️ Dominant only | ✅ All shown | ❌ N/A | ✅ All shown | ❌ Single period |
| **Noise Robustness** | ✅ Good | ⚠️ Moderate | ⚠️ Moderate | ✅ Good | ⚠️ Moderate |
| **Interpretation** | ✅ Clear (p-value) | ⚠️ Subjective | ✅ Intuitive | ⚠️ Complex | ✅ Visual |

### Quick Decision Guide

**Choose Fisher Z-Transformation when:**
- You need statistical significance testing (p-values for publications)
- Confirming expected rhythms (hypothesis-driven research)
- Comparing rhythm strength across experimental conditions
- Data quality is moderate (method is noise-robust)

**Choose FFT Power Spectrum when:**
- Exploring unknown rhythms (no prior period expectation)
- Need to see all frequency components at once
- Fast screening of many ROIs required
- Identifying harmonics and secondary periods

**Choose ROI Similarity when:**
- Finding which animals are synchronized
- Detecting social groups or behavioral clusters
- Identifying phase-shifted (anti-phase) relationships
- All ROIs have similar periods

**Choose Coherence Analysis when:**
- Need frequency-specific synchronization measure
- Validating similarity matrix findings
- ROIs share rhythm at specific frequency but differ otherwise
- Detecting harmonic coupling

**Choose Phase Clustering when:**
- Need precise timing of activity peaks
- Measuring circadian clock strength (not just presence)
- Visualizing phase relationships for publication
- All ROIs confirmed to share same period (from Fisher/FFT)

### Recommended Workflow

**Standard Analysis Pipeline:**
1. **Fisher Z or FFT** → Detect periods and confirm rhythms exist
2. **Cross-validate** → Both methods should agree on period (±1 hour)
3. **ROI Similarity** → Identify synchronized groups
4. **Coherence** (optional) → Verify frequency-specific synchronization
5. **Phase Clustering** → Quantify timing relationships using confirmed period

**Exploratory Pipeline:**
1. **FFT** → Quick scan for any rhythms
2. **Fisher Z** → Statistical confirmation of FFT findings
3. **ROI Similarity** → Look for clusters and relationships
4. **Phase Clustering** → Detailed timing analysis

**Troubleshooting Pipeline:**
- If Fisher finds rhythm but FFT doesn't → Check data quality, harmonics
- If Similarity shows grouping but Coherence doesn't → Different rhythm components
- If Phase Clustering shows odd results → Verify period consistency first

---

## Fisher Z-Transformation Periodogram

### What It Does

Fisher's Z-transformation is a statistical method for detecting periodic patterns in time-series data. It tests whether activity follows a rhythmic pattern by correlating the signal with sine and cosine waves at different frequencies.

### How It Works

1. **Correlation Analysis**: For each test period (e.g., 12h, 18h, 24h), the method calculates how well the activity data correlates with sine and cosine waves of that period
2. **Z-Score Calculation**: Converts correlation strength into a statistical Z-score
3. **Significance Testing**: Uses chi-square distribution (df=2) to determine if rhythms are statistically significant
4. **Peak Detection**: Identifies the dominant period (strongest rhythm)

### Mathematical Background

For a given test period T, the Z-score is calculated as:

```
Z = N × (r_cos² + r_sin²)
```

Where:
- N = number of samples
- r_cos = correlation with cosine wave
- r_sin = correlation with sine wave

The squared coherence (r_cos² + r_sin²) represents the "power" of that periodic component.

### Parameters

- **Minimum Period**: Shortest cycle to test (default: 12.0 hours)
- **Maximum Period**: Longest cycle to test (default: 36.0 hours)
- **Significance Level**: Statistical threshold (default: 0.05 = 95% confidence)
- **Bin Size**: Optional data averaging (seconds)

### Output Interpretation

#### Z-Score Plot
- **Blue curve**: Z-scores across all tested periods
- **Gray dashed line**: Significance threshold (p < 0.05)
- **Colored vertical line**: Dominant period (if significant)
- **Colored marker**: Peak Z-score value

#### Statistical Metrics
- **Dominant Period**: Period with highest Z-score (hours)
- **Z-Score**: Strength of rhythm (higher = stronger)
- **p-value**: Statistical significance (p < 0.05 = significant)
- **Critical Z**: Threshold for significance (typically ~6.0 for p=0.05)

#### What Values Mean

| Z-Score | p-value | Interpretation |
|---------|---------|----------------|
| > 10 | < 0.01 | Very strong, highly significant rhythm |
| 6-10 | 0.01-0.05 | Significant rhythm |
| 3-6 | 0.05-0.22 | Weak, possibly significant |
| < 3 | > 0.22 | No significant rhythm |

### Example Results

```
ROI 1:
  ✓ Significant circadian rhythm detected (p=0.0001)
  Dominant period: 24.12 hours
  Z-score: 125.45
```

**Interpretation**: This ROI shows a very strong 24-hour rhythm with extremely high statistical confidence.

### Advantages

✅ **Statistical Rigor**
- Provides p-values for objective significance testing
- Chi-square distribution (df=2) is well-established
- Clear yes/no answer: rhythm is significant or not

✅ **Hypothesis Testing**
- Specifically tests for periodicity at each frequency
- Ideal for confirming expected rhythms (e.g., "is there a 24h rhythm?")
- Robust against noise with sufficient data

✅ **Interpretability**
- Z-scores are intuitive (higher = stronger rhythm)
- Direct biological interpretation
- Easy to report in publications

✅ **Sensitivity**
- Can detect weak rhythms with sufficient data
- Works well even with moderate noise levels
- Effective for irregular waveforms (non-sinusoidal)

### Limitations

⚠️ **Computational Cost**
- Slower than FFT for large datasets
- Tests each period individually (100 periods = 100 tests)
- Multiple comparison problem (need correction for many tests)

⚠️ **Data Requirements**
- Requires ≥3 complete cycles for statistical power
- Short recordings may not reach significance
- Minimum ~10 samples needed per test

⚠️ **Resolution**
- Period resolution depends on tested range and number of test periods
- Cannot provide finer resolution than ~1% of period
- May miss periods outside predefined range

⚠️ **Assumptions**
- Assumes stationary rhythm (consistent throughout recording)
- Cannot detect transient or changing rhythms
- Sensitive to long-term trends (requires detrending)

⚠️ **Multiple Rhythms**
- Reports only dominant period
- May miss secondary rhythms
- Harmonics can complicate interpretation

### When to Use

**Best For:**
- Confirming hypothesized rhythms (e.g., circadian studies)
- Publication-quality statistical validation
- Comparing rhythm strength across conditions
- Detecting rhythms in noisy data

**Not Ideal For:**
- Exploratory analysis (use FFT instead)
- Very short datasets (< 3 cycles)
- Rapidly changing rhythms
- Identifying all frequency components simultaneously

### Best Practices

1. **Data Duration**: Need at least 3 complete cycles for reliable detection
   - For 24h rhythms: minimum 72 hours of data
   - For 12h rhythms: minimum 36 hours of data

2. **Period Range**: Set based on expected rhythms
   - Circadian: 20-28 hours
   - Ultradian: 2-12 hours
   - Custom ranges for specific research questions

3. **Binning**: Use for high-resolution data
   - Raw data at 5-second intervals → bin to 60 seconds
   - Reduces noise while preserving rhythmic patterns

4. **Multiple Comparisons**: Adjust significance threshold when testing many periods
   - Bonferroni correction: p_threshold = 0.05 / n_tests
   - Or use False Discovery Rate (FDR) control

---

## FFT Power Spectrum

### What It Does

Fast Fourier Transform (FFT) converts time-series data into the frequency domain, revealing all periodic components simultaneously. This is complementary to Fisher Z-transformation and provides standard spectral analysis.

### How It Works

1. **Detrending**: Removes linear trends from data
2. **Windowing**: Applies Hann window to reduce spectral leakage
3. **Zero-Padding**: Pads signal 4× for better frequency resolution
4. **FFT Computation**: Calculates power spectrum
5. **Peak Detection**: Identifies significant frequency peaks

### Mathematical Background

The power spectrum is calculated as:

```
Power(f) = |FFT(signal)|²
```

Where:
- FFT transforms time-domain signal to frequency domain
- Power represents the strength of oscillation at each frequency
- Dominant period = 1 / frequency with maximum power

### Parameters

- **Minimum Period**: Shortest cycle to analyze (hours)
- **Maximum Period**: Longest cycle to analyze (hours)
- **Window Function**: Type of windowing (default: "hann")
  - "hann": Good general-purpose window
  - "hamming": Similar to Hann, slightly different sidelobe properties
  - "blackman": Better frequency resolution, lower spectral leakage
- **Bin Size**: Optional data averaging

### Output Interpretation

#### Power Spectrum Plot
- **Colored curve**: Power across all frequencies
- **Colored vertical line**: Dominant period
- **Colored marker**: Peak power value

#### Spectral Metrics
- **Dominant Period**: Period with maximum power (hours)
- **Dominant Power**: Strength at dominant frequency
- **Frequency Peaks**: List of all significant peaks (period, power)

#### Power Values

Power is in arbitrary units (AU²). Relative values matter more than absolute values:

| Relative Power | Interpretation |
|----------------|----------------|
| 10× higher than others | Very strong dominant rhythm |
| 2-5× higher | Clear rhythm |
| Similar to others | Weak or multiple competing rhythms |

### Comparison with Fisher Z-Transformation

| Feature | Fisher Z | FFT |
|---------|----------|-----|
| Output | Statistical significance | Power spectrum |
| Strength | Tests specific hypotheses | Explores all frequencies |
| Interpretation | p-values, clear yes/no | Relative power levels |
| Use Case | Confirm expected rhythms | Discover unknown rhythms |

**Agreement**: Both should identify the same dominant period. Typical differences:
- Fisher: 24.0h
- FFT: 23.7-24.3h (slightly different due to frequency resolution)

Differences < 1 hour indicate excellent agreement.

### Advantages

✅ **Computational Efficiency**
- Very fast (O(N log N) algorithm)
- Analyzes all frequencies simultaneously
- Scales well to large datasets

✅ **Exploratory Power**
- Reveals all periodic components at once
- No need to specify expected periods in advance
- Identifies harmonics and secondary rhythms automatically

✅ **Standard Method**
- Widely used in signal processing
- Extensive literature and validation
- Compatible with other spectral analysis tools

✅ **Visual Clarity**
- Power spectrum shows full frequency content
- Easy to identify dominant peaks
- Reveals complex rhythmic structures

✅ **Resolution**
- Zero-padding provides excellent frequency resolution
- Can distinguish closely-spaced periods
- Continuous spectrum (not limited to discrete test periods)

### Limitations

⚠️ **No Statistical Testing**
- Power values are in arbitrary units
- No p-values or significance thresholds
- Subjective interpretation of "strong" vs "weak" peaks

⚠️ **Spectral Leakage**
- Non-integer number of cycles causes frequency spreading
- Windowing reduces but doesn't eliminate leakage
- Can create artificial side lobes in spectrum

⚠️ **Harmonics Confusion**
- Strong fundamental generates harmonics (2×, 3× frequency)
- Harmonics can be mistaken for independent rhythms
- Example: 24h rhythm creates peaks at 12h, 8h, 6h

⚠️ **Amplitude Sensitivity**
- Assumes sinusoidal oscillations
- Non-sinusoidal waveforms spread power across frequencies
- Square waves or sharp transitions create many harmonics

⚠️ **Detrending Required**
- Very sensitive to linear trends
- DC component (zero frequency) can dominate spectrum
- Long-term drift must be removed

⚠️ **Edge Effects**
- Beginning and end of recording affect spectrum
- Windowing reduces but impacts amplitude estimates
- Very short recordings have poor frequency resolution

### When to Use

**Best For:**
- Exploratory analysis (unknown rhythms)
- Identifying multiple periodic components
- Comparing spectral signatures across conditions
- Fast screening of many ROIs
- Detecting harmonics and complex rhythms

**Not Ideal For:**
- Statistical hypothesis testing
- Definitive yes/no rhythm detection
- Very noisy data requiring significance testing
- When absolute power values needed

### Best Practices

1. **Window Selection**:
   - Hann: Default, good for most cases
   - Blackman: Use for noisy data (better spectral leakage suppression)
   - Hamming: Similar to Hann, slightly different sidelobe properties
   - None: Only for very clean periodic signals with integer cycles

2. **Zero-Padding**: Automatically applied (4× padding)
   - Improves frequency resolution without adding information
   - Provides smooth interpolation between frequency bins
   - Does not increase statistical power

3. **Interpretation**:
   - Look for clear peaks above background noise floor
   - Multiple peaks may indicate harmonics (e.g., 24h and 12h, 8h, 6h)
   - Broad peaks suggest irregular or variable-period rhythms
   - Check if secondary peaks are harmonics (integer divisors of dominant)

4. **Validation**:
   - Always cross-validate with Fisher Z-transformation
   - Both methods should agree on dominant period (±1 hour)
   - Use Fisher for statistical confirmation of FFT findings

---

## ROI Similarity Matrix

### What It Does

Computes pairwise cross-correlations between all ROIs to identify synchronized or anti-phase activity patterns. Uses hierarchical clustering to group similar ROIs.

### How It Works

1. **Normalization**: Standardizes each ROI's activity (mean=0, std=1)
2. **Cross-Correlation**: Calculates correlation at different time lags
3. **Max Correlation**: Finds maximum correlation (can be at non-zero lag)
4. **Clustering**: Groups ROIs by similarity using hierarchical clustering
5. **Visualization**: Displays correlation matrix and dendrogram

### Mathematical Background

Cross-correlation at lag τ:

```
r(τ) = Σ[x(t) × y(t+τ)] / √[Σx(t)² × Σy(t+τ)²]
```

The maximum correlation captures both:
- **In-phase** synchronization (lag = 0)
- **Anti-phase** relationships (lag = ½ period)
- **Phase-shifted** patterns (other lags)

### Parameters

- **Maximum Lag**: Maximum time shift to test (default: 12 hours)

### Output Interpretation

#### Correlation Matrix Heatmap
- **Green**: High positive correlation (synchronized)
- **Yellow**: Low correlation (independent)
- **Red**: Negative correlation (anti-phase)

| Correlation | Lag | Interpretation |
|-------------|-----|----------------|
| > 0.9 | 0 | Perfect synchronization |
| 0.7-0.9 | 0 | Strong synchronization |
| 0.5-0.7 | 0 | Moderate synchronization |
| 0.7-0.9 | ½ period | Anti-phase relationship |
| < 0.3 | any | Independent/unrelated |

#### Dendrogram (Hierarchical Clustering)
- **Height**: Dissimilarity (1 - correlation)
- **Branches**: ROIs that cluster together
- **Colors**: Different clusters (automatically detected)
- **Red dashed line**: Cluster threshold (30% of max distance)

#### Similarity Table
For each ROI pair:
- **Correlation**: Strength of relationship (-1 to +1)
- **Lag (hours)**: Time offset for maximum correlation
- **Similarity**: Percentage (0-100%)

### Example Results

```
High Similarity Pairs:
  ROI 1 ↔ ROI 2: r=0.98, lag=0.0h (98% similar, synchronized)
  ROI 3 ↔ ROI 4: r=0.95, lag=0.1h (95% similar, nearly synchronized)

Moderate Similarity:
  ROI 1 ↔ ROI 3: r=0.82, lag=12.0h (anti-phase, same period)

Low Similarity:
  ROI 1 ↔ ROI 5: r=0.35, lag=2.1h (independent rhythms)
```

### Clustering Interpretation

**Group A (ROIs 1-2)**: Synchronized 24h rhythm
**Group B (ROIs 3-4)**: Synchronized 24h rhythm, anti-phase to Group A
**Group C (ROIs 5-6)**: Different period (20h rhythm)

### Advantages

✅ **Lag Detection**
- Automatically finds optimal time offset for maximum correlation
- Detects anti-phase relationships (phase-shifted rhythms)
- Reveals sequential behaviors (e.g., dominance hierarchies)

✅ **Visual Clarity**
- Heatmap provides immediate overview of all pairwise relationships
- Dendrogram shows natural groupings
- Easy to identify synchronized clusters

✅ **Normalization**
- Correlation coefficient (-1 to +1) is standardized
- Independent of absolute activity levels
- Allows fair comparison between ROIs with different amplitudes

✅ **Clustering**
- Hierarchical clustering automatically groups similar ROIs
- Objective grouping based on similarity threshold
- Reveals social structure and behavioral coordination

✅ **Comprehensive**
- Analyzes all possible pairs simultaneously
- Single analysis provides complete synchronization picture
- Useful for exploratory analysis

### Limitations

⚠️ **Period Assumption**
- Assumes all ROIs have similar period lengths
- Cannot distinguish "same period, different phase" from "different periods"
- Mixed periods (e.g., 20h and 24h) show artificially low correlation

⚠️ **Computational Cost**
- O(N²) pairs for N ROIs (6 ROIs = 15 pairs, 20 ROIs = 190 pairs)
- Cross-correlation at each lag is computationally intensive
- Large datasets with many ROIs can be slow

⚠️ **Lag Interpretation**
- Lag values can be ambiguous (e.g., +6h vs -18h in 24h cycle)
- Maximum lag parameter affects results
- Anti-phase may appear as high positive correlation at large lag

⚠️ **Stationarity Requirement**
- Assumes consistent relationship throughout recording
- Transient synchronization events are averaged out
- Cannot detect changes in coordination over time

⚠️ **No Frequency Information**
- Overall correlation doesn't specify which frequencies are synchronized
- High correlation could be due to any shared frequency component
- Cannot distinguish circadian vs ultradian synchronization

⚠️ **Sensitivity to Noise**
- Random noise reduces correlation coefficients
- May fail to detect weak synchronization
- Short recordings amplify noise effects

### When to Use

**Best For:**
- Identifying synchronized groups (social clusters)
- Detecting phase relationships (in-phase vs anti-phase)
- Comparing overall behavioral similarity
- Finding dominant hierarchies (lag-based sequences)
- Exploratory analysis of social structure

**Not Ideal For:**
- Mixed-period datasets (different rhythm frequencies)
- Frequency-specific synchronization (use Coherence instead)
- Very short recordings (< 2 cycles)
- When timing precision is critical
- Transient or changing synchronization

### Best Practices

1. **Lag Selection**:
   - Set to ½ of maximum expected period
   - For 24h rhythms: 12-hour lag captures anti-phase
   - Too small: May miss anti-phase relationships
   - Too large: Increases computation time

2. **Cluster Interpretation**:
   - Tight clusters (height < 0.3): Very similar behavior
   - Loose clusters (height > 0.7): Weakly related
   - Isolated branches: Unique patterns
   - Compare with domain knowledge (e.g., known social groups)

3. **Biological Meaning**:
   - High correlation + zero lag → Social synchronization, shared zeitgeber
   - High correlation + non-zero lag → Sequential behavior, dominance
   - Anti-phase (r < 0 or lag ≈ ½ period) → Competition, resource partitioning
   - Low correlation → Independent rhythms, different periods, or no coordination

4. **Validation**:
   - Cross-check with Coherence analysis
   - Verify period similarity with Fisher/FFT first
   - Consider biological context (social species vs solitary)

---

## Coherence Analysis

### What It Does

Measures frequency-specific synchronization between ROI pairs using Welch's method. Unlike simple correlation, coherence identifies which frequency components are synchronized.

### How It Works

1. **Welch's Method**: Divides data into overlapping segments
2. **Cross-Spectral Density**: Computes frequency-by-frequency correlation
3. **Coherence Calculation**: Normalizes to 0-1 scale
4. **Dominant Frequency**: Identifies strongest coherent frequency

### Mathematical Background

Coherence at frequency f:

```
Coh(f) = |Pxy(f)|² / [Pxx(f) × Pyy(f)]
```

Where:
- Pxy = cross-spectral density
- Pxx, Pyy = auto-spectral densities
- Coherence ranges from 0 (no synchronization) to 1 (perfect synchronization)

### Parameters

- **Segment Length**: Number of samples per segment (default: 256)
- **Overlap**: Fraction of overlap between segments (default: 0.5)
- **Window**: Window function (default: "hann")

### Output Interpretation

#### Coherence Heatmap
- **Blue**: High coherence (synchronized at specific frequency)
- **Yellow**: Moderate coherence
- **Red**: Low coherence (independent)

| Coherence | Interpretation |
|-----------|----------------|
| 0.8-1.0 | Very strong synchronization at this frequency |
| 0.6-0.8 | Strong synchronization |
| 0.4-0.6 | Moderate synchronization |
| < 0.4 | Weak or no synchronization |

#### Coherence vs Correlation

| Metric | What It Measures | Use Case |
|--------|------------------|----------|
| Correlation | Overall similarity | General synchronization |
| Coherence | Frequency-specific similarity | Identifying shared rhythms |

**Example**: Two ROIs might have low overall correlation but high coherence at 24h frequency, indicating they share circadian rhythm but differ in other ways.

### Common Patterns

1. **High coherence at single frequency**: Shared dominant rhythm
2. **High coherence at multiple harmonics**: Complex rhythmic relationship
   - Example: High at 24h and 12h suggests circadian + ultradian coupling
3. **Broad high coherence**: General behavioral synchronization
4. **Low coherence everywhere**: Independent activity patterns

### Advantages

✅ **Frequency-Specific**
- Identifies which frequency components are synchronized
- Can detect shared circadian rhythm even with different ultradian patterns
- Distinguishes synchronization at multiple frequencies simultaneously

✅ **Robust to Phase Shifts**
- Coherence is phase-invariant (doesn't matter if signals are offset in time)
- Detects synchronization regardless of time lag
- Better than correlation for shifted rhythms

✅ **Statistical Averaging**
- Welch's method averages across segments
- Reduces variance, increases reliability
- More robust to noise than single-window analysis

✅ **Standardized Metric**
- Coherence ranges 0-1 (like correlation)
- Well-established interpretation
- Widely used in neuroscience and signal processing

✅ **Harmonic Detection**
- Reveals coupling at harmonics (2×, 3× fundamental)
- Identifies complex rhythmic relationships
- Detects frequency locking

### Limitations

⚠️ **Requires Periodic Signals**
- Only meaningful for rhythmic data
- Arrhythmic or transient behaviors show artificially low coherence
- Cannot be used for non-oscillatory coordination

⚠️ **Segment Length Trade-off**
- Long segments: Good frequency resolution but poor averaging (noisy estimates)
- Short segments: Good averaging but poor resolution (can't distinguish close frequencies)
- Must be tuned based on data characteristics

⚠️ **No Phase Information**
- Coherence magnitude ignores phase relationships
- Cannot distinguish in-phase from anti-phase synchronization
- Complementary to similarity matrix (which includes lag)

⚠️ **Interpretation Challenges**
- Coherence values are harder to interpret than correlation
- No clear threshold for "significant" coherence
- Requires comparison with null hypothesis or shuffled controls

⚠️ **Stationarity Assumption**
- Assumes consistent synchronization throughout recording
- Time-varying coupling is averaged out
- Cannot detect onset or offset of synchronization

⚠️ **Computational Cost**
- More intensive than simple correlation
- Requires FFT for each segment and pair
- Scales poorly with many ROIs (N² pairs)

⚠️ **Harmonics Complication**
- Strong fundamental generates harmonic coherence
- Can mistake harmonics for independent synchronized rhythms
- Example: 24h synchronization creates coherence at 12h, 8h, 6h

### When to Use

**Best For:**
- Verifying frequency-specific synchronization
- ROIs with same period but different non-rhythmic components
- Validating similarity matrix findings
- Detecting harmonic relationships
- Research requiring phase-independent synchronization measure

**Not Ideal For:**
- Non-rhythmic behaviors
- Phase relationship analysis (use Similarity or Phase Clustering)
- Very short recordings (insufficient segments)
- Exploratory analysis (similarity matrix is more intuitive)
- When timing/lag information is important

### Best Practices

1. **Segment Length**:
   - Longer segments → better frequency resolution, fewer segments
   - Shorter segments → more statistical averaging, poorer resolution
   - Default (256 samples) balances both
   - For circadian analysis with 5-min sampling: 256 samples = ~21 hours

2. **Interpretation**:
   - Focus on coherence at biologically relevant frequencies
   - Harmonics (2×, 3× fundamental) are often artifacts of strong fundamental
   - Compare coherence with similarity matrix for validation
   - High coherence + high correlation = strong synchronized rhythm
   - High coherence + low correlation = synchronized rhythm but different baselines

3. **Validation**:
   - Check if coherence peaks match Fisher/FFT dominant periods
   - Compare coherence values across ROI pairs for consistency
   - Use shuffled/randomized data as null hypothesis control

4. **Limitations Awareness**:
   - Requires stationary signals (consistent behavior over time)
   - Less reliable for very short datasets (< 5 segments)
   - May miss transient synchronization events
   - Cannot replace Phase Clustering for timing analysis

---

## Phase Clustering

### What It Does

Uses Hilbert transform to extract instantaneous phase and amplitude of activity rhythms. Clusters ROIs by their activity timing (phase) to identify synchronized groups.

### How It Works

1. **Bandpass Filtering**: Isolates the dominant frequency
2. **Hilbert Transform**: Extracts instantaneous phase and amplitude
3. **Phase Extraction**: Converts to timing within the cycle
4. **Clustering**: Groups ROIs by phase similarity
5. **Visualization**: Polar plot showing phase relationships

### Mathematical Background

Hilbert transform for signal x(t):

```
H[x(t)] = (1/π) ∫ x(τ)/(t-τ) dτ
```

Analytical signal:

```
z(t) = x(t) + i·H[x(t)] = A(t)·e^(iφ(t))
```

Where:
- A(t) = instantaneous amplitude (strength of rhythm)
- φ(t) = instantaneous phase (timing within cycle)

### Parameters

- **Dominant Period**: Period for filtering (from Fisher or FFT analysis)
- **Bandwidth**: Filter bandwidth (default: ±10% of dominant period)
- **Phase Threshold**: Clustering threshold (default: 45°)

### Output Interpretation

#### Polar Plot

**Radial Axis (Distance from Center)**:
- Represents **rhythmic amplitude** (strength of oscillation)
- NOT total activity level
- Measures how well activity follows a clean periodic pattern

**Angular Position (Angle)**:
- Represents **phase** (timing of peak activity)
- 0° (North) = reference phase
- Angles increase clockwise

**Color**:
- Each ROI has a consistent color matching other plots

#### Amplitude Values

| Amplitude | Interpretation |
|-----------|----------------|
| > 80 | Very strong, regular rhythm |
| 50-80 | Strong rhythm |
| 20-50 | Moderate rhythm |
| < 20 | Weak or irregular rhythm |

**Important**: High amplitude ≠ high activity!
- High activity + irregular → moderate amplitude
- Low activity + regular → can have high amplitude

#### Phase Relationships

| Phase Difference | Interpretation |
|------------------|----------------|
| 0-45° | Synchronized (same phase) |
| 45-135° | Partially offset |
| 135-225° | Anti-phase (opposite) |
| 225-315° | Partially offset (other direction) |

#### Example Results

```
ROI Phase Clusters:

  Early Active: 3 ROIs (synchronized)
    ROI 1: Peak at 1.5h (amplitude: 25.94)
    ROI 2: Peak at 1.6h (amplitude: 102.72)
    ROI 3: Peak at 1.6h (amplitude: 61.97)

  Late Active: 3 ROIs (synchronized, anti-phase to early)
    ROI 4: Peak at 3.1h (amplitude: 16.17)
    ROI 5: Peak at 3.2h (amplitude: 86.77)
    ROI 6: Peak at 3.3h (amplitude: 11.54)
```

**Interpretation**:
- ROI 2: Highest amplitude (102.72) → most regular 3.2h rhythm
- ROI 6: Lowest amplitude (11.54) → weakest/most irregular rhythm
- Two groups are ~1.6h apart in 3.2h cycle → anti-phase relationship

### Activity vs Rhythmicity

This is a critical distinction:

**High Activity, High Rhythmicity (e.g., ROI 2)**:
- Frequent movement
- Very regular timing
- High amplitude in phase plot
- **Example**: Animal that moves consistently every 3.2 hours

**High Activity, Low Rhythmicity (e.g., ROI 3)**:
- Very frequent movement
- Irregular timing
- Moderate amplitude
- **Example**: Hyperactive animal with no clear rhythm

**Low Activity, High Rhythmicity (e.g., ROI 5)**:
- Infrequent movement
- Very regular timing
- High amplitude
- **Example**: Calm animal with strong circadian clock

**Low Activity, Low Rhythmicity (e.g., ROI 6)**:
- Infrequent movement
- Irregular timing
- Low amplitude
- **Example**: Sick, stressed, or arrhythmic animal

### Advantages

✅ **Instantaneous Phase**
- Provides precise timing of peak activity within cycle
- Quantifies exact phase relationships (not just "synchronized" or "not")
- Reveals fine-grained temporal coordination

✅ **Amplitude Separation**
- Distinguishes rhythm strength from total activity
- Identifies animals with strong vs weak circadian control
- Measures rhythmicity independently of movement quantity

✅ **Visual Interpretation**
- Polar plot provides intuitive representation
- Clustering is immediately apparent visually
- Easy to identify synchronized groups and anti-phase relationships

✅ **Automatic Clustering**
- Objectively groups ROIs by phase similarity
- Threshold-based classification (Early Active, Late Active, etc.)
- Reduces subjectivity in identifying behavioral groups

✅ **Biological Insight**
- Reveals circadian clock strength (amplitude)
- Identifies zeitgeber effects (synchronized phases)
- Detects social coordination or competition (phase relationships)

### Limitations

⚠️ **Requires Known Period**
- Must use predetermined dominant period from Fisher/FFT
- Cannot detect periods; only analyzes timing at known period
- Not suitable for exploratory period finding

⚠️ **Single-Period Assumption**
- Assumes all ROIs oscillate at same period
- Mixed periods (e.g., 20h and 24h) produce meaningless clusters
- Phase is undefined for arrhythmic or multi-period signals

⚠️ **Hilbert Transform Sensitivity**
- Requires bandpass filtering around target frequency
- Sensitive to filter bandwidth selection
- Can introduce artifacts for non-sinusoidal waveforms

⚠️ **Amplitude Interpretation**
- Amplitude is abstract (not in physical units)
- Relative values matter, absolute values are arbitrary
- Cannot compare amplitudes across different datasets/experiments

⚠️ **Phase Wrapping**
- Phase is circular (0° = 360°)
- Statistical analysis of phase requires circular statistics
- Mean phase can be misleading with wide distributions

⚠️ **Snapshot Limitation**
- Provides single average phase over entire recording
- Cannot detect changes in phase relationships over time
- Transient synchronization is averaged out

⚠️ **Clustering Threshold**
- Phase threshold (e.g., 45°) is somewhat arbitrary
- Different thresholds produce different cluster assignments
- May over- or under-split natural behavioral groups

### When to Use

**Best For:**
- Quantifying precise timing of activity peaks
- Identifying circadian clock strength (amplitude)
- Detecting social synchronization vs individual rhythms
- Comparing zeitgeber entrainment across conditions
- Visualizing phase relationships in publications

**Not Ideal For:**
- Exploratory period detection (use Fisher/FFT first)
- Mixed-period datasets
- Arrhythmic or transient behaviors
- When total activity level is more relevant than rhythm
- Time-varying synchronization (use windowed analysis)

### Best Practices

1. **Period Selection**:
   - **Always** use dominant period from Fisher or FFT analysis first
   - Verify all ROIs share similar period before phase clustering
   - Don't use phase clustering for exploratory period detection
   - Consider separate analyses if ROIs have different periods

2. **Interpretation**:
   - **Critical**: Amplitude measures rhythm strength, NOT activity level
   - High activity + irregular = moderate amplitude
   - Low activity + regular = can have high amplitude
   - Phase clustering is most reliable when all ROIs share same period
   - Mixed periods can produce misleading clusters

3. **Biological Meaning**:
   - Synchronized phases (Δφ < 45°) → Social coordination, shared zeitgeber
   - Anti-phase (Δφ ≈ 180°) → Competition, resource partitioning, territoriality
   - Wide phase distribution → Individual differences, weak coupling, multiple zeitgebers
   - High amplitude → Strong circadian clock, robust rhythm
   - Low amplitude → Weak clock, sick/stressed, arrhythmic

4. **Validation**:
   - Cross-check clusters with similarity matrix
   - Verify period consistency with Fisher/FFT
   - Consider biological context (social species, group housing)
   - Use circular statistics for phase averaging and variance

---

## Interpreting Results

### Comprehensive Analysis Workflow

1. **Fisher Z-Transformation**: Confirm significant rhythms exist
   - Check p-values (p < 0.05 = significant)
   - Identify dominant period(s)

2. **FFT Power Spectrum**: Validate period detection
   - Should agree with Fisher (±1 hour)
   - Identify harmonics and secondary peaks

3. **ROI Similarity**: Find synchronized groups
   - High correlation = similar behavior
   - Check lag values for phase relationships

4. **Coherence**: Verify frequency-specific synchronization
   - High coherence at dominant frequency confirms shared rhythm
   - Check for coherence at harmonics

5. **Phase Clustering**: Quantify timing relationships
   - Amplitude = rhythm strength
   - Phase = timing within cycle
   - Clusters = behavioral groups

### Cross-Validation

**All Methods Should Agree**:

| Analysis | Output | Expected Agreement |
|----------|--------|-------------------|
| Fisher | Period: 24.0h | ✓ |
| FFT | Period: 23.7h | ✓ (within 1h) |
| Similarity | High r for ROIs 1-2 | ✓ (if synchronized) |
| Coherence | High at 24h for ROIs 1-2 | ✓ (confirms shared rhythm) |
| Phase | ROIs 1-2 at same angle | ✓ (confirms in-phase) |

**Disagreement Indicates**:
- Multiple competing rhythms
- Transient vs sustained patterns
- Technical issues (insufficient data, artifacts)

### Example: Complete Analysis

**Dataset**: 6 animals, 72 hours of recording

**Fisher Results**:
```
ROI 1: 24.0h (p < 0.0001) ✓ Significant
ROI 2: 24.0h (p < 0.0001) ✓ Significant
ROI 3: 24.1h (p < 0.0001) ✓ Significant
ROI 4: 24.1h (p < 0.0001) ✓ Significant
ROI 5: 20.0h (p < 0.0001) ✓ Significant
ROI 6: 20.1h (p < 0.0001) ✓ Significant
```

**FFT Results**:
```
ROI 1-4: 23.7-24.3h (excellent agreement)
ROI 5-6: 20.2-20.4h (excellent agreement)
```

**Similarity Matrix**:
```
Group A (ROIs 1-2): r > 0.98 (synchronized, phase 0)
Group B (ROIs 3-4): r > 0.98 (synchronized, phase π)
Group C (ROIs 5-6): r > 0.97 (synchronized, different period)
A ↔ B: r ≈ 0.82 (same period, anti-phase)
A/B ↔ C: r < 0.4 (different periods)
```

**Coherence**:
```
High coherence at 24h: ROIs 1-4
High coherence at 20h: ROIs 5-6
Low coherence between groups
```

**Phase Clustering**:
```
ROIs 1-2: Phase 0°, amplitudes 85-95
ROIs 3-4: Phase 180°, amplitudes 80-90
ROIs 5-6: Not clustered (different period)
```

**Biological Interpretation**:
- Two distinct behavioral groups
- Group A (1-2) and B (3-4): Same 24h circadian rhythm, anti-phase
  - Possibly competitive behavior or turn-taking at resources
- Group C (5-6): Different 20h rhythm
  - Genetic variant, experimental manipulation, or environmental difference
- All rhythms are strong and statistically significant
- High synchronization within groups suggests social entrainment

---

## Export Functionality

### Available Formats

All Extended Analysis results can be exported to:
1. **Excel (.xlsx)**: Multi-sheet workbooks with tables and spectral data
2. **PNG images**: High-resolution plots (300 DPI)

### Export Button

Located in the Extended Analysis tab, next to method selection.

### Excel Export Contents

#### Fisher Z-Transformation
**Sheet 1 - Summary**:
- ROI ID
- Dominant Period (hours)
- Z-Score
- p-value
- Significance (Yes/No)
- Mean Activity
- Std Activity

**Sheet 2 - ROI_X_Periodogram** (one sheet per ROI):
- Period_hours
- Z_Score
- Full periodogram for custom plotting

**Sheet 3 - Parameters**:
- Analysis method
- Min/max period range
- Significance level
- Bin size
- Timestamp

#### FFT Power Spectrum
**Sheet 1 - Summary**:
- ROI ID
- Dominant Period (hours)
- Dominant Power
- Mean Activity
- Number of Peaks

**Sheet 2 - Peak Details**:
- All detected peaks
- Period, frequency, power, prominence

**Sheet 3 - ROI_X_Spectrum** (one sheet per ROI):
- Period_hours
- Power
- Full spectrum for custom plotting (downsampled to max 10,000 points)

**Sheet 4 - Parameters**:
- Analysis settings
- Window function
- Timestamp

#### ROI Similarity
**Sheet 1 - Correlation Matrix**:
- Full NxN correlation matrix
- Row/column labels

**Sheet 2 - Pairwise Similarities**:
- ROI pair (e.g., "1-2")
- Correlation
- Lag (hours)
- Similarity (%)

**Sheet 3 - Clustering** (if available):
- Cluster ID
- ROI members
- Average within-cluster correlation

**Sheet 4 - Parameters**:
- Max lag
- Timestamp

#### Coherence Analysis
**Sheet 1 - Coherence Matrix**:
- Average coherence per ROI pair

**Sheet 2 - Dominant Frequencies**:
- ROI pair
- Dominant frequency (Hz)
- Dominant period (hours)
- Coherence at dominant frequency

**Sheet 3 - Parameters**:
- Segment length
- Overlap
- Window function
- Timestamp

#### Phase Clustering
**Sheet 1 - ROI Phases**:
- ROI ID
- Phase (radians)
- Phase (degrees)
- Peak time (hours within cycle)
- Amplitude
- Cluster assignment

**Sheet 2 - Clusters**:
- Cluster name
- Member ROIs
- Mean phase
- Phase spread (std)

**Sheet 3 - Parameters**:
- Dominant period
- Bandwidth
- Phase threshold
- Timestamp

### Plot Export (PNG)

All plots exported at 300 DPI with:
- White background
- Consistent ROI colors
- High-quality anti-aliasing
- Tight bounding box (minimal whitespace)

### Data Downsampling

For Excel compatibility, spectral data (periodograms, power spectra) is automatically downsampled:
- **Maximum points**: 10,000 per ROI
- **Method**: Regular interval sampling
- **Preservation**: Maintains overall shape and peaks

Excel row limit: 1,048,576 rows per sheet (downsampling ensures compatibility)

### Best Practices

1. **File Naming**: Use descriptive names
   ```
   experiment_condition_fisher_z_2024-01-15.xlsx
   treatment_group_A_fft_spectrum.xlsx
   ```

2. **Metadata**: Parameters sheet documents exact settings
   - Critical for reproducibility
   - Include in methods sections

3. **Plotting**: Spectral data sheets enable custom plots
   - Import into GraphPad, Origin, MATLAB
   - Publication-quality figures with your preferred styling

4. **Version Control**: Export raw data alongside plots
   - Allows re-analysis with different parameters
   - Supports peer review and data sharing

---

## Color Consistency

### ROI-Specific Color Palette

All Extended Analysis plots use consistent colors for each ROI, matching the main ROI Intensity plot.

**Default Color Scheme** (matplotlib default):
```
ROI 1: #1f77b4 (Blue)
ROI 2: #ff7f0e (Orange)
ROI 3: #2ca02c (Green)
ROI 4: #d62728 (Red)
ROI 5: #9467bd (Purple)
ROI 6: #8c564b (Brown)
ROI 7: #e377c2 (Pink)
ROI 8: #7f7f7f (Gray)
ROI 9: #bcbd22 (Olive)
ROI 10: #17becf (Cyan)
```

### Color Application

**Fisher Z-Transformation**:
- Periodogram curves: ROI-specific color
- Dominant period marker: ROI-specific color with black edge
- Significance threshold: Gray (shared across all ROIs)

**FFT Power Spectrum**:
- Power curves: ROI-specific color
- Peak markers: ROI-specific color with black edge

**Phase Clustering**:
- Phase vectors: ROI-specific color
- Scatter points: ROI-specific color with black edge

**Heatmaps (Unchanged)**:
- Similarity Matrix: Green colormap (shows correlation values)
- Coherence Matrix: Red-Yellow-Blue colormap (shows coherence values)
- Rationale: Heatmaps show pairwise relationships, not individual ROIs

### Benefits

1. **Immediate ROI Identification**: Instantly recognize ROIs across all plots
2. **Cross-Plot Comparison**: Easy to track individual ROIs between analyses
3. **Publication Quality**: Consistent figures for papers and presentations
4. **Reduced Cognitive Load**: No mental mapping of "ROI 3 is green in this plot but blue in that plot"

### Example Usage

Looking at Fisher, FFT, and Phase plots together:
- **Orange (ROI 2)** shows strong peak at 24h in Fisher → high power at 24h in FFT → phase at 0° with amplitude 102.7
- **All analyses tell the same story in the same color**

---

## Best Practices

### Experimental Design

1. **Recording Duration**:
   - **Minimum**: 3 complete cycles of expected rhythm
     - 24h rhythm: 72 hours minimum
     - 12h rhythm: 36 hours minimum
     - 3h rhythm: 9 hours minimum
   - **Recommended**: 5-7 days for circadian studies
   - **Rationale**: Statistical power increases with more cycles

2. **Sampling Rate**:
   - **Video frame rate**: 1-5 fps sufficient for most behavior
   - **Analysis sampling**: 5-30 second intervals
   - **Trade-off**: Higher rate = more data but larger files

3. **Environmental Control**:
   - **Light/Dark cycles**: Document precisely
   - **Temperature**: Maintain ±1°C
   - **Feeding**: Consistent timing or ad libitum
   - **Social factors**: Group housing vs isolation

### Data Quality

1. **Baseline Period**:
   - Use first 2-4 hours for baseline calculation
   - Ensure animals are acclimated before recording starts
   - Avoid baseline from experimental manipulation period

2. **Artifact Removal**:
   - Check for equipment failures (camera stops, lighting changes)
   - Identify and exclude outlier periods
   - Document any manual interventions

3. **ROI Consistency**:
   - Maintain consistent ROI definitions across time
   - Avoid overlapping ROIs
   - Ensure ROI size appropriate for animal

### Analysis Parameters

1. **Period Range Selection**:
   ```
   Circadian (mammals): 20-28 hours
   Circadian (insects): 18-26 hours
   Ultradian (feeding): 2-6 hours
   Ultradian (grooming): 0.5-2 hours
   Custom: Based on pilot data or literature
   ```

2. **Significance Thresholds**:
   - **Standard**: p < 0.05 (95% confidence)
   - **Conservative**: p < 0.01 (Bonferroni correction for multiple comparisons)
   - **Exploratory**: p < 0.10 (hypothesis generation)

3. **Binning Guidelines**:
   ```
   Raw sampling: 5 seconds → Bin to: 60 seconds (reduces noise)
   Raw sampling: 30 seconds → Bin to: 120-300 seconds (optional)
   Raw sampling: 300 seconds → No binning needed
   ```

### Statistical Considerations

1. **Multiple Comparisons**:
   - Testing 6 ROIs × 100 periods = 600 tests
   - Consider Bonferroni correction: p_threshold = 0.05 / 600 = 0.000083
   - Or use False Discovery Rate (FDR) control

2. **Sample Size**:
   - **Power analysis**: Calculate required N for detecting expected effect size
   - **Minimum**: 3-5 animals per condition
   - **Recommended**: 8-12 animals per condition

3. **Replication**:
   - **Technical replicates**: Multiple recordings of same animals
   - **Biological replicates**: Different animals
   - **Experimental replicates**: Repeat entire experiment

### Common Issues and Solutions

#### Issue: No Significant Rhythms Detected

**Possible Causes**:
1. Insufficient data duration (< 3 cycles)
2. Highly variable behavior
3. Period outside tested range
4. Arrhythmic condition (e.g., SCN lesion)

**Solutions**:
- Extend recording duration
- Widen period range
- Reduce bin size (more temporal resolution)
- Check raw data for obvious patterns

#### Issue: Multiple Peaks in FFT

**Interpretation**:
1. **Harmonics**: Peaks at 24h, 12h, 8h → fundamental + harmonics (normal)
2. **Multiple rhythms**: Peaks at 24h and 16h → competing oscillators
3. **Artifacts**: Very high frequency peaks → noise or equipment issues

**Solutions**:
- Harmonics are normal, focus on fundamental frequency
- Multiple rhythms may require different period ranges
- Filter high-frequency noise

#### Issue: Disagreement Between Fisher and FFT

**Typical Scenarios**:
```
Fisher: 24.0h (p < 0.001)
FFT: 23.5h (high power)
Difference: 0.5h (acceptable, within resolution limits)
```

**Concerning**:
```
Fisher: 24.0h (p < 0.001)
FFT: 18.0h (high power)
Difference: 6h (investigate!)
```

**Solutions**:
- Small differences (< 1h): Normal due to methodology
- Large differences (> 2h): Check data quality, may have multiple rhythms
- Visually inspect raw data for clarity

#### Issue: Low Coherence Despite High Similarity

**Interpretation**:
- ROIs synchronized in overall pattern but not frequency-specific
- Different frequency components dominate in each ROI
- Transient synchronization vs sustained coupling

**Solutions**:
- Check coherence at multiple frequencies
- Use similarity matrix as primary measure
- Consider time-resolved coherence analysis

### Publishing Results

#### Methods Section Template

```
Extended Analysis of Circadian Rhythms

Activity data were analyzed using the napari-hdf5-activity plugin
(version X.X.X). Circadian rhythms were detected using Fisher's
Z-transformation with a significance threshold of p < 0.05, testing
periods from 20 to 28 hours. Results were validated using Fast
Fourier Transform (FFT) power spectrum analysis with Hann windowing.

Synchronization between animals was assessed using cross-correlation
analysis (maximum lag: 12 hours) and coherence analysis (Welch's
method, 256-sample segments, 50% overlap). Phase relationships were
quantified using Hilbert transform-based phase clustering.

Data were binned to 60-second intervals prior to analysis. Only ROIs
showing significant circadian rhythms (Fisher Z-transformation,
p < 0.05) were included in synchronization analyses.
```

#### Reporting Standards

**Fisher Z-Transformation**:
```
"All animals exhibited significant circadian rhythms
(ROI 1: period = 24.2 ± 0.3 h, Z = 125.4, p < 0.0001;
ROI 2: period = 23.8 ± 0.2 h, Z = 118.7, p < 0.0001;
n = 6 animals)"
```

**Synchronization**:
```
"Animals showed strong behavioral synchronization
(pairwise correlation: r = 0.89 ± 0.05, p < 0.001;
coherence at 24h: 0.85 ± 0.03; n = 15 pairs)"
```

**Phase Relationships**:
```
"Phase clustering identified two groups: early active
(ROIs 1-3, peak at 13.2 ± 0.4 h, n = 3) and late active
(ROIs 4-6, peak at 1.8 ± 0.5 h, n = 3), exhibiting
anti-phase relationship (Δφ = 180 ± 12°, p < 0.001)"
```

#### Figure Legends

**Fisher Z-Transformation**:
```
Figure 1. Circadian rhythm analysis using Fisher Z-transformation.
(A-F) Periodograms for individual ROIs showing Z-score across tested
periods (20-28 h). Colored curves represent Z-scores, with gray dashed
lines indicating significance threshold (p = 0.05). Colored vertical
lines and markers indicate dominant periods. All ROIs showed significant
circadian rhythms (p < 0.0001).
```

**Phase Clustering**:
```
Figure 3. Phase relationships of circadian activity.
Polar plot showing instantaneous phase and amplitude for each ROI
(n = 6 animals). Radial distance represents rhythmic amplitude
(strength of oscillation), angular position represents phase
(timing of peak activity within 24h cycle). Colors correspond to
individual ROIs. Two distinct clusters are evident: early active
(0°, ROIs 1-3) and late active (180°, ROIs 4-6).
```

---

## Advanced Topics

### Custom Period Ranges

For specific research questions, adjust period ranges:

**Jet Lag Studies**:
```
Pre-shift: 20-28 hours (circadian)
Post-shift Day 1-3: 18-30 hours (wider to capture transients)
Post-shift Day 4+: 20-28 hours (re-entrained)
```

**Ultradian + Circadian**:
```
Analysis 1: 2-6 hours (ultradian feeding rhythms)
Analysis 2: 20-28 hours (circadian rest-activity)
Compare both for coupling analysis
```

### Time-Resolved Analysis

For non-stationary data (changing rhythms over time):

1. **Sliding Window**:
   - Analyze 24-hour windows with 12-hour step
   - Track period changes over days
   - Useful for entrainment studies

2. **Before/After Comparison**:
   - Analyze baseline period (days 1-3)
   - Analyze treatment period (days 4-7)
   - Statistical comparison of parameters

### Combining Multiple Methods

**Workflow for Maximum Insight**:

1. Fisher → Detect significant rhythms, get periods
2. FFT → Validate periods, check for harmonics
3. Use dominant period in Similarity, Coherence, Phase analyses
4. Cross-validate: All methods should tell consistent story

**Red Flags**:
- Fisher significant but FFT shows no peak → check data quality
- High similarity but low coherence → may be artifact
- Phase clustering shows groups but similarity doesn't → different periods

### Batch Analysis

For multiple experiments or conditions:

1. **Export all results to Excel**
2. **Import into statistical software** (R, Python, SPSS)
3. **Compare parameters**:
   - ANOVA on dominant periods across conditions
   - t-tests on amplitudes between groups
   - Correlation analysis of similarity matrices

4. **Meta-analysis**:
   - Combine p-values across experiments
   - Weighted averages of periods
   - Cluster consistency across datasets

---

## Troubleshooting

### Error Messages

**"Time series too short for analysis"**
- Need minimum 10 samples
- Solution: Reduce bin size or collect more data

**"No data in specified period range"**
- Period range outside data duration
- Solution: Widen range or collect longer recording

**"Insufficient data for FFT analysis"**
- < 10 samples after binning
- Solution: Reduce bin size

**"No significant circadian rhythm detected"**
- Not an error, just result
- Check if rhythm truly absent or parameters need adjustment

### Performance Issues

**Slow Analysis**:
- Large datasets (> 100,000 points) take time
- Solution: Use binning to reduce data size
- FFT is fastest, Fisher is slower

**Memory Errors**:
- Very long recordings (weeks) may exceed memory
- Solution: Analyze shorter segments separately
- Or increase bin size significantly

### Data Quality Checks

Before Extended Analysis:

1. **Visual Inspection**:
   - Plot raw ROI intensity over time
   - Look for obvious rhythms by eye
   - Check for artifacts, dropouts

2. **Basic Statistics**:
   - Coefficient of Variation (CV) < 1.0 for reasonable data
   - Check for outliers (> 3 SD from mean)
   - Ensure non-zero variance

3. **Duration Check**:
   - Verify recording length ≥ 3 × expected period
   - Check for gaps in data

---

## References

### Scientific Background

**Fisher Z-Transformation**:
- Fisher, R.A. (1929). "Tests of significance in harmonic analysis." Proceedings of the Royal Society A.
- Enright, J.T. (1965). "The search for rhythmicity in biological time-series." Journal of Theoretical Biology.

**FFT Methods**:
- Cooley, J.W., & Tukey, J.W. (1965). "An algorithm for the machine calculation of complex Fourier series." Mathematics of Computation.
- Welch, P.D. (1967). "The use of fast Fourier transform for the estimation of power spectra." IEEE Transactions on Audio and Electroacoustics.

**Circadian Analysis**:
- Refinetti, R., Lissen, G.C., & Halberg, F. (2007). "Procedures for numerical analysis of circadian rhythms." Biological Rhythm Research.
- Levine, J.D., Funes, P., Dowse, H.B., & Hall, J.C. (2002). "Resetting the circadian clock by social experience in Drosophila." Science.

**Phase Analysis**:
- Gabor, D. (1946). "Theory of communication." Journal of the Institution of Electrical Engineers.
- Pikovsky, A., Rosenblum, M., & Kurths, J. (2001). "Synchronization: A Universal Concept in Nonlinear Sciences." Cambridge University Press.

### Software Implementation

**Algorithms Used**:
- NumPy FFT: `numpy.fft.rfft` with zero-padding
- SciPy Signal Processing: `scipy.signal.welch`, `scipy.signal.find_peaks`, `scipy.signal.hilbert`
- SciPy Statistics: `scipy.stats.chi2` for Fisher significance
- SciPy Clustering: `scipy.cluster.hierarchy` for dendrogram

**Validation**:
All methods have been validated against:
- Synthetic data with known periods (test_data_*.h5)
- Published circadian datasets
- Cross-validation between methods

---

## Contact and Support

For questions, bug reports, or feature requests:
- GitHub Issues: https://github.com/[your-repo]/napari-hdf5-activity/issues
- Email: [your-email]

Please include:
- napari-hdf5-activity version
- Sample data (if possible)
- Error messages or unexpected results
- Analysis parameters used

---

## Changelog

### Version 1.0 (2024)
- Initial implementation of Extended Analysis tab
- Fisher Z-Transformation periodogram
- FFT Power Spectrum analysis
- ROI Similarity Matrix with clustering
- Coherence Analysis
- Phase Clustering with polar visualization
- Excel and PNG export functionality
- ROI-specific color consistency
- Comprehensive documentation

---

## License

This software is provided under [your license].
Use in publications should cite:
```
[Your Citation Information]
```
