# Norton period-search port

`classify_norton.py` is based on Andrew Norton's work developed for SuperWASP
Variable Stars, described in Norton (2018), *A Zooniverse Project to Classify
Periodic Variable Stars from SuperWASP*, Research Notes of the AAS 2(4), 216.
**DOI: [10.3847/2515-5172/aaf291](https://doi.org/10.3847/2515-5172/aaf291).**

The numerical reference is Andrew's supplied October 2024 `runfindper6.f`, not
an implementation reconstructed from the paper. Its SHA-256 is
`880553a992b286852bef8da34818a6121121b397ab5ac8e3f638ee3bf39e6289`.
The CLEAN routine is credited in that source to Harry Lehto (SDEEPCL, April 1992),
with Norton's modifications. The reference also credits Clive Page's fitting
routines. `file.list` and `period_results.dat` describe the old I/O; they do not
supply the five SuperWASP light curves needed to reproduce those example results.
These reference files are not runtime dependencies.

## Detection procedure

The Python functions are in `norton_algorithm.py`. Numba compiles their inner
loops; no Fortran compiler or GPU is needed to run the search.

1. Apply NGTS quality filtering and, by default, inverse-variance flux averaging
   into 5-minute bins. Preserve flux units (ADU/s); do not convert to magnitudes
   or normalize the flux amplitude. Convert HJD day differences to seconds in
   double precision. The minimum input is 1,000 usable points after this binning,
   matching the original launcher threshold before Norton clipping.
2. Remove the lowest 1% and highest 1% plus one point (the original inclusive
   indexing), then sequentially reject points more than five measurement errors
   from the mean of up to three neighbours on either side. Previously rejected
   neighbours are excluded. This is distinct from the NGTS pipeline's outlier
   bit, which the default quality mask retains.
3. Compute the unweighted, mean-subtracted Deeming transform and sampling window
   using the trigonometric recurrence. Use the original CLEAN grid: half-power
   beam width `1.2067 / baseline`, six grid samples per beam width, maximum
   frequency `2.7778e-4 Hz` (approximately a one-hour minimum period), capped at
   99,997 frequency intervals. CLEAN uses gain 0.2, the original moment-dependent
   iteration limit, adaptive gain, rejection of poor components, treatment of
   low-frequency components, and Gaussian beam restoration plus residuals.
4. Fit a straight line to log10 power versus log10 frequency. Find peaks at least
   3.5 RMS residuals above this background and refine their frequencies with the
   three-point log-power parabola. Merge recurrences within 2%, retaining the
   original seed-comparison and frequency ordering rules.
5. Refine each candidate from 0.99 to approximately 1.01 times its period, using
   successive factors of 1.00001 (approximately 2,000 trials). Fold into 50 bins.
   A bin needs **more than** 25 points spanning more than one bin-width in time;
   at least 45 bins must qualify. Apply the original raggedness and scatter
   exclusions. Maximise the Davies-style ratio
   `(N - M) * SSA / ((M - 1) * SSE)`, with the same period-dependent threshold:
   10 below a day, and `10 * (1 + 0.75 * log10(period_days))` otherwise.
6. Keep accepted periods in descending period order, up to 50 per source. Retain
   all CLEAN candidates dynamically while evaluating them, rather than risking
   the original fixed-size array overflow. Diagnostics record the number of
   accepted periods before the final limit.

`period_flag` is computed at the **initial CLEAN period**, just as in the source;
refinement does not recompute it. Codes retain the original precedence:

| Code | Warning |
| --- | --- |
| 0 | No Norton period warning |
| 1, 2 | Around one day / half a day |
| 3–16 | Around the corresponding fractional sidereal day |
| 20 | Near an integer number of days |
| 87, 88, 89 | Roughly quarter, third, half month |
| 91, 92, 93 | Roughly one, two, three months |
| 31–33 | Three additional empirical SuperWASP artefact intervals |

These are SuperWASP-derived warnings, not a validated NGTS artefact model. The
`sigma` value is CLEAN log-power significance, **not** a period uncertainty or a
calibrated false-alarm probability. `chi_squared_ratio` is the folding statistic,
not an ordinary reduced chi-squared.

## Deliberate numerical repairs and retained legacy behaviours

- All numerical arrays use float64. HJD offsets are subtracted before converting
  to seconds. The old `MAX1`/`MIN1` extrema calls are replaced by real-valued
  extrema. On the installed gfortran compiler, the unmodified source produced
  `TIMMIN = -2147483648` on the synthetic test, giving a spurious baseline and
  forcing the frequency grid to its cap. That bug is not reproduced.
- The power-background line is solved directly by least squares, rather than
  the early-stopped Marquardt fit. Small significance differences are expected.
- Zero denominators, empty neighbourhoods, zero baselines, constant data and
  nonfinite numerical results are handled explicitly. No uninitialised array
  values or accidental out-of-bounds accesses are reproduced.
- The legacy integer `scatbin` truncation and its use of the current bin's
  population in the comparison with other bins are retained. They make this
  particular rejection test depend on flux units.
- The source's reverse `DO` loops omit the negative stride. For its fixed 50-bin
  folds, those loops do not run: the raggedness calculation traverses neighbouring
  populated bins without a closing phase-wrap edge. The port retains this.
- The 100-bin display profiles use the separate `fold` routine's 2% tail trim and
  10-sigma local rejection, starting from the NGTS-prepared light curve. They
  rotate the minimum-flux bin to the first bin. Unlike the ASCII output, empty
  bins contain null fluxes; counts, propagated errors and phase shift are also
  retained. They use the local first-observation time origin rather than a WASP
  absolute epoch, and do not reproduce the ancillary routine's 100,000-row cap.

These choices preserve the scientific detection procedure but are not a claim
of bit-for-bit equivalence to every build of the original Fortran.

## Validation

The tests check the Fourier transform against an independent direct complex
exponential sum, warning flags, sequential clipping, bin population thresholds,
injected-period recovery, noise rejection, and checkpoint recovery between
candidate refinements. An optional cross-language test compiles the supplied
`data/runfindper6.f` when gfortran is available, replacing only the four extrema
calls and promoting REAL arithmetic to double precision. It runs the original
launcher, CLEAN routine and folding search on the same deterministic synthetic
light curve. The recovered period and folding ratio agree within the reference
output precision; significance agrees to 0.01.

```bash
uv run python -m pytest -q
```
