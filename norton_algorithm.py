# Python port of Andrew Norton's SuperWASP Variable Stars period detector.
# DOI: 10.3847/2515-5172/aaf291; CLEAN based on Harry Lehto's SDEEPCL (1992).
"""Numerical port of runfindper6.f; times and returned periods are in seconds.

Numba compiles Python loops, without requiring the original Fortran at runtime.
Double precision and explicit degenerate-input checks replace unsafe legacy
array bounds and uninitialised values. See docs/norton_port.md for port details.
"""

import numpy as np
from numba import njit

DEFAULTS = {
    "trim_fraction": 0.01,
    "running_half_window": 3,
    "spike_sigma": 5.0,
    "frequency_max_hz": 2.7778e-4,
    "frequency_bins_max": 99997,
    "clean_density": 6.0,
    "clean_gain": 0.2,
    "clean_components_max": 99999,
    "peak_sigma": 3.5,
    "period_tolerance": 0.02,
    "fold_bins": 50,
    "min_bin_population": 25,
    "min_filled_fraction": 0.9,
    "max_raggedness": 0.10,
    "max_scatter_ratio": 2.0,
    "chisq_limit": 10.0,
    "refine_factor": 1.00001,
    "refine_half_range": 0.01,
    "max_periods": 50,
    "legacy_integer_scatter": True,
    "profile_trim_fraction": 0.02,
    "profile_spike_sigma": 10.0,
}


@njit(cache=True)
def _local_clip(t, x, e, half_window, sigma):
    keep = np.ones(len(t), dtype=np.bool_)
    for i in range(len(t)):
        total = 0.0
        count = 0
        for j in range(max(0, i - half_window), min(len(t), i + half_window + 1)):
            if j != i and keep[j]:
                total += x[j]
                count += 1
        if count == 0 or abs(x[i] - total / count) > sigma * e[i]:
            keep[i] = False
    return t[keep], x[keep], e[keep]


def clip_spikes(t, x, e, config):
    order = np.argsort(x, kind="stable")
    chop = int(len(x) * config["trim_fraction"])
    keep = np.ones(len(x), dtype=bool)
    keep[order[:chop]] = False
    # Fortran's inclusive (npts-nchop):npts removes nchop+1 high points.
    keep[order[len(x) - chop - 1 :]] = False
    return _local_clip(
        t[keep], x[keep], e[keep], config["running_half_window"], config["spike_sigma"]
    )


@njit(cache=True)
def dirty_transform(t, x, df, bins):
    """Deeming transform and spectral window via the original trig recurrence."""
    real = np.zeros(bins + 1)
    imag = np.zeros(bins + 1)
    wr = np.zeros(2 * (bins + 1) + 1)
    wi = np.zeros(2 * (bins + 1) + 1)
    for i in range(len(t)):
        angle = -2 * np.pi * df * t[i]
        ds, dc = np.sin(angle), np.cos(angle)
        rs, rc = 0.0, 1.0
        for k in range(len(wr)):
            wr[k] += rc
            wi[k] += rs
            if k <= bins:
                real[k] += rc * x[i]
                imag[k] += rs * x[i]
            rs, rc = dc * rs + ds * rc, dc * rc - ds * rs
    return real, imag, wr, wi


@njit(cache=True)
def clean_components(
    real, imag, wr, wi, npoints, width, gain, max_iterations, max_components
):
    """SDEEPCL variable gain, alias subtraction and Gaussian restoration."""
    bins = len(real) - 1
    components_r = np.zeros(bins + 1)
    components_i = np.zeros(bins + 1)
    banned = np.zeros(bins + 1, dtype=np.bool_)
    normalization = float(npoints * npoints)
    convergence = 0.5 * np.sqrt(gain * gain + np.exp(np.log(2) / width**2))
    previous = 1e15
    older = (1 + 1.1 * convergence) * previous
    counted = 0
    total = 0
    # Low-frequency components do not count toward MAXITE in SDEEPCL.
    count_floor = 1.1 * 3.6 * bins / npoints
    stop = 0
    while total < max_components:
        peak = -1
        maximum = 0.0
        for k in range(bins + 1):
            value = (real[k] ** 2 + imag[k] ** 2) / normalization
            if not banned[k] and value > maximum:
                peak, maximum = k, value
        if peak < 0:
            stop = 1
            break
        if counted >= max_iterations:
            break
        if (
            maximum > (1 + 2 * convergence) * previous
            and previous > (1 + 2 * convergence) * older
        ):
            stop = 2
            break
        older, previous = previous, maximum
        dct, dst = real[peak], imag[peak]
        numerator, denominator = 0.0, 0.0
        for k in range(1, bins + 1):
            a, b = abs(k - peak), k + peak
            sign = 1.0 if k >= peak else -1.0
            dc1, dc2, ds1, ds2 = wr[a], wr[b], sign * wi[a], wi[b]
            r = (dc1 + dc2) * dct + (ds2 - ds1) * dst
            s = (dc1 - dc2) * dst + (ds1 + ds2) * dct
            numerator += real[k] * r + imag[k] * s
            denominator += r * r + s * s
        r0 = wr[peak] * dct + wi[peak] * dst
        numerator += real[0] * r0
        denominator += 2 * r0 * r0
        if denominator <= 0 or not np.isfinite(denominator):
            stop = 3
            break
        adaptive_gain = gain * numerator * npoints / denominator
        if adaptive_gain / gain <= 1.0 / npoints:
            for k in range(bins + 1):
                if abs(k - peak) <= width:
                    banned[k] = True
            continue
        total += 1
        if peak >= count_floor:
            counted += 1
        banned[:] = False
        cr, ci = adaptive_gain * dct, adaptive_gain * dst
        components_r[peak] += cr
        components_i[peak] += ci
        for k in range(bins + 1):
            a, b = abs(k - peak), k + peak
            sign = 1.0 if k >= peak else -1.0
            dc1, dc2, ds1, ds2 = wr[a], wr[b], sign * wi[a], wi[b]
            real[k] -= (cr * (dc1 + dc2) - ci * (ds1 - ds2)) / npoints
            imag[k] -= (cr * (ds1 + ds2) + ci * (dc1 - dc2)) / npoints
    if total == max_components:
        stop = 4
    constant = -np.log(2) / (width / 2) ** 2
    for peak in range(bins + 1):
        cr, ci = components_r[peak], components_i[peak]
        if cr == 0 and ci == 0:
            continue
        # NINT uses nearest integer, ties away from zero.
        lower = max(0, int(np.floor(peak - 5 * width + 1 + 0.5)))
        upper = min(bins, int(np.floor(peak + 5 * width - 1 + 0.5)))
        for k in range(lower, upper + 1):
            direct = np.exp(max(-50.0, (k - peak) ** 2 * constant))
            mirror = np.exp(max(-50.0, (k + peak) ** 2 * constant))
            real[k] += cr * (direct + mirror)
            imag[k] += ci * (direct - mirror)
    return (real * real + imag * imag) / normalization, counted, total, stop


def clean_spectrum(t, x, config):
    baseline = np.ptp(t)
    hpbw = 1.2067 / baseline
    bins = min(
        int(config["frequency_max_hz"] / hpbw * config["clean_density"]) + 1,
        config["frequency_bins_max"],
    )
    if bins < 3:
        raise ValueError("Time baseline is too short for the CLEAN frequency grid")
    df = config["frequency_max_hz"] / bins
    centered = x - np.mean(x)
    second = np.mean(centered**2)
    fourth = np.mean(centered**4)
    spike_factor = 1 - fourth / ((len(x) - 1) * second**2)
    iterations = min(
        config["clean_components_max"],
        int(0.5 / config["clean_gain"] * np.sqrt(10 * len(x)) * spike_factor**2),
    )
    if iterations < 1:
        raise ValueError("Degenerate CLEAN iteration count")
    real, imag, wr, wi = dirty_transform(t - np.mean(t), centered, df, bins)
    power, counted, total, stop = clean_components(
        real,
        imag,
        wr,
        wi,
        len(x),
        hpbw / df,
        config["clean_gain"],
        iterations,
        config["clean_components_max"],
    )
    return (
        np.arange(bins + 1) * df,
        power,
        {
            "frequency_bins": bins,
            "clean_iterations": counted,
            "clean_components": total,
            "clean_stop": [
                "iteration_limit",
                "no_unblocked_peak",
                "divergence",
                "degenerate_gain",
                "component_limit",
            ][stop],
        },
    )


def period_flag(period):
    """Original flags and precedence, applied to the initial CLEAN period."""
    days = period / 86400
    sidereal_days = period / 86164
    if 0.8 <= days <= 1.3:
        return 1
    if 0.46 <= days <= 0.58:
        return 2
    if abs(days - int(days + 0.5)) <= 0.02:
        return 20
    if days < 1:
        for divisor in range(3, 17):
            if abs(sidereal_days - 1 / divisor) <= 0.02 / divisor:
                return divisor
    for low, high, flag in [
        (5.8, 7.3, 87),
        (9, 11, 88),
        (12.5, 16, 89),
        (27, 34, 91),
        (54, 64, 92),
        (81, 96, 93),
    ]:
        if low <= days <= high:
            return flag
    for low, high, flag in [
        (177800, 182000, 31),
        (278600, 285100, 32),
        (380000, 407400, 33),
    ]:
        if low <= period <= high:
            return flag
    return 0


def spectral_candidates(frequency, power, config):
    f, p = frequency[1:], power[1:]
    if not np.isfinite(p).all() or not np.any(p > 0):
        raise ValueError("Nonfinite or empty CLEAN power spectrum")
    # Legacy carries the previous log power at zero-valued frequencies.
    first = np.flatnonzero(p > 0)[0]
    logp = np.empty(len(p))
    previous = np.log10(p[first])
    for i, value in enumerate(p):
        if value > 0:
            previous = np.log10(value)
        logp[i] = previous
    logf = np.log10(f)
    slope, intercept = np.polyfit(logf, logp, 1)
    background = intercept + slope * logf
    deviation = np.sqrt(np.mean((logp[p > 0] - background[p > 0]) ** 2))
    if deviation <= 0 or not np.isfinite(deviation):
        return [], {"log_power_slope": float(slope), "log_power_scatter": 0.0}
    peaks = []
    for i in range(1, len(p) - 1):
        if p[i] <= 0 or logp[i] < background[i] + config["peak_sigma"] * deviation:
            continue
        if logp[i] < logp[i - 1] or logp[i] < logp[i + 1]:
            continue
        ym, y, yp = np.log(np.maximum(p[i - 1 : i + 2], np.finfo(float).tiny))
        curvature = ym + yp - 2 * y
        if curvature >= 0:
            continue
        df = f[i] - f[i - 1]
        offset = -df * (yp - ym) / (2 * curvature)
        peak_frequency = f[i] + offset
        peak_logpower = (
            y + curvature / (2 * df * df) * offset**2 + (yp - ym) / (2 * df) * offset
        )
        period = 1 / peak_frequency
        significance = (peak_logpower / np.log(10) - background[i]) / deviation
        peaks.append(
            {
                "clean_period_seconds": float(period),
                "sigma": float(significance),
                "period_flag": period_flag(period),
            }
        )
    # Preserve the original grouping order and its comparison to the seed peak.
    used = set()
    unique = []
    for i, seed in enumerate(peaks):
        if i in used:
            continue
        used.add(i)
        chosen = seed
        for j in range(i + 1, len(peaks)):
            if (
                abs(
                    (seed["clean_period_seconds"] - peaks[j]["clean_period_seconds"])
                    / seed["clean_period_seconds"]
                )
                <= config["period_tolerance"]
            ):
                used.add(j)
                if peaks[j]["sigma"] > seed["sigma"]:
                    chosen = peaks[j]
        unique.append(chosen.copy())
    return unique, {
        "log_power_slope": float(slope),
        "log_power_intercept": float(intercept),
        "log_power_scatter": float(deviation),
    }


@njit(cache=True)
def fold_statistic(
    t,
    x,
    period,
    bins,
    minpop,
    filled_fraction,
    rag_limit,
    scatter_limit,
    integer_scatter,
):
    count = np.zeros(bins, dtype=np.int64)
    sums = np.zeros(bins)
    lo = np.full(bins, t[-1])
    hi = np.full(bins, t[0])
    assigned = np.empty(len(t), dtype=np.int64)
    for i in range(len(t)):
        phase = ((t[i] - t[0]) / period) % 1.0
        b = min(bins - 1, int(phase * bins))
        assigned[i] = b
        count[b] += 1
        sums[b] += x[i]
        lo[b] = min(lo[b], t[i])
        hi[b] = max(hi[b], t[i])
    populated = (count > minpop) & ((hi - lo) > period / bins)
    m = np.sum(populated)
    if m < filled_fraction * bins:
        return 0.0
    means = np.zeros(bins)
    overall = 0.0
    minimum, maximum = np.inf, -np.inf
    for b in range(bins):
        if populated[b]:
            means[b] = sums[b] / count[b]
            overall += means[b]
            minimum = min(minimum, means[b])
            maximum = max(maximum, means[b])
    overall /= m
    if maximum <= minimum:
        return 0.0
    # The legacy backwards DO loops omit a negative stride, so the last edge
    # is neither doubled nor closed cyclically. Sum adjacent occupied bins.
    path, previous = 0.0, -1
    for b in range(bins):
        if populated[b]:
            if previous >= 0:
                path += abs(means[b] - means[previous])
            previous = b
    if path / bins / (maximum - minimum) > rag_limit:
        return 0.0
    ssa = 0.0
    residual = np.zeros(bins)
    for b in range(bins):
        if populated[b]:
            ssa += count[b] * (means[b] - overall) ** 2
    for i in range(len(t)):
        b = assigned[i]
        if populated[b]:
            residual[b] += (x[i] - means[b]) ** 2
    for b in range(bins):
        if populated[b]:
            other = 0.0
            for j in range(bins):
                if populated[j] and j != b:
                    scatter = np.sqrt(residual[j] / count[b])
                    other += int(scatter) if integer_scatter else scatter
            scatter = np.sqrt(residual[b] / count[b])
            current = int(scatter) if integer_scatter else scatter
            if current > scatter_limit * other / (m - 1):
                return 0.0
    denominator = np.sum(residual) * (m - 1)
    if denominator <= 0:
        return 0.0
    return (len(t) - m) * ssa / denominator


@njit(cache=True)
def refine_period(
    t,
    x,
    initial,
    bins,
    minpop,
    fraction,
    rag,
    scatter,
    chisq_limit,
    half_range,
    factor,
    integer_scatter,
):
    period = initial * (1 - half_range)
    end = initial * (1 + half_range)
    ntrials = int(np.log10(end / period) / np.log10(factor))
    best_period, best_ratio = 0.0, 0.0
    for _ in range(ntrials):
        ratio = fold_statistic(
            t, x, period, bins, minpop, fraction, rag, scatter, integer_scatter
        )
        threshold = (
            chisq_limit
            if period < 86400
            else chisq_limit * (1 + 0.75 * np.log10(period / 86400))
        )
        if ratio >= threshold and ratio > best_ratio:
            best_period, best_ratio = period, ratio
        period *= factor
    return best_period, best_ratio


def folded_profile(t, x, e, period, bins=100):
    phase = (t / period) % 1.0
    index = np.minimum((phase * bins).astype(int), bins - 1)
    count = np.bincount(index, minlength=bins)
    total = np.bincount(index, weights=x, minlength=bins)
    errors = np.bincount(index, weights=e * e, minlength=bins)
    occupied = count > 0
    means = np.full(bins, np.inf)
    means[occupied] = total[occupied] / count[occupied]
    minimum_bin = int(np.argmin(means))
    count, total, errors = (
        np.roll(array, -minimum_bin) for array in (count, total, errors)
    )
    return {
        "phase_shift": (minimum_bin + 0.5) / bins,
        "phase": ((np.arange(bins) + 0.5) / bins).tolist(),
        "flux": [float(total[i] / count[i]) if count[i] else None for i in range(bins)],
        "flux_error": [
            float(np.sqrt(errors[i]) / count[i]) if count[i] else None
            for i in range(bins)
        ],
        "count": count.tolist(),
    }


def make_profile(prepared, period, config):
    """The independent 2% / 10-sigma display clipping in Norton's fold routine."""
    display_config = {
        **config,
        "trim_fraction": config["profile_trim_fraction"],
        "spike_sigma": config["profile_spike_sigma"],
    }
    return folded_profile(*clip_spikes(*prepared, display_config), period)


def detect_periods(t, flux, error, config=None):
    config = {**DEFAULTS, **(config or {})}
    t, flux, error = (np.asarray(array, dtype=np.float64) for array in (t, flux, error))
    if len(t) < 3 or not all(np.isfinite(a).all() for a in (t, flux, error)):
        raise ValueError("Expected finite time, flux and error arrays")
    if (
        len(t) != len(flux)
        or len(t) != len(error)
        or np.any(error <= 0)
        or np.any(np.diff(t) < 0)
    ):
        raise ValueError("Light curve must be time sorted with positive errors")
    prepared = (t, flux, error)
    t, flux, error = clip_spikes(*prepared, config)
    if len(t) < 3 or np.ptp(t) <= 0 or np.ptp(flux) <= 0:
        return [], {
            "n_clean": len(t),
            "reason": "insufficient_data_after_spike_removal",
        }
    frequency, power, diagnostics = clean_spectrum(t, flux, config)
    candidates, background = spectral_candidates(frequency, power, config)
    diagnostics.update(background, n_clean=len(t), n_clean_peaks=len(candidates))
    return refine_candidates(t, flux, error, candidates, config, prepared), diagnostics


def refine_candidates(t, flux, error, candidates, config, profile_data=None):
    centered = flux - np.mean(flux)
    accepted = []
    for candidate in candidates:
        period, ratio = refine_period(
            t,
            centered,
            candidate["clean_period_seconds"],
            config["fold_bins"],
            config["min_bin_population"],
            config["min_filled_fraction"],
            config["max_raggedness"],
            config["max_scatter_ratio"],
            config["chisq_limit"],
            config["refine_half_range"],
            config["refine_factor"],
            config["legacy_integer_scatter"],
        )
        if period > 0:
            accepted.append(
                {
                    **candidate,
                    "period_seconds": float(period),
                    "period_days": float(period / 86400),
                    "chi_squared_ratio": float(ratio),
                    "folded_profile": make_profile(
                        profile_data if profile_data is not None else (t, flux, error),
                        period,
                        config,
                    ),
                }
            )
    accepted.sort(key=lambda item: item["period_seconds"], reverse=True)
    for i, item in enumerate(accepted[: config["max_periods"]], start=1):
        item["period_number"] = i
    return accepted[: config["max_periods"]]
