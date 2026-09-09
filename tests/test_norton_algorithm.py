"""Check the numerical port against independent formulas and the supplied Fortran."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import norton_algorithm as norton


def synthetic_curve():
    rng = np.random.default_rng(124)
    t = np.sort(rng.uniform(0, 30 * 86400, 2200))
    flux = (
        1000 + 100 * np.sin(2 * np.pi * t / (0.73 * 86400)) + rng.normal(0, 8, len(t))
    )
    return t, flux, np.full(len(t), 8.0)


def test_dirty_transform_matches_direct_complex_sum():
    rng = np.random.default_rng(45)
    t = np.sort(rng.uniform(-20, 20, 81))
    x = rng.normal(size=len(t))
    bins, df = 32, 0.007
    r, i, wr, wi = norton.dirty_transform(t, x, df, bins)
    exponent = np.exp(-2j * np.pi * df * np.arange(2 * (bins + 1) + 1)[:, None] * t)
    np.testing.assert_allclose(r + 1j * i, exponent[: bins + 1] @ x, atol=1e-12)
    np.testing.assert_allclose(wr + 1j * wi, exponent.sum(axis=1), atol=1e-12)


@pytest.mark.parametrize(
    "seconds,flag",
    [
        (86400, 1),
        (43200, 2),
        (86164 / 3, 3),
        (86164 / 16, 16),
        (2 * 86400, 20),
        (6.4 * 86400, 87),
        (9.5 * 86400, 88),
        (14.4 * 86400, 89),
        (29.5 * 86400, 91),
        (59.5 * 86400, 92),
        (88.5 * 86400, 93),
        (180000, 31),
        (281000, 32),
        (390000, 33),
        (0.73 * 86400, 0),
    ],
)
def test_original_warning_flags_and_precedence(seconds, flag):
    assert norton.period_flag(seconds) == flag


def test_trimming_preserves_asymmetric_legacy_endpoints():
    t = np.arange(100, dtype=float)
    x = np.arange(100, dtype=float)
    clipped, _, _ = norton.clip_spikes(t, x, np.full(100, 1000.0), norton.DEFAULTS)
    np.testing.assert_array_equal(clipped, t[1:98])


def test_local_clip_is_sequential_and_excludes_rejected_neighbours():
    t = np.arange(5, dtype=float)
    x = np.array([0, 100, 0, 0, 0], dtype=float)
    result = norton._local_clip(t, x, np.ones(5), 1, 5)
    # First point is rejected using its future neighbour; second then uses only
    # the next zero-valued point and is also rejected.
    np.testing.assert_array_equal(result[0], [2, 3, 4])


def test_population_cut_is_strictly_greater_than_25():
    t = np.arange(1250, dtype=float)
    flux = np.sin(2 * np.pi * t / 50)
    assert norton.fold_statistic(t, flux, 50.0, 50, 25, 0.9, 0.1, 2.0, True) == 0


def test_synthetic_period_recovery():
    periods, diagnostics = norton.detect_periods(*synthetic_curve())
    assert diagnostics["n_clean"] == 2153
    assert diagnostics["frequency_bins"] == 3580
    assert len(periods) == 1
    assert periods[0]["period_days"] == pytest.approx(0.73, abs=0.0002)
    assert periods[0]["period_flag"] == 0
    assert periods[0]["chi_squared_ratio"] > 3000
    assert len(periods[0]["folded_profile"]["flux"]) == 100
    assert np.argmin(periods[0]["folded_profile"]["flux"]) == 0


def test_noise_does_not_produce_an_accepted_period():
    rng = np.random.default_rng(68)
    t = np.sort(rng.uniform(0, 10 * 86400, 1600))
    periods, _ = norton.detect_periods(
        t, 1000 + rng.normal(0, 8, len(t)), np.full(len(t), 8.0)
    )
    assert periods == []


def test_fortran_reference_with_documented_precision_and_extrema_repairs(tmp_path):
    reference = Path(__file__).resolve().parents[1] / "data" / "runfindper6.f"
    compiler = shutil.which("gfortran")
    if compiler is None or not reference.exists():
        pytest.skip(
            "Optional cross-language check needs gfortran and data/runfindper6.f"
        )
    source = reference.read_text()
    for old, new in [
        ("MAX1((TIMMAX),time(IA))", "MAX(TIMMAX,DBLE(time(IA)))"),
        ("MIN1((TIMMIN),time(IA))", "MIN(TIMMIN,DBLE(time(IA)))"),
        ("MAX1(TLRG,T(I))", "MAX(TLRG,DBLE(T(I)))"),
        ("MIN1(TSML,T(I))", "MIN(TSML,DBLE(T(I)))"),
    ]:
        assert old in source
        source = source.replace(old, new)
    source_path = tmp_path / "reference.f"
    source_path.write_text(source)
    executable = tmp_path / "reference"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=legacy",
            "-ffixed-line-length-none",
            "-mcmodel=medium",
            "-fdefault-real-8",
            "-fdefault-double-8",
            str(source_path),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    t, flux, error = synthetic_curve()
    np.savetxt(tmp_path / "synthetic.txt", np.c_[t, flux, error], fmt="%.10f")
    (tmp_path / "file.list").write_text("synthetic.txt\n")
    (tmp_path / "logfile.dat").touch()
    (tmp_path / "period_results.dat").touch()
    subprocess.run(
        [str(executable)], cwd=tmp_path, check=True, capture_output=True, timeout=60
    )
    fields = (tmp_path / "period_results.dat").read_text().split()
    assert len(fields) == 6
    periods, _ = norton.detect_periods(t, flux, error)
    actual = periods[0]
    assert actual["period_seconds"] == pytest.approx(float(fields[2]), abs=0.01)
    # np.polyfit solves OLS directly instead of the legacy early-stopped fit.
    assert actual["sigma"] == pytest.approx(float(fields[3]), abs=0.01)
    assert actual["chi_squared_ratio"] == pytest.approx(float(fields[4]), abs=0.02)
    assert actual["period_flag"] == int(fields[5])
