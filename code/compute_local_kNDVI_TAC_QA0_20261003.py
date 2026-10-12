"""Offline kNDVI lag-1 robust AR slopes, with explicit QA/date/estimator rules.

Run --preflight first; --full repeats the gate and computes the entire grid.
No Earth Engine connection or exports are made by this script.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from compare_new_lag1_TAC_20260928 import ROOT, np, rasterio, Window
from rasterio.transform import from_origin
from numba import njit, prange, set_num_threads
from threadpoolctl import threadpool_limits

OUT = ROOT / "output/kNDVI_TAC_local_QA0_20261003"
RASTERS = OUT / "rasters_2000_2024"
AUDIT = ROOT / "output/MOD13C1_local_input_audit_20261003/audit.json"
YEARS = np.arange(2005, 2026)
GRID = from_origin(-180, 80, .05, .05)


@njit(cache=True)
def line_fit(x, y, keep, n):
    count = 0
    sx = sy = sxx = sxy = 0.0
    for i in range(n):
        if keep[i]:
            count += 1
            sx += x[i]
            sy += y[i]
            sxx += x[i]*x[i]
            sxy += x[i]*y[i]
    denom = count*sxx-sx*sx
    if count <= 2 or denom <= 1e-22:
        return np.nan, np.nan
    slope = (count*sxy-sx*sy)/denom
    return (sy-slope*sx)/count, slope


@njit(cache=True)
def robust_line(x, y, n):
    keep = np.ones(n, np.bool_)
    intercept, slope = line_fit(x, y, keep, n)
    if not np.isfinite(slope):
        return np.nan
    errors = np.empty(n, np.float64)
    for i in range(n):
        errors[i] = abs(y[i]-intercept-slope*x[i])
    # Conformance-tested against GEE, not a claimed public server specification.
    beta = max(4*np.median(errors), 1e-14)
    for iteration in range(n):
        changed = False
        for i in range(n):
            if keep[i] and abs(y[i]-intercept-slope*x[i]) > beta:
                keep[i] = False
                changed = True
        if not changed:
            return slope
        intercept, slope = line_fit(x, y, keep, n)
        if not np.isfinite(slope):
            return np.nan
    return np.nan


@njit(cache=True, inline="never")
def solve_harmonics(gram, response):
    # Keep singular-system handling outside the parallel pixel loop.
    try:
        return np.linalg.solve(gram, response)
    except Exception:
        return np.full(8, np.nan)


@njit(cache=True, parallel=True)
def calculate(kndvi, design, outer, left, right, starts, ends):
    pixels, nt = kndvi.shape
    result = np.full((pixels, len(starts), 3), np.nan, np.float32)
    for p in prange(pixels):
        k = kndvi[p]
        total = 0
        for i in range(nt):
            if np.isfinite(k[i]):
                total += 1
        if total <= 8:
            continue
        sums = np.zeros(nt+1, np.float64)
        counts = np.zeros(nt+1, np.int32)
        for i in range(nt):
            sums[i+1] = sums[i]
            counts[i+1] = counts[i]
            if np.isfinite(k[i]):
                sums[i+1] += k[i]
                counts[i+1] += 1
        detrended = np.full(nt, np.nan, np.float32)
        gram = np.zeros((8,8), np.float64)
        response = np.zeros(8, np.float64)
        for i in range(nt):
            if not np.isfinite(k[i]):
                continue
            n = counts[right[i]]-counts[left[i]]
            mean = np.float32((sums[right[i]]-sums[left[i]])/n)
            detrended[i] = np.float32(k[i]-mean)
            for j in range(8):
                response[j] += design[i,j]*detrended[i]
                for q in range(8):
                    gram[j,q] += outer[i,j,q]
        coef = solve_harmonics(gram, response)
        if not np.all(np.isfinite(coef)):
            continue
        residual = np.full(nt, np.nan, np.float32)
        for i in range(nt):
            if np.isfinite(detrended[i]):
                fit = 0.0
                for j in range(8):
                    fit += design[i,j]*coef[j]
                residual[i] = np.float32(detrended[i]-fit)
        for a in range(len(starts)):
            start, end = starts[a], ends[a]
            observations = 0
            for i in range(start, end):
                if np.isfinite(residual[i]):
                    observations += 1
            x = np.empty(end-start, np.float64)
            y = np.empty(end-start, np.float64)
            pairs = 0
            for i in range(start, end-1):
                if np.isfinite(residual[i]) and np.isfinite(residual[i+1]):
                    x[pairs] = residual[i]
                    y[pairs] = residual[i+1]
                    pairs += 1
            if observations:
                result[p,a,1] = observations
            if pairs:
                result[p,a,2] = pairs
            if observations > 15 and pairs > 15:
                result[p,a,0] = np.float32(robust_line(x, y, pairs))
    return result


def configuration():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    packs = [p for p in audit["packs"] if p["original_submission_period"]]
    dates = np.array([d for p in packs for d in p["dates"]], dtype="datetime64[D]")
    assert len(dates) == 571 and dates[-1] < np.datetime64("2025-01-01")
    days = (dates-np.datetime64("1970-01-01")).astype(np.int64)
    t = (days.astype(float)/365.2425*(2*np.pi)).astype(np.float32)
    # Center/scale only the linear design column; the fitted function is unchanged.
    centered = (t.astype(float)-float(t.mean()))/(float(t.max())-float(t.min()))
    design = np.column_stack([np.ones(len(t)), centered]
        +[np.cos((t*j).astype(np.float32)).astype(float) for j in [1,2,3]]
        +[np.sin((t*j).astype(np.float32)).astype(float) for j in [1,2,3]])
    outer = np.einsum("ij,ik->ijk", design, design)
    left = np.searchsorted(days, days-912, side="left")
    right = np.searchsorted(days, days+912, side="left")
    starts = np.array([np.searchsorted(dates, np.datetime64(f"{y-5}-01-01")) for y in YEARS])
    ends = np.array([np.searchsorted(dates, np.datetime64(f"{y}-01-01")) for y in YEARS])
    return packs, dates, (design, outer, left, right, starts, ends)


def kndvi_from_source(ndvi, qa):
    good = (qa == 0) & np.isfinite(ndvi) & (ndvi != -9999)
    if np.any(good & ((ndvi < -.2001) | (ndvi > 1.0001))):
        raise ValueError("A QA0 NDVI value lies outside the product range")
    values = np.full(ndvi.shape, np.nan, np.float32)
    values[good] = np.tanh(ndvi[good].astype(float)**2).astype(np.float32)
    return values


def preflight(args_tuple):
    conformance = json.loads((OUT/"robust_independent_conformance.json").read_text(encoding="utf-8"))
    assert conformance["all_passed"], "Independent robust conformance evidence is missing"
    for case in conformance["checks"]:
        x, y = np.array(case["x"]), np.array(case["y"])
        assert abs(robust_line(x, y, len(x))-case["expected"][1]) < 1e-10
    sources = [np.load(OUT/f"point_{p}_stages.npz") for p in range(5)]
    values = np.vstack([kndvi_from_source(d["ndvi"], d["qa"]) for d in sources])
    results = calculate(values, *args_tuple)
    prototype = json.loads((OUT/"point_prototype.json").read_text(encoding="utf-8"))
    checks = []
    for row in prototype:
        if row["year_days"] != 365.2425:
            continue
        actual = results[row["point"], row["endpoint"]-2005].astype(float)
        expected = np.array(row["expected"])
        assert np.array_equal(actual[1:], expected[1:]), (row, actual)
        assert abs(actual[0]-expected[0]) < 2e-5, (row, actual)
        checks.append({"point": row["point"], "endpoint": row["endpoint"],
                       "local": actual.tolist(), "GEE_QA0": expected.tolist(),
                       "TAC_absolute_difference": float(abs(actual[0]-expected[0]))})
    report = {"passed": True, "independent_robust_test_cases": len(conformance["checks"]),
              "actual_point_window_checks": checks,
              "maximum_point_TAC_difference": max(c["TAC_absolute_difference"] for c in checks)}
    (OUT/"local_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: 12 robust tests; 10 actual point/window checks; max TAC difference={report['maximum_point_TAC_difference']:.3g}", flush=True)
    return report


def read_strip(packs, start, height, nt, workers):
    values = np.full((nt, height, 7200), np.nan, np.float32)
    offsets = np.cumsum([0]+[len(p["dates"]) for p in packs])

    def read_pack(job):
        index, pack = job
        count = len(pack["dates"])
        vi_bands = [3*i+1 for i in range(count)]
        qa_bands = [3*i+3 for i in range(count)]
        for tile in pack["tiles"]:
            path = Path(tile["path"])
            before = (path.stat().st_size, path.stat().st_mtime_ns)
            with rasterio.open(path) as f:
                window = Window(0, start, tile["width"], height)
                ndvi = f.read(vi_bands, window=window)
                qa = f.read(qa_bands, window=window)
            converted = kndvi_from_source(ndvi, qa)
            values[offsets[index]:offsets[index+1], :, tile["col"]:tile["col"]+tile["width"]] = converted
            assert before == (path.stat().st_size, path.stat().st_mtime_ns), "Input changed during reading"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(read_pack, enumerate(packs)))
    return np.ascontiguousarray(values.reshape(nt, -1).T)


def run_full(args, packs, dates, args_tuple, checked):
    RASTERS.mkdir(parents=True, exist_ok=True)
    paths = [RASTERS/f"KNDVI_TAC_lag1_5year_{year}_01_01_local_QA0.tif" for year in YEARS]
    assert not any(p.exists() for p in paths), "Final rasters exist; do not overwrite previous results"
    report = {"status": "running", "source_audit": str(AUDIT), "source_period": ["2000-01-01","2025-01-01"],
              "actual_first_date": str(dates[0]), "actual_last_date": str(dates[-1]), "image_count": len(dates),
              "qa": "SummaryQA == 0", "NDVI_scale_applied": 1, "kNDVI_formula": "tanh(NDVI**2)",
              "detrend_half_window_days": 912, "detrend_end_exclusive": True,
              "harmonic_order": 3, "harmonic_year_days": 365.2425,
              "lag_definition": "adjacent positions in the original dated collection, both observations valid; missing observations never removed",
              "TAC_estimator": "Talwar robust AR(1) slope with intercept",
              "robust_local_rule": "fixed beta=4*median(abs(initial OLS residual)); permanently remove excluded observations until retained mask unchanged; numerical beta floor 1e-14",
              "conformance_note": "Local rule is empirically tested against GEE; it is not a published specification or proof of identity at every pixel.",
              "validity": "observation_count > 15 and pair_count > 15; no truncation of AR slopes to [-1,1]",
              "grid": [2800,7200], "transform": list(GRID), "threads": args.threads, "io_workers": args.io_workers,
              "preflight": checked, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "endpoint_2025_note": "Window 2020-2024; edge sensitivity, not silently included in original primary trend analysis",
              "completed_strips": [], "outputs": [str(p) for p in paths]}
    report_path = OUT/"local_run.json"

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    profile = dict(driver="GTiff", width=7200, height=2800, count=3, dtype="float32", crs="EPSG:4326",
                   transform=GRID, nodata=-9999, tiled=True, blockxsize=256, blockysize=256,
                   compress="deflate", predictor=3, BIGTIFF="IF_SAFER", NUM_THREADS="2", interleave="band")
    quality = [dict(endpoint_year=int(y), window_start=f"{y-5}-01-01", window_end_exclusive=f"{y}-01-01",
                    valid_TAC=0, TAC_min=None, TAC_max=None, TAC_outside_minus1_plus1=0,
                    valid_TAC_with_invalid_counts=0, pairs_exceed_observations=0, nonfinite_nonmissing=0) for y in YEARS]
    started = time.monotonic()
    try:
        with rasterio.Env(GDAL_CACHEMAX=1024**3), ExitStack() as stack:
            writers = [stack.enter_context(rasterio.open(str(p)+".partial.tif", "w", **profile)) for p in paths]
            for year, writer in zip(YEARS, writers):
                for band, name in enumerate(["TAC_lag1","observation_count","pair_count"], 1):
                    writer.set_band_description(band, name)
                writer.update_tags(qa="SummaryQA=0", source_period="2000-2024", window_start=f"{year-5}-01-01",
                    window_end_exclusive=f"{year}-01-01", estimator="Talwar robust AR(1) slope with intercept",
                    lag="1", no_extra_NDVI_scaling="true", endpoint_year=str(year))
            for start in range(0, 2800, args.strip_height):
                height = min(args.strip_height, 2800-start)
                tick = time.monotonic()
                kndvi = read_strip(packs, start, height, len(dates), args.io_workers)
                read_seconds = time.monotonic()-tick
                result = calculate(kndvi, *args_tuple)
                del kndvi
                calc_seconds = time.monotonic()-tick-read_seconds
                for j, writer in enumerate(writers):
                    a = np.ascontiguousarray(result[:,j,:].reshape(height,7200,3).transpose(2,0,1))
                    tac, obs, pairs = a
                    valid = np.isfinite(tac)
                    q = quality[j]
                    q["valid_TAC"] += int(valid.sum())
                    q["TAC_outside_minus1_plus1"] += int(np.count_nonzero(valid & (np.abs(tac)>1)))
                    q["valid_TAC_with_invalid_counts"] += int(np.count_nonzero(valid & (~np.isfinite(obs) | ~np.isfinite(pairs) | (obs<=15) | (pairs<=15))))
                    q["pairs_exceed_observations"] += int(np.count_nonzero(np.isfinite(obs) & np.isfinite(pairs) & (pairs>obs)))
                    q["nonfinite_nonmissing"] += int(np.count_nonzero(np.isinf(a)))
                    if valid.any():
                        low, high = float(tac[valid].min()), float(tac[valid].max())
                        q["TAC_min"] = low if q["TAC_min"] is None else min(q["TAC_min"], low)
                        q["TAC_max"] = high if q["TAC_max"] is None else max(q["TAC_max"], high)
                    assert not q["valid_TAC_with_invalid_counts"] and not q["pairs_exceed_observations"] and not q["nonfinite_nonmissing"]
                    a[~np.isfinite(a)] = -9999
                    writer.write(a, window=Window(0,start,7200,height))
                del result
                report["quality_so_far"] = quality
                report["completed_strips"].append({"row":start,"height":height,"read_seconds":read_seconds,"calculation_seconds":calc_seconds})
                report["elapsed_seconds"] = time.monotonic()-started
                save()
                print(f"Rows {start+height}/2800; read={read_seconds:.1f}s compute={calc_seconds:.1f}s elapsed={report['elapsed_seconds']:.1f}s", flush=True)
        for path in paths:
            Path(str(path)+".partial.tif").replace(path)
            with rasterio.open(path) as f:
                assert f.shape == (2800,7200) and f.count == 3 and f.transform.almost_equals(GRID)
                assert f.descriptions == ("TAC_lag1","observation_count","pair_count")
        report.update(status="completed", quality=quality, elapsed_seconds=time.monotonic()-started)
        save()
        print(f"COMPLETED: 21 local TAC rasters; {report['elapsed_seconds']:.1f}s; {RASTERS}", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error), elapsed_seconds=time.monotonic()-started)
        save()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--strip-height", type=int, default=256)
    args = parser.parse_args()
    if not (args.preflight or args.full):
        print("No computation; use --preflight or --full.")
        return
    set_num_threads(args.threads)
    packs, dates, args_tuple = configuration()
    with threadpool_limits(limits=1, user_api="blas"):
        checked = preflight(args_tuple)
        if args.full:
            run_full(args, packs, dates, args_tuple, checked)


if __name__ == "__main__":
    main()
