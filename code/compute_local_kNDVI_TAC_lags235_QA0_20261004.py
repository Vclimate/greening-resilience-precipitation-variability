"""Offline lag-2/3/5 sensitivity with the validated lag-1 preprocessing.

The source is read once per strip and the residual series is shared by all lags.
No Earth Engine connection or export tasks are created.
"""
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import compute_local_kNDVI_TAC_QA0_20261003 as core
from numba import njit, prange, set_num_threads
from threadpoolctl import threadpool_limits

np, rasterio, Window = core.np, core.rasterio, core.Window
ROOT, GRID, YEARS = core.ROOT, core.GRID, core.YEARS
LAGS = np.array([2, 3, 5], dtype=np.int64)
OUT = ROOT / "output/kNDVI_TAC_local_QA0_lag2_3_5_20261004"
BASE_RUN = core.OUT / "local_run.json"
GEE_SMOKE = ROOT / "output/MOD13C1_TAC_corrected_20261001/run_20261001_145248.json"


@njit(cache=True)
def lag_window(residual, start, end, lags):
    result = np.full((len(lags), 3), np.nan, np.float32)
    observations = 0
    for i in range(start, end):
        if np.isfinite(residual[i]):
            observations += 1
    x = np.empty(end-start, np.float64)
    y = np.empty(end-start, np.float64)
    for j in range(len(lags)):
        lag = lags[j]
        pairs = 0
        for i in range(start, end-lag):
            if np.isfinite(residual[i]) and np.isfinite(residual[i+lag]):
                x[pairs] = residual[i]
                y[pairs] = residual[i+lag]
                pairs += 1
        if observations:
            result[j,1] = observations
        if pairs:
            result[j,2] = pairs
        if observations > 15 and pairs > 15:
            result[j,0] = np.float32(core.robust_line(x, y, pairs))
    return result


@njit(cache=True, parallel=True)
def calculate_lags(kndvi, design, outer, left, right, starts, ends, lags):
    pixels, nt = kndvi.shape
    result = np.full((pixels, len(starts), len(lags), 3), np.nan, np.float32)
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
        coef = core.solve_harmonics(gram, response)
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
            result[p,a] = lag_window(residual, starts[a], ends[a], lags)
    return result


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def preflight(args_tuple):
    original = json.loads(BASE_RUN.read_text(encoding="utf-8"))
    core_hash = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    assert original["status"] == "completed" and core_hash == original["script_sha256"], "The validated baseline implementation changed"
    all_lags = np.array([1,2,3,5], dtype=np.int64)
    tests = []
    for scenario in ["complete", "missing_every_third", "alternating", "cropped_window"]:
        residual = (.98**np.arange(64)).astype(np.float32)
        start, end = 0, 64
        if scenario == "missing_every_third":
            residual[::3] = np.nan
        elif scenario == "alternating":
            residual[::2] = np.nan
        elif scenario == "cropped_window":
            start, end = 10, 44
        actual = lag_window(residual, start, end, all_lags)
        observations = int(np.isfinite(residual[start:end]).sum())
        for j, lag in enumerate(all_lags):
            paired = np.isfinite(residual[start:end-lag]) & np.isfinite(residual[start+lag:end])
            count = int(paired.sum())
            assert actual[j,1] == observations
            if count:
                assert actual[j,2] == count
            else:
                assert np.isnan(actual[j,2])
            if observations > 15 and count > 15:
                assert abs(actual[j,0]-.98**lag) < 1e-6
            else:
                assert np.isnan(actual[j,0])
            tests.append(dict(scenario=scenario, lag=int(lag), observations=observations, pairs=count))
    with ExitStack() as stack:
        sources = [stack.enter_context(np.load(core.OUT/f"point_{p}_stages.npz")) for p in range(5)]
        values = np.vstack([core.kndvi_from_source(d["ndvi"], d["qa"]) for d in sources])
    actual = calculate_lags(values, *args_tuple, all_lags)
    baseline = core.calculate(values, *args_tuple)
    np.testing.assert_allclose(actual[:,:,0,:], baseline, rtol=0, atol=0, equal_nan=True)
    smoke = json.loads(GEE_SMOKE.read_text(encoding="utf-8"))
    checks = []
    for row in smoke["smoke_tests"]:
        if row["lag"] not in LAGS or row["window"] != 5 or row["endpoint"] != 2008:
            continue
        lag = row["lag"]
        position = int(np.flatnonzero(all_lags == lag)[0])
        for expected in row["samples"]:
            point = expected["point_id"]
            local = actual[point,2008-2005,position,:].astype(float)
            target = np.array([expected[f"TAC_lag{lag}"],expected["observation_count"],expected["pair_count"]])
            assert np.array_equal(local[1:], target[1:]), (lag, point, local, target)
            error = float(abs(local[0]-target[0]))
            assert error < 1e-4, (lag, point, local, target)
            checks.append(dict(lag=int(lag),point=point,endpoint=2008,local=local.tolist(),GEE=target.tolist(),TAC_absolute_difference=error))
    assert len(checks) == 9
    report = dict(passed=True, synthetic_checks=tests, baseline_lag1_exact_match=True,
                  actual_GEE_lag_checks=checks, maximum_GEE_point_difference=max(c["TAC_absolute_difference"] for c in checks),
                  shared_core_sha256=core_hash)
    save(OUT/"preflight.json",report)
    print(f"PASS: 16 pairing/window tests; lag1 exactly reproduced; 9 GEE lag checks; max difference={report['maximum_GEE_point_difference']:.3g}",flush=True)
    return report


def read_strip(packs, start, height, nt, workers, initial_stamps):
    values = np.full((nt,height,7200), np.nan, np.float32)
    offsets = np.cumsum([0]+[len(p["dates"]) for p in packs])

    def worker(job):
        index, pack = job
        indices = [3*i+band for i in range(len(pack["dates"])) for band in [1,3]]
        for tile in pack["tiles"]:
            path = Path(tile["path"])
            before = (path.stat().st_size,path.stat().st_mtime_ns)
            assert before == initial_stamps[str(path)], "Source changed after job start"
            with rasterio.open(path) as f:
                assert f.crs.to_epsg() == 4326 and f.nodata == -9999
                data = f.read(indices, window=Window(0,start,tile["width"],height))
            converted = core.kndvi_from_source(data[0::2], data[1::2])
            values[offsets[index]:offsets[index+1],:,tile["col"]:tile["col"]+tile["width"]] = converted
            assert before == (path.stat().st_size,path.stat().st_mtime_ns), "Source changed during reading"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, enumerate(packs)))
    return np.ascontiguousarray(values.reshape(nt,-1).T)


def run_full(args, packs, dates, args_tuple, checked):
    paths = {(int(lag),int(year)): OUT/f"lag{lag}/rasters_2000_2024/KNDVI_TAC_lag{lag}_5year_{year}_01_01_local_QA0.tif"
             for lag in LAGS for year in YEARS}
    assert not any(path.exists() for path in paths.values()), "Final results already exist; refusing to overwrite"
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    initial_stamps = {t["path"]:(Path(t["path"]).stat().st_size,Path(t["path"]).stat().st_mtime_ns) for p in packs for t in p["tiles"]}
    base = json.loads(BASE_RUN.read_text(encoding="utf-8"))
    report = {k:base[k] for k in ["source_audit","source_period","actual_first_date","actual_last_date","image_count","qa",
              "NDVI_scale_applied","kNDVI_formula","detrend_half_window_days","detrend_end_exclusive","harmonic_order",
              "harmonic_year_days","robust_local_rule","conformance_note","validity","grid","transform","endpoint_2025_note"]}
    report.update(status="running", lags=LAGS.tolist(), planned_files=63, threads=args.threads,io_workers=args.io_workers,
                  lag_definition="Original collection positions i and i+lag, both valid and both inside the same window; never delete missing dates",
                  estimator="Talwar robust lagged slope with intercept; not a multivariate AR(lag) model or Pearson correlation",
                  source_stamps=initial_stamps, preflight=checked,baseline_lag1_run=str(BASE_RUN),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  completed_strips=[],outputs=[str(path) for path in paths.values()])
    quality = {key:dict(lag=key[0],endpoint_year=key[1],window_start=f"{key[1]-5}-01-01",window_end_exclusive=f"{key[1]}-01-01",
                valid_TAC=0,TAC_min=None,TAC_max=None,TAC_outside_minus1_plus1=0,
                valid_TAC_with_invalid_counts=0,pairs_exceed_observations=0,nonfinite_nonmissing=0) for key in paths}
    report_path = OUT/"local_run.json"
    save(report_path,report)
    profile = dict(driver="GTiff",width=7200,height=2800,count=3,dtype="float32",crs="EPSG:4326",transform=GRID,
                   nodata=-9999,tiled=True,blockxsize=256,blockysize=256,compress="deflate",predictor=3,
                   BIGTIFF="IF_SAFER",NUM_THREADS="2",interleave="band")
    started = time.monotonic()
    try:
        with rasterio.Env(GDAL_CACHEMAX=1024**3), ExitStack() as stack:
            writers = {key:stack.enter_context(rasterio.open(str(path)+".partial.tif","w",**profile)) for key,path in paths.items()}
            for (lag,year), writer in writers.items():
                for band,name in enumerate([f"TAC_lag{lag}","observation_count","pair_count"],1):
                    writer.set_band_description(band,name)
                writer.update_tags(qa="SummaryQA=0",source_period="2000-2024",window_start=f"{year-5}-01-01",
                    window_end_exclusive=f"{year}-01-01",lag=str(lag),no_extra_NDVI_scaling="true",
                    estimator="Talwar robust lagged slope with intercept",endpoint_year=str(year))
            for start in range(0,2800,args.strip_height):
                height = min(args.strip_height,2800-start)
                tick = time.monotonic()
                kndvi = read_strip(packs,start,height,len(dates),args.io_workers,initial_stamps)
                read_seconds = time.monotonic()-tick
                this_lags = np.array([1,2,3,5],dtype=np.int64) if start == 0 else LAGS
                results = calculate_lags(kndvi,*args_tuple,this_lags)
                del kndvi
                calc_seconds = time.monotonic()-tick-read_seconds
                if start == 0:
                    check = results[:,2006-2005,0,:].reshape(height,7200,3).transpose(2,0,1)
                    reference = core.RASTERS/"KNDVI_TAC_lag1_5year_2006_01_01_local_QA0.tif"
                    with rasterio.open(reference) as f:
                        original = f.read(window=Window(0,start,7200,height))
                    original[original == -9999] = np.nan
                    np.testing.assert_allclose(check,original,rtol=0,atol=1e-6,equal_nan=True)
                    report["first_strip_lag1_anchor"] = "passed; modified joint reader and kernel agree with previous local output"
                for j,year in enumerate(YEARS):
                    for lag in LAGS:
                        position = int(np.flatnonzero(this_lags == lag)[0])
                        key = (int(lag),int(year))
                        a = np.ascontiguousarray(results[:,j,position,:].reshape(height,7200,3).transpose(2,0,1))
                        tac,obs,pairs = a
                        good = np.isfinite(tac)
                        q = quality[key]
                        q["valid_TAC"] += int(good.sum())
                        q["TAC_outside_minus1_plus1"] += int(np.count_nonzero(good & (np.abs(tac)>1)))
                        q["valid_TAC_with_invalid_counts"] += int(np.count_nonzero(good & (~np.isfinite(obs) | ~np.isfinite(pairs) | (obs<=15) | (pairs<=15))))
                        q["pairs_exceed_observations"] += int(np.count_nonzero(np.isfinite(obs) & np.isfinite(pairs) & (pairs>obs)))
                        q["nonfinite_nonmissing"] += int(np.count_nonzero(np.isinf(a)))
                        if good.any():
                            low,high = float(tac[good].min()),float(tac[good].max())
                            q["TAC_min"] = low if q["TAC_min"] is None else min(q["TAC_min"],low)
                            q["TAC_max"] = high if q["TAC_max"] is None else max(q["TAC_max"],high)
                        assert not q["valid_TAC_with_invalid_counts"] and not q["pairs_exceed_observations"] and not q["nonfinite_nonmissing"]
                        a[~np.isfinite(a)] = -9999
                        writers[key].write(a,window=Window(0,start,7200,height))
                del results
                report["completed_strips"].append(dict(row=start,height=height,read_seconds=read_seconds,calculation_seconds=calc_seconds))
                report["quality_so_far"] = list(quality.values())
                report["elapsed_seconds"] = time.monotonic()-started
                save(report_path,report)
                print(f"Rows {start+height}/2800; read={read_seconds:.1f}s compute={calc_seconds:.1f}s elapsed={report['elapsed_seconds']:.1f}s; all three lags",flush=True)
        for (lag,year),path in paths.items():
            Path(str(path)+".partial.tif").replace(path)
            with rasterio.open(path) as f:
                assert f.shape == (2800,7200) and f.count == 3 and f.crs.to_epsg() == 4326
                assert f.transform.almost_equals(GRID) and f.descriptions == (f"TAC_lag{lag}","observation_count","pair_count")
                assert f.nodata == -9999 and set(f.dtypes) == {"float32"}
        report.update(status="completed",quality=list(quality.values()),elapsed_seconds=time.monotonic()-started)
        save(report_path,report)
        with (OUT/"quality_summary.csv").open("w",newline="",encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(next(iter(quality.values()))))
            writer.writeheader()
            writer.writerows(quality.values())
        print(f"COMPLETED: 63 rasters, lags 2/3/5; {report['elapsed_seconds']:.1f}s; {OUT}",flush=True)
    except Exception as error:
        report.update(status="failed",error=str(error),elapsed_seconds=time.monotonic()-started)
        save(report_path,report)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--full",action="store_true")
    parser.add_argument("--threads",type=int,default=24)
    parser.add_argument("--io-workers",type=int,default=4)
    parser.add_argument("--strip-height",type=int,default=256)
    args = parser.parse_args()
    if not (args.preflight or args.full):
        print("No computation; use --preflight or --full.")
        return
    set_num_threads(args.threads)
    packs,dates,args_tuple = core.configuration()
    with threadpool_limits(limits=1,user_api="blas"):
        checked = preflight(args_tuple)
        if args.full:
            run_full(args,packs,dates,args_tuple,checked)


if __name__ == "__main__":
    main()
