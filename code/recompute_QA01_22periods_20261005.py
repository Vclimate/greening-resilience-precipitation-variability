"""Unified 2000-2025 preprocessing, 22 TAC endpoints, and 19-endpoint RF inputs.

All previous outputs are preserved. The lag definition remains collection-index
lag; globally absent composites are recorded, never silently synthesized.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import run_RF_QA01_reanalysis_20261005 as old
from numba import njit, prange, set_num_threads
from threadpoolctl import threadpool_limits

prep, core, rf = old.prep, old.prep.core, old.rf
np, pd, rasterio, Window = prep.np, prep.pd, prep.rasterio, prep.Window
OUT = prep.ROOT / 'output/QA01_unified22_20261005'
DATA = OUT / 'data'
MODELS = OUT / 'RF_endpoints_2005_2023_QA01_repeated30'
YEARS = np.arange(2005, 2027)
CONFIGS = [(5, 1), (5, 2), (5, 3), (5, 5), (7, 1), (10, 1)]


def configuration():
    audit = rf.read(core.AUDIT)
    assert audit['status'] == 'passed'
    packs = [p for p in audit['packs'] if p['year'] <= 2025]
    dates = np.array([d for p in packs for d in p['dates']], dtype='datetime64[D]')
    assert np.all(dates[1:] > dates[:-1]) and dates[-1] < np.datetime64('2026-01-01')
    assert len(np.unique(dates.astype('datetime64[Y]'))) == 26
    days = (dates - np.datetime64('1970-01-01')).astype('int64')
    t = (days / 365.2425 * (2*np.pi)).astype(np.float32)
    centered = (t.astype(float)-float(t.mean())) / (float(t.max())-float(t.min()))
    design = np.column_stack([np.ones(len(t)), centered] +
        [np.cos((t*j).astype(np.float32)).astype(float) for j in [1, 2, 3]] +
        [np.sin((t*j).astype(np.float32)).astype(float) for j in [1, 2, 3]])
    outer = np.einsum('ij,ik->ijk', design, design)
    left, right = np.searchsorted(days, days-912), np.searchsorted(days, days+912)
    starts = np.array([[np.searchsorted(dates, np.datetime64(f'{y-w}-01-01')) for y in YEARS]
                       for w, lag in CONFIGS])
    ends = np.array([np.searchsorted(dates, np.datetime64(f'{y}-01-01')) for y in YEARS])
    return packs, dates, (design, outer, left, right, starts, ends)


@njit(cache=True, parallel=True)
def calculate(values, design, outer, left, right, starts, ends, lags, minimum_endpoints):
    pixels, nt = values.shape
    out = np.full((pixels, len(lags), len(ends), 3), np.nan, np.float32)
    for p in prange(pixels):
        k = values[p]
        sums = np.zeros(nt+1, np.float64)
        counts = np.zeros(nt+1, np.int32)
        for i in range(nt):
            sums[i+1] = sums[i]
            counts[i+1] = counts[i]
            if np.isfinite(k[i]):
                sums[i+1] += k[i]
                counts[i+1] += 1
        if counts[-1] <= 8:
            continue
        dt = np.full(nt, np.nan, np.float32)
        gram, response = np.zeros((8, 8)), np.zeros(8)
        for i in range(nt):
            if np.isfinite(k[i]):
                n = counts[right[i]]-counts[left[i]]
                mean = np.float32((sums[right[i]]-sums[left[i]]) / n)
                dt[i] = np.float32(k[i]-mean)
                for j in range(8):
                    response[j] += design[i, j]*dt[i]
                    for q in range(8):
                        gram[j, q] += outer[i, j, q]
        coef = core.solve_harmonics(gram, response)
        if not np.all(np.isfinite(coef)):
            continue
        residual = np.full(nt, np.nan, np.float32)
        for i in range(nt):
            if np.isfinite(dt[i]):
                fitted = 0.0
                for j in range(8):
                    fitted += design[i, j]*coef[j]
                residual[i] = np.float32(dt[i]-fitted)
        x, y = np.empty(nt), np.empty(nt)
        for c in range(len(lags)):
            lag = lags[c]
            for a in range(len(ends)):
                if a < minimum_endpoints[c]:
                    continue
                start, end = starts[c, a], ends[a]
                obs, pairs = 0, 0
                for i in range(start, end):
                    if np.isfinite(residual[i]):
                        obs += 1
                for i in range(start, end-lag):
                    if np.isfinite(residual[i]) and np.isfinite(residual[i+lag]):
                        x[pairs], y[pairs] = residual[i], residual[i+lag]
                        pairs += 1
                if obs:
                    out[p, c, a, 1] = obs
                if pairs:
                    out[p, c, a, 2] = pairs
                if obs > 15 and pairs > 15:
                    out[p, c, a, 0] = core.robust_line(x, y, pairs)
    return out


def read_strip(packs, start, height, variant, stamps):
    nt = sum(len(p['dates']) for p in packs)
    values = np.full((nt, height, 7200), np.nan, np.float32)
    offsets = np.cumsum([0]+[len(p['dates']) for p in packs])
    def worker(job):
        index, pack = job
        vb = 1 if variant == 'kNDVI' else 2
        bands = [3*i+b for i in range(len(pack['dates'])) for b in [vb, 3]]
        for tile in pack['tiles']:
            path = Path(tile['path'])
            assert (path.stat().st_size, path.stat().st_mtime_ns) == stamps[str(path)]
            with rasterio.open(path) as src:
                a = src.read(bands, window=Window(0, start, tile['width'], height))
            vi, qa = a[0::2], a[1::2]
            if variant == 'kNDVI':
                converted = old.qa01_values(vi, qa)
            else:
                good = ((qa == 0) | (qa == 1)) & np.isfinite(vi) & (vi != -9999)
                assert not np.any(good & ((vi < -.2001) | (vi > 1.0001)))
                converted = np.where(good, vi, np.nan).astype(np.float32)
            values[offsets[index]:offsets[index+1], :, tile['col']:tile['col']+tile['width']] = converted
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, enumerate(packs)))
    return values


def preflight(args):
    design, outer, left, right, starts, ends = args
    rng = np.random.default_rng(20261005)
    k = rng.uniform(.05, .65, (6, len(design))).astype(np.float32)
    k[1, ::3] = np.nan
    k[2, ::2] = np.nan
    reference = core.calculate(k, design, outer, left, right, starts[0], ends)
    result = calculate(k, *args, np.array([lag for w, lag in CONFIGS]),
                       np.array([max(0, w-5) for w, lag in CONFIGS]))
    np.testing.assert_array_equal(result[:, 0], reference)
    assert np.isnan(result[:, 4, :2]).all() and np.isnan(result[:, 5, :5]).all()
    evidence = rf.read(core.OUT/'robust_independent_conformance.json')
    assert evidence['all_passed']
    prep.dump(OUT/'preflight.json', dict(passed=True, same_lag1_as_validated_core=True,
                                        robust_cases=len(evidence['checks']), endpoints=YEARS.tolist()))


def rebuild(variant, base, threads, strip_height, repair_through=0):
    folder = DATA / variant
    done = folder / 'complete.json'
    if done.exists() and not repair_through:
        record = rf.read(done)
        assert record['status'] == 'completed'
        assert all(Path(p).exists() and Path(p).stat().st_size == stamp['size'] for p, stamp in record['outputs'].items())
        return record
    packs, dates, args = configuration()
    configs = CONFIGS if variant == 'kNDVI' else [(5, 1)]
    stamps = {t['path']: (Path(t['path']).stat().st_size, Path(t['path']).stat().st_mtime_ns)
              for p in packs for t in p['tiles']}
    paths = [(folder/f'lag{lag}_{w}year' / f'{variant}_TAC_lag{lag}_{w}year_{y}_01_01_QA01.tif', c, a)
             for c, (w, lag) in enumerate(configs) for a, y in enumerate(YEARS) if y >= 2000+w]
    green, annual_path = folder/'greenness_2000_2025.tif', folder/'annual_means_2000_2025.tif'
    response = folder/'TAC_trend_RF_endpoints_2005_2023.tif'
    all_paths = [green, annual_path, response] + [p for p, c, a in paths]
    if repair_through:
        assert variant == 'kNDVI' and done.exists() and all(p.exists() for p in all_paths)
        assert repair_through == 896, 'Repair scope is established by the independent validity audit'
        prep.dump(folder/'interrupted_before_repair.json', rf.read(done))
    else:
        assert not any(p.exists() for p in all_paths), 'Completed outputs already exist without marker'
    partials = [Path(str(p)+'.partial.tif') for p in all_paths]
    resume = 0
    previous_elapsed = 0.0
    if any(p.exists() for p in partials):
        assert all(p.exists() for p in partials), 'Incomplete writer set; inspect before resuming'
        assert (folder/'partial_resume_verified.json').exists(), 'Interrupted raster caches require an integrity audit before resuming'
        state = rf.read(folder/'status.json')
        assert state['status'] == 'running'
        resume = int(state['completed_rows'])
        previous_elapsed = float(state['elapsed_seconds'])
        assert 0 <= resume <= 2800
    for p in all_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
    set_num_threads(threads)
    date_year = dates.astype('datetime64[Y]').astype(int)+1970
    annual_years = np.arange(2000, 2026)
    lags = np.array([lag for w, lag in configs])
    minimum = np.array([max(0, w-5) for w, lag in configs])
    started = time.monotonic()-previous_elapsed
    with ExitStack() as stack, threadpool_limits(limits=1), rasterio.Env(GDAL_CACHEMAX=512*1024**2):
        writers = [stack.enter_context(rasterio.open(str(p) if repair_through else str(p)+'.partial.tif',
                                                   'r+' if resume or repair_through else 'w',
                                                   **({} if resume or repair_through else prep.profile(count))))
                   for p, count in zip(all_paths, [6, 26, 3]+[3]*len(paths))]
        for i, name in enumerate([f'{variant}_mean_2000_2023', f'{variant}_trend_2000_2023',
                                 f'{variant}_mean_2000_2025', f'{variant}_trend_2000_2025',
                                 'QA01_missing_fraction_2000_2023', 'QA01_missing_fraction_2000_2025'], 1):
            writers[0].set_band_description(i, name)
        for i, y in enumerate(annual_years, 1):
            writers[1].set_band_description(i, f'{variant}_mean_{y}')
        for i, name in enumerate(['delta_TAC', 'OLS_pvalue', 'valid_window_count'], 1):
            writers[2].set_band_description(i, name)
        for writer, (path, c, a) in zip(writers[3:], paths):
            w, lag = configs[c]
            for i, name in enumerate([f'TAC_lag{lag}', 'observation_count', 'pair_count'], 1):
                writer.set_band_description(i, name)
            writer.update_tags(window_start=f'{YEARS[a]-w}-01-01', window_end_exclusive=f'{YEARS[a]}-01-01',
                               reference_year=str(YEARS[a]-1), lag_definition='original collection positions')
        for writer in writers:
            writer.update_tags(qa='SummaryQA in [0,1]', preprocessing_period='2000-2025',
                               detrend_half_window_days='912', harmonic_order='3',
                               domain='Original stable vegetation RF base; legacy MDN<0.5; new MDN applied in analyses')
        block_starts = sorted(set([resume]+list(range(((resume+strip_height-1)//strip_height)*strip_height, 2800, strip_height))))
        block_starts = [s for s in block_starts if s < (repair_through or 2800)]
        for start in block_starts:
            height = min(strip_height-start%strip_height, (repair_through or 2800)-start)
            selected = base[start:start+height].ravel()
            window = Window(0, start, 7200, height)
            targets = [np.full((count, height*7200), -9999, np.float32)
                       for count in [6, 26, 3]+[3]*len(paths)]
            if selected.any():
                full = read_strip(packs, start, height, variant, stamps)
                values = np.ascontiguousarray(full.reshape(len(dates), -1)[:, selected].T)
                del full
                annual = np.full((26, len(values)), np.nan)
                for j, y in enumerate(annual_years):
                    a = values[:, date_year == y]
                    count = np.isfinite(a).sum(axis=1)
                    np.divide(np.nansum(a, axis=1, dtype=float), count, out=annual[j], where=count > 0)
                green_bands = []
                for last in [2023, 2025]:
                    a = annual[:last-1999]
                    count = np.isfinite(a).sum(axis=0)
                    mean = np.full(len(values), np.nan)
                    np.divide(np.nansum(a, axis=0), count, out=mean, where=count > 0)
                    slope, _, _ = prep.fit_pixels(a, annual_years[:last-1999])
                    green_bands.extend([mean, slope])
                for last in [2023, 2025]:
                    pos = date_year <= last
                    green_bands.append(1 - np.isfinite(values[:, pos]).sum(axis=1)/int(pos.sum()))
                targets[0][:, selected] = np.stack(green_bands).astype(np.float32)
                targets[1][:, selected] = annual.astype(np.float32)
                derived = calculate(values, *args, lags, minimum)
                assert (np.nan_to_num(derived[:, :, :, 2]) <= np.nan_to_num(derived[:, :, :, 1])).all()
                tac = derived[:, 0, :19, 0].T.copy()
                tac[np.abs(tac) > 1] = np.nan
                slope, p, n = prep.fit_pixels(tac, YEARS[:19])
                targets[2][:, selected] = np.stack([slope, p, n]).astype(np.float32)
                for target, (path, c, a) in zip(targets[3:], paths):
                    target[:, selected] = derived[:, c, a, :].T
                del values, annual, derived
            for writer, target in zip(writers, targets):
                target[~np.isfinite(target)] = -9999
                writer.write(target.reshape(target.shape[0], height, 7200), window=window)
            prep.dump(folder/'status.json', dict(status='running', completed_rows=start+height,
                                               total_rows=2800, elapsed_seconds=time.monotonic()-started))
            prep.log(f'{variant} unified22 rows {start+height}/2800; {(time.monotonic()-started)/60:.1f} min')
    if not repair_through:
        for p in all_paths:
            Path(str(p)+'.partial.tif').replace(p)
    if variant == 'kNDVI':
        # These quantities must be unchanged by extending harmonic fitting to 2025.
        with rasterio.open(old.DATA/'greenness_QA01_2000_2024.tif') as before, rasterio.open(green) as after:
            for row in range(0, 2800, 256):
                window = Window(0, row, 7200, min(256, 2800-row))
                np.testing.assert_array_equal(before.read([1, 2, 5], window=window),
                                              after.read([1, 2, 5], window=window))
    gaps = np.diff(dates).astype('timedelta64[D]').astype(int)
    record = dict(status='completed', variant=variant, source_dates=[str(dates[0]), str(dates[-1])],
                  image_count=len(dates), source_stamps=stamps, preprocessing_period='2000-2025',
                  lag_definition='original collection positions, as in frozen method',
                  gaps_over16=[dict(first=str(dates[i]), second=str(dates[i+1])) for i in np.flatnonzero(gaps > 16)],
                  TAC_endpoints=YEARS.tolist(), TAC_reference_years=(YEARS-1).tolist(),
                  RF_endpoints=YEARS[:19].tolist(), total_domain_pixels=int(base.sum()),
                  outputs={str(p): dict(sha256=rf.digest(p), size=p.stat().st_size) for p in all_paths},
                  elapsed_seconds=time.monotonic()-started, script_sha256=rf.digest(__file__),
                  full_grid_same_2000_2023_greenness_and_missingness_as_previous_QA01=variant == 'kNDVI',
                  repaired_interrupted_writer_rows=repair_through)
    prep.dump(done, record)
    return record


def rf_inputs(manifest, support, lc, missing, base):
    old.OUT, old.DATA, old.MODELS = OUT, DATA, MODELS
    table, record = old.tables(manifest, support, lc, missing, base,
                              DATA/'kNDVI/greenness_2000_2025.tif',
                              DATA/'kNDVI/TAC_trend_RF_endpoints_2005_2023.tif')
    record.update(preprocessing='Unified 2000-2025 rolling detrend/three harmonics; 22 endpoints; RF first19',
                  raster_evidence=str(DATA/'kNDVI/complete.json'), unified22=True)
    prep.dump(DATA/'source_manifest.json', record)
    return table, record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['kNDVI', 'EVI', 'all'], default='all')
    p.add_argument('--threads', type=int, default=24)
    p.add_argument('--strip-height', type=int, default=256)
    p.add_argument('--repair-through-row', type=int, default=0)
    args = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    manifest, support, lc, missing, base = old.masks()
    packs, dates, config = configuration()
    preflight(config)
    variants = ['kNDVI', 'EVI'] if args.variant == 'all' else [args.variant]
    for variant in variants:
        rebuild(variant, base, args.threads, args.strip_height, args.repair_through_row)
        if variant == 'kNDVI':
            rf_inputs(manifest, support, lc, missing, base)
    prep.log('Unified22 requested rasters and inputs COMPLETED')


if __name__ == '__main__':
    main()
