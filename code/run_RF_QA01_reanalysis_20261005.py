"""QA=0/1 sensitivity: rebuild local lag1 response/greenness and repeat RF.

Spatial domain and all non-vegetation predictors remain those of the QA0 RF.
No QA0 outputs are modified, and no Earth Engine requests are made.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import time

import prepare_RF_local_QA0_20261004 as prep
import run_RF_local_QA0_repeated30_20261004 as rf
from numba import set_num_threads
from threadpoolctl import threadpool_limits

np, pd, rasterio, Window = prep.np, prep.pd, prep.rasterio, prep.Window
OUT = prep.ROOT / 'output/RF_recomputed_QA01_20261005'
DATA = OUT / 'data'
MODELS = OUT / 'RF_endpoints_2005_2023_QA01_MDNlt0p5_repeated30'
TAC = DATA / 'lag1_rasters_2000_2024'
YEARS = prep.core.YEARS


def qa01_values(ndvi, qa):
    good = ((qa == 0) | (qa == 1)) & np.isfinite(ndvi) & (ndvi != -9999)
    if np.any(good & ((ndvi < -.2001) | (ndvi > 1.0001))):
        raise ValueError('QA0/1 NDVI outside expected physical range')
    values = np.full(ndvi.shape, np.nan, np.float32)
    values[good] = np.tanh(ndvi[good].astype(float)**2).astype(np.float32)
    return values


def read_strip(packs, start, height, nt, stamps):
    values = np.full((nt, height, 7200), np.nan, np.float32)
    offsets = np.cumsum([0]+[len(p['dates']) for p in packs])

    def worker(job):
        index, pack = job
        indices = [3*i+band for i in range(len(pack['dates'])) for band in [1, 3]]
        for tile in pack['tiles']:
            path = Path(tile['path'])
            before = (path.stat().st_size, path.stat().st_mtime_ns)
            assert before == stamps[str(path)]
            with rasterio.open(path) as src:
                assert src.crs.to_epsg() == 4326 and src.nodata == -9999
                data = src.read(indices, window=Window(0, start, tile['width'], height))
            converted = qa01_values(data[0::2], data[1::2])
            strict = prep.core.kndvi_from_source(data[0::2], data[1::2])
            assert (np.isfinite(converted) | ~np.isfinite(strict)).all()
            selected = np.isfinite(strict)
            np.testing.assert_array_equal(converted[selected], strict[selected])
            values[offsets[index]:offsets[index+1], :, tile['col']:tile['col']+tile['width']] = converted
            assert before == (path.stat().st_size, path.stat().st_mtime_ns)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, enumerate(packs)))
    return values


def masks():
    manifest = rf.read(prep.MANIFEST)
    support = {s['variable']: s for s in manifest['response_and_masks']}
    for spec in manifest['predictors'] + list(support.values()):
        assert '05_随机森林归因部分' not in spec['path']
    lc = prep.package.mat(support['land_cover'])
    stable = prep.package.mat(support['stable_vegetation_mask'])
    old = prep.package.mat(support['MDN'])
    base = (stable == 1) & np.isfinite(lc) & (lc != 0) & np.isfinite(old) & (old < .5)
    return manifest, support, lc, old, base


def preflight(args_tuple):
    qa = np.array([-1, 0, 1, 2, 3, 4, 0, 1], dtype=np.float32)
    ndvi = np.array([.5, .5, .5, .5, .5, .5, -9999, np.nan], dtype=np.float32)
    actual = qa01_values(ndvi, qa)
    np.testing.assert_array_equal(np.isfinite(actual), [False, True, True, False, False, False, False, False])
    np.testing.assert_allclose(actual[1:3], np.tanh(.25), rtol=1e-7)
    samples = []
    for point in range(5):
        with np.load(prep.core.OUT/f'point_{point}_stages.npz') as saved:
            samples.append(qa01_values(saved['ndvi'], saved['qa']))
    samples = np.stack(samples)
    assert samples.shape[1] == len(args_tuple[0])
    result = prep.core.calculate(samples, *args_tuple)
    assert result.shape == (5, 21, 3)
    assert (np.nan_to_num(result[:, :, 2]) <= np.nan_to_num(result[:, :, 1])).all()
    prep.checked_fit()
    evidence = rf.read(prep.core.OUT/'robust_independent_conformance.json')
    assert evidence['all_passed']
    record = dict(passed=True, QA_retained=[0, 1], QA_excluded=[-1, 2, 3, 4],
                  nodata_rejected=True, physical_NDVI_scaling=1,
                  point_fixture_shape=list(result.shape), OLS_unit_check=True,
                  robust_implementation_reused=str(prep.core.__file__),
                  core_script_sha256=rf.digest(prep.core.__file__),
                  QA0_baseline_core_sha256=rf.read(prep.core.OUT/'local_run.json')['script_sha256'])
    assert record['core_script_sha256'] == record['QA0_baseline_core_sha256']
    prep.dump(OUT/'preflight.json', record)


def rebuild_rasters(base, threads, strip_height):
    done = DATA/'raster_complete.json'
    green = DATA/'greenness_QA01_2000_2024.tif'
    response = DATA/'TAC_trend_endpoints_2005_2023_QA01.tif'
    if done.exists():
        saved = rf.read(done)
        assert saved['status'] == 'completed'
        assert all(rf.digest(p) == h for p, h in saved['outputs'].items())
        return green, response
    packs, dates, args_tuple = prep.core.configuration()
    preflight(args_tuple)
    stamps = {t['path']: (Path(t['path']).stat().st_size, Path(t['path']).stat().st_mtime_ns)
              for pack in packs for t in pack['tiles']}
    paths = [TAC/f'KNDVI_TAC_lag1_5year_{year}_01_01_local_QA01.tif' for year in YEARS]
    outputs = [green, response] + paths
    assert not any(path.exists() for path in outputs), 'Refuse to mix finalized raster results'
    TAC.mkdir(parents=True, exist_ok=True)
    set_num_threads(threads)
    date_years = dates.astype('datetime64[Y]').astype(int)+1970
    annual_years = np.arange(2000, 2025)
    started = time.monotonic()
    state = dict(status='running', qa='SummaryQA == 0 or SummaryQA == 1',
                 preprocessing_period='2000-2024', missingness_period='2000-2023',
                 annual_response_endpoints='2005-2023', threads=threads, strip_height=strip_height,
                 spatial_domain='Original RF stable MASK, nonzero land cover, legacy MDN<0.5',
                 total_domain_pixels=int(base.sum()), source_stamps=stamps, completed_rows=0,
                 script_sha256=rf.digest(__file__), qa0_baseline=str(prep.OUT))
    with ExitStack() as stack, threadpool_limits(limits=1):
        writers = [stack.enter_context(rasterio.open(str(p)+'.partial.tif', 'w', **prep.profile(c)))
                   for p, c in zip(outputs, [6, 3]+[3]*21)]
        green_writer, trend_writer, tac_writers = writers[0], writers[1], writers[2:]
        for i, name in enumerate(['kNDVI_mean_2000_2023', 'kNDVI_trend_2000_2023',
                                 'kNDVI_mean_2000_2024', 'kNDVI_trend_2000_2024',
                                 'QA01_missing_fraction_2000_2023', 'QA01_missing_fraction_2000_2024'], 1):
            green_writer.set_band_description(i, name)
        for i, name in enumerate(['delta_TAC', 'OLS_pvalue', 'valid_window_count'], 1):
            trend_writer.set_band_description(i, name)
        for year, writer in zip(YEARS, tac_writers):
            for i, name in enumerate(['TAC_lag1', 'observation_count', 'pair_count'], 1):
                writer.set_band_description(i, name)
            writer.update_tags(window_start=f'{year-5}-01-01', window_end_exclusive=f'{year}-01-01',
                               estimator='Validated Talwar robust lagged slope with intercept')
        for writer in writers:
            writer.update_tags(qa='SummaryQA in [0,1]', domain=state['spatial_domain'],
                               NDVI_scale_applied='1', kNDVI_formula='tanh(NDVI**2)')
        baseline = stack.enter_context(rasterio.open(prep.DATA/'greenness_QA0_2000_2024.tif'))
        for start in range(0, 2800, strip_height):
            height = min(strip_height, 2800-start)
            selected = base[start:start+height].ravel()
            window = Window(0, start, 7200, height)
            target_green = np.full((6, height*7200), -9999, np.float32)
            target_trend = np.full((3, height*7200), -9999, np.float32)
            target_tac = np.full((21, 3, height*7200), -9999, np.float32)
            if selected.any():
                full = read_strip(packs, start, height, len(dates), stamps)
                values = np.ascontiguousarray(full.reshape(len(dates), -1)[:, selected].T)
                del full
                annual = np.full((25, len(values)), np.nan, np.float64)
                for j, year in enumerate(annual_years):
                    block = values[:, date_years == year]
                    count = np.isfinite(block).sum(axis=1)
                    np.divide(np.nansum(block, axis=1, dtype=np.float64), count,
                              out=annual[j], where=count > 0)
                bands = []
                for last in [2023, 2024]:
                    a = annual[:last-1999]
                    count = np.isfinite(a).sum(axis=0)
                    mean = np.full(len(values), np.nan)
                    np.divide(np.nansum(a, axis=0), count, out=mean, where=count > 0)
                    slope, _, _ = prep.fit_pixels(a, annual_years[:last-1999])
                    bands.extend([mean, slope])
                for last in [2023, 2024]:
                    positions = date_years <= last
                    bands.append(1-np.isfinite(values[:, positions]).sum(axis=1)/positions.sum())
                strict_missing = baseline.read(5, window=window).ravel()[selected]
                assert (bands[4] <= strict_missing+1e-7).all()
                g = np.stack(bands).astype(np.float32)
                g[~np.isfinite(g)] = -9999
                target_green[:, selected] = g
                calculated = prep.core.calculate(values, *args_tuple)
                assert (np.nan_to_num(calculated[:, :, 2]) <= np.nan_to_num(calculated[:, :, 1])).all()
                tac = calculated.transpose(1, 2, 0).copy()
                tac[~np.isfinite(tac)] = -9999
                target_tac[:, :, selected] = tac
                a = calculated[:, :19, 0].T.copy()
                a[np.abs(a) > 1] = np.nan
                slope, p, n = prep.fit_pixels(a, np.arange(2005, 2024))
                t = np.stack([slope, p, n]).astype(np.float32)
                t[~np.isfinite(t)] = -9999
                target_trend[:, selected] = t
                del values, annual, calculated, tac, a
            green_writer.write(target_green.reshape(6, height, 7200), window=window)
            trend_writer.write(target_trend.reshape(3, height, 7200), window=window)
            for j, writer in enumerate(tac_writers):
                writer.write(target_tac[j].reshape(3, height, 7200), window=window)
            state.update(completed_rows=start+height, elapsed_seconds=time.monotonic()-started)
            prep.dump(DATA/'raster_status.json', state)
            prep.log(f'QA01 source/TAC/greenness rows {start+height}/2800; elapsed {state["elapsed_seconds"]/60:.1f} min')
    for path in outputs:
        Path(str(path)+'.partial.tif').replace(path)
    state.update(status='completed', outputs={str(p): rf.digest(p) for p in outputs})
    prep.dump(done, state)
    return green, response


def tables(manifest, support, lc, old, base, green, response):
    target = DATA/'analysis_endpoints_2005_2023_QA01_MDNlt0p5.parquet'
    record = DATA/'source_manifest.json'
    if record.exists():
        saved = rf.read(record)
        assert saved['table_sha256'] == rf.digest(target)
        return target, saved
    specs = manifest['predictors']
    for s in specs+[support[n] for n in ['land_cover', 'stable_vegetation_mask', 'MDN', 'climate_zone']]:
        assert rf.digest(s['path']) == s['sha256'], s['variable']
    names = [s['variable'] for s in specs]
    ids = np.flatnonzero(base)
    rows, cols = np.divmod(ids, 7200)
    df = pd.DataFrame(dict(pixel_id=((rows+200)*7200+cols).astype('int32'),
                          longitude=-180+(cols+.5)*.05, latitude=80-(rows+.5)*.05,
                          vegetation=np.where(lc.ravel()[ids] <= 5, 'Forest', 'Non-forest'),
                          legacy_MDN=old.ravel()[ids]))
    df['climate_zone'] = prep.package.mat(support['climate_zone']).ravel()[ids]
    for s in specs:
        if s['variable'] in ['kNDVI', 'dkNDVI']:
            continue
        a = prep.package.mat(s).ravel()[ids].astype(np.float32)
        a[(~np.isfinite(a)) | (np.abs(a) > 1e30) | np.isin(a, [-9999, -32768])] = np.nan
        df[s['variable']] = a
    with rasterio.open(green) as src:
        g = src.read(masked=True).filled(np.nan).reshape(6, -1)[:, ids]
    with rasterio.open(response) as src:
        a = src.read(masked=True).filled(np.nan).reshape(3, -1)[:, ids]
    df['kNDVI'], df['dkNDVI'] = g[0], g[1]
    df['QA01_missing_fraction'] = g[4]
    df['y_continuous'], df['tac_pvalue'], df['valid_TAC_windows'] = a
    df['y'] = (a[0] >= 0).astype('int8')
    valid = np.isfinite(a[0]) & np.isfinite(a[1])
    sig = valid & (a[1] < .05)
    complete = np.isfinite(df[names]).all(axis=1).to_numpy()
    missing_ok = np.isfinite(g[4]) & (g[4] >= 0) & (g[4] < .5)
    keep = sig & complete & missing_ok
    counts = []
    for veg in ['Forest', 'Non-forest']:
        group = df.vegetation.eq(veg).to_numpy()
        counts.append(dict(vegetation=veg, original_base=int(group.sum()),
                           QA01_missing_lt50=int((group & missing_ok).sum()),
                           valid_trend=int((group & valid).sum()), significant=int((group & sig).sum()),
                           significant_complete_before_missing_filter=int((group & sig & complete).sum()),
                           n_eligible=int((group & keep).sum()),
                           decline=int((group & keep & (a[0] >= 0)).sum()),
                           increase=int((group & keep & (a[0] < 0)).sum()),
                           zero_trends=int((group & keep & (a[0] == 0)).sum())))
    retained = df.loc[keep].reset_index(drop=True)
    assert retained.pixel_id.is_unique
    assert (retained.tac_pvalue < .05).all() and (retained.QA01_missing_fraction < .5).all()
    retained.to_parquet(target, index=False)
    df[['pixel_id', 'vegetation', 'QA01_missing_fraction', 'y_continuous', 'tac_pvalue']].to_parquet(
        DATA/'all_original_base_diagnostics.parquet', index=False)
    pd.DataFrame(counts).to_csv(DATA/'cohort_counts.csv', index=False, encoding='utf-8-sig')
    saved = dict(status='completed', qa='SummaryQA in [0,1]', candidates=names,
                 table=str(target), table_sha256=rf.digest(target), counts=counts,
                 original_masks_retained=True, forbidden_data_read=False,
                 preprocessing='same 2000-2024 rolling detrend/harmonics/Talwar implementation as QA0',
                 missingness='finite NDVI retained by QA0/1; actual 548 observations in 2000-2023; strict <50%',
                 response='OLS TAC trend on endpoints 2005-2023; [-1,1] TAC filter; OLS P<0.05',
                 nonvegetation_predictors='same approved 20 other candidate predictor sources',
                 source_manifest=str(prep.MANIFEST), source_manifest_sha256=rf.digest(prep.MANIFEST),
                 script_sha256=rf.digest(__file__), raster_evidence=str(DATA/'raster_complete.json'))
    prep.dump(record, saved)
    prep.log('QA01 cohorts: '+str(counts))
    return target, saved


def comparison():
    qa0 = prep.OUT/'RF_endpoints_2005_2023_QA0_MDNlt0p5_repeated30'
    perf, importance = [], []
    for label, folder in [('QA0', qa0), ('QA01', MODELS)]:
        for veg in ['Forest', 'Non-forest']:
            performance = pd.read_csv(folder/veg/'summary/performance_summary.csv')
            performance.insert(0, 'vegetation', veg)
            performance.insert(0, 'QA', label)
            perf.append(performance)
            rank = pd.read_csv(folder/veg/'summary/importance_summary.csv')
            rank.loc[rank.retention_count.eq(0), 'rank_by_median'] = np.nan
            rank.to_csv(folder/veg/'summary/importance_summary.csv', index=False) if label == 'QA01' else None
            rank.insert(0, 'vegetation', veg)
            rank.insert(0, 'QA', label)
            importance.append(rank)
    pd.concat(perf, ignore_index=True).to_csv(OUT/'QA0_vs_QA01_performance.csv', index=False, encoding='utf-8-sig')
    ranks = pd.concat(importance, ignore_index=True)
    ranks.to_csv(OUT/'QA0_vs_QA01_importance.csv', index=False, encoding='utf-8-sig')
    lines = ['# QA=0/1 RF reanalysis', '',
             'Forests and non-forests each have 30 fits. Only QA selection and its derived variables are changed.',
             'This is random-holdout internal validation, not spatially independent validation.',
             'Intervals are repeat-fit percentile ranges, not independent external confidence intervals.', '']
    for veg in ['Forest', 'Non-forest']:
        frame = pd.read_csv(MODELS/veg/'summary/performance_summary.csv').set_index('metric')
        eligible = rf.read(MODELS/veg/'runs/00/design.json')['n_eligible']
        top = ranks.loc[ranks.QA.eq('QA01') & ranks.vegetation.eq(veg) & ranks.retention_count.gt(0)].head(6)
        lines += [f'## {veg}', f'Eligible pixels: {eligible:,}',
                  f"Median accuracy: {frame.loc['accuracy','median']:.2%}",
                  f"Median AUC: {frame.loc['roc_auc','median']:.4f}",
                  'Top six: '+', '.join(top.variable), '']
    (OUT/'results_summary.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--strip-height', type=int, default=64)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    manifest, support, lc, old, base = masks()
    green, response = rebuild_rasters(base, args.threads, args.strip_height)
    table, saved = tables(manifest, support, lc, old, base, green, response)
    if args.prepare_only:
        return
    rf.CONFIG = dict(rf.CONFIG, version='local_QA01_rebuilt_RF_30_v1',
                     missing_mask='QA0/1 missing fraction 2000-2023 <0.5, original masks retained')
    MODELS.mkdir(parents=True, exist_ok=True)
    protocol = dict(configuration=rf.CONFIG, source_manifest_sha256=rf.digest(DATA/'source_manifest.json'),
                    table_sha256=saved['table_sha256'], script_sha256=rf.digest(__file__),
                    reused_training_script_sha256=rf.digest(rf.__file__),
                    helper_sha256=rf.digest(rf.LEGACY_SCRIPT), qa='0 or 1',
                    seeds_same_as_QA0=True, output=str(MODELS))
    if (MODELS/'protocol.json').exists():
        assert rf.read(MODELS/'protocol.json') == protocol
    else:
        prep.dump(MODELS/'protocol.json', protocol)
    prep.dump(MODELS/'status.json', dict(status='running', models_planned=60))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(rf.run_group, veg, MODELS, table, saved['candidates'])
                       for veg in ['Forest', 'Non-forest']]
            for future in futures:
                future.result()
        rf.figures(MODELS)
        for extension in ['png', 'pdf']:
            (MODELS/f'figures/Fig_RF_QA0_OOB_top10_30runs.{extension}').replace(
                MODELS/f'figures/Fig_RF_QA01_OOB_top10_30runs.{extension}')
        comparison()
        prep.dump(MODELS/'status.json', dict(status='completed', models=60, qa='0 or 1'))
        prep.log('QA01 RF ALL 60 COMPLETE')
    except Exception as error:
        prep.dump(MODELS/'status.json', dict(status='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()
