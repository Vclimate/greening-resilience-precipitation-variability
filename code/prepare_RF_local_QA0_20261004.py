"""Rebuild RF cohorts from QA0 local TAC and verified covariate sources."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import json
from pathlib import Path
import time
from contextlib import ExitStack

import compute_local_kNDVI_TAC_lags235_QA0_20261004 as joint
import check_complete_RF_package as package
from compare_complete_new_lag1_TAC_20261001 import fit_pixels
from scipy.stats import linregress

core = joint.core
np, pd, rasterio, Window = core.np, package.pd, core.rasterio, core.Window
ROOT = core.ROOT
OUT = ROOT / 'output/RF_recomputed_QA0_20261004'
DATA = OUT / 'data'
MANIFEST = ROOT / 'input/GEE_recomputed_20260927/RF_input_manifest_22.json'
GRID = core.GRID
SHAPE = (2800, 7200)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)), encoding='utf-8')


def log(message):
    print(time.strftime('%H:%M:%S') + ' ' + message, flush=True)


def profile(count):
    return dict(driver='GTiff', height=2800, width=7200, count=count,
                dtype='float32', crs='EPSG:4326', transform=GRID, nodata=-9999,
                tiled=True, blockxsize=256, blockysize=256, compress='deflate', predictor=3)


def checked_fit():
    x = np.arange(2005, 2026)
    rng = np.random.default_rng(20261004)
    a = rng.normal(size=(21, 8)) + .08 * np.arange(21)[:, None]
    a[::3, 1::2] = np.nan
    slope, p, n = fit_pixels(a, x)
    for i in range(8):
        good = np.isfinite(a[:, i])
        expected = linregress(x[good], a[good, i])
        np.testing.assert_allclose([slope[i], p[i]], [expected.slope, expected.pvalue], rtol=1e-10, atol=1e-12)
        assert n[i] == good.sum()


def greenness():
    target = DATA / 'greenness_QA0_2000_2024.tif'
    record = DATA / 'greenness_complete.json'
    if record.exists():
        saved = json.loads(record.read_text(encoding='utf-8'))
        assert package.digest(target) == saved['sha256']
        log('Verified completed QA0 greenness cache')
        return target
    packs, dates, _ = core.configuration()
    stamps = {t['path']: (Path(t['path']).stat().st_size, Path(t['path']).stat().st_mtime_ns)
              for p in packs for t in p['tiles']}
    years = np.arange(2000, 2025)
    date_years = dates.astype('datetime64[Y]').astype(int) + 1970
    temp = Path(str(target) + '.partial.tif')
    descriptions = ('kNDVI_mean_2000_2023', 'kNDVI_trend_2000_2023',
                    'kNDVI_mean_2000_2024', 'kNDVI_trend_2000_2024',
                    'QA0_missing_fraction_2000_2023', 'QA0_missing_fraction_2000_2024')
    with rasterio.open(temp, 'w', **profile(6)) as writer:
        for i, name in enumerate(descriptions, 1):
            writer.set_band_description(i, name)
        writer.update_tags(qa='SummaryQA == 0', kNDVI='tanh(NDVI**2)',
                           averaging='mean of available 16-day observations within years, then equal-year mean',
                           trend='OLS of annual means, at least 3 finite years')
        for start in range(0, 2800, 256):
            height = min(256, 2800-start)
            values = joint.read_strip(packs, start, height, len(dates), 4, stamps)
            annual = np.full((25, len(values)), np.nan, np.float64)
            for j, year in enumerate(years):
                positions = np.flatnonzero(date_years == year)
                block = values[:, positions]
                count = np.isfinite(block).sum(axis=1)
                np.divide(np.nansum(block, axis=1, dtype=np.float64), count,
                          out=annual[j], where=count > 0)
            bands = []
            for last in [2023, 2024]:
                a = annual[:last-1999]
                count = np.isfinite(a).sum(axis=0)
                mean = np.full(len(values), np.nan)
                np.divide(np.nansum(a, axis=0), count, out=mean, where=count > 0)
                slope, _, _ = fit_pixels(a, years[:last-1999])
                bands.extend([mean, slope])
            for last in [2023, 2024]:
                selected = date_years <= last
                bands.append(1-np.isfinite(values[:, selected]).sum(axis=1)/selected.sum())
            result = np.stack(bands).reshape(6, height, 7200).astype('float32')
            result[~np.isfinite(result)] = -9999
            writer.write(result, window=Window(0, start, 7200, height))
            del values, annual, result
            log(f'QA0 kNDVI predictors: rows {start+height}/2800')
    temp.replace(target)
    dump(record, dict(status='completed', path=str(target), sha256=package.digest(target),
                      source_audit=str(core.AUDIT), source_stamps=stamps,
                      qa='SummaryQA == 0', NDVI_scale=1, bands=descriptions))
    return target


def response_maps():
    outputs = {last: DATA/f'TAC_trend_endpoints_2005_{last}_QA0.tif' for last in [2023, 2025]}
    record = DATA/'response_complete.json'
    if record.exists():
        saved = json.loads(record.read_text(encoding='utf-8'))
        assert all(package.digest(Path(p)) == h for p, h in saved['outputs'].items())
        log('Verified completed TAC trend cache')
        return outputs
    run = json.loads((core.OUT/'local_run.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and run['qa'] == 'SummaryQA == 0'
    files = [core.RASTERS/f'KNDVI_TAC_lag1_5year_{year}_01_01_local_QA0.tif' for year in range(2005, 2026)]
    hashes = {str(path): package.digest(path) for path in files}
    with ExitStack() as stack:
        readers = [stack.enter_context(rasterio.open(p)) for p in files]
        for r in readers:
            assert r.shape == SHAPE and r.transform.almost_equals(GRID) and r.crs.to_epsg() == 4326
        writers = {last: stack.enter_context(rasterio.open(str(p)+'.partial.tif', 'w', **profile(3)))
                   for last, p in outputs.items()}
        for last, writer in writers.items():
            for i, name in enumerate(['delta_TAC', 'OLS_pvalue', 'valid_window_count'], 1):
                writer.set_band_description(i, name)
            writer.update_tags(first_endpoint='2005', last_endpoint=str(last), range_filter='-1 <= TAC <= 1',
                               minimum_windows='3', significance='two-sided OLS t-test, not HAC-adjusted')
        for start in range(0, 2800, 256):
            height = min(256, 2800-start)
            a = np.stack([r.read(1, masked=True, window=Window(0, start, 7200, height))
                          .filled(np.nan).reshape(-1) for r in readers])
            a[np.abs(a) > 1] = np.nan
            for last, writer in writers.items():
                count = last-2004
                slope, p, n = fit_pixels(a[:count], np.arange(2005, last+1))
                for i in [17, min(12123, len(slope)-1)]:
                    good = np.isfinite(a[:count, i])
                    if good.sum() > 2 and np.ptp(a[:count, i][good]) > 0:
                        expected = linregress(np.arange(2005, last+1)[good], a[:count, i][good])
                        np.testing.assert_allclose([slope[i], p[i]], [expected.slope, expected.pvalue],
                                                   rtol=1e-7, atol=1e-10)
                result = np.stack([slope, p, n]).reshape(3, height, 7200).astype('float32')
                result[~np.isfinite(result)] = -9999
                writer.write(result, window=Window(0, start, 7200, height))
            log(f'New TAC trends: rows {start+height}/2800')
    for path in outputs.values():
        Path(str(path)+'.partial.tif').replace(path)
    dump(record, dict(status='completed', inputs=hashes,
                      outputs={str(p): package.digest(p) for p in outputs.values()},
                      original_range_rule='[-1,1]', regression='OLS with intercept, available windows >=3',
                      pvalue='ordinary two-sided OLS t-test; overlapping windows not corrected'))
    return outputs


def build_tables(response, green):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    support = {s['variable']: s for s in manifest['response_and_masks']}
    predictors = manifest['predictors']
    names = [s['variable'] for s in predictors]
    sources = {s['path']: s['sha256'] for s in predictors + [support[n] for n in
               ['land_cover', 'stable_vegetation_mask', 'MDN', 'climate_zone']]}
    assert not any('05_随机森林归因部分' in p for p in sources)
    for path, expected in sources.items():
        assert package.digest(Path(path)) == expected, path
    lc = package.mat(support['land_cover'])
    stable = package.mat(support['stable_vegetation_mask'])
    mdn = package.mat(support['MDN'])
    base = (stable == 1) & np.isfinite(lc) & (lc != 0) & np.isfinite(mdn) & (mdn < .5)
    ids = np.flatnonzero(base)
    rows, cols = np.divmod(ids, 7200)
    table = pd.DataFrame(dict(pixel_id=((rows+200)*7200+cols).astype('int32'),
                            longitude=-180+(cols+.5)*.05, latitude=80-(rows+.5)*.05,
                            vegetation=np.where(lc.ravel()[ids] <= 5, 'Forest', 'Non-forest'),
                            legacy_MDN=mdn.ravel()[ids]))
    table['climate_zone'] = package.mat(support['climate_zone']).ravel()[ids]
    del lc, stable, mdn, base
    stats = []
    for spec in predictors:
        if spec['variable'] in ['kNDVI', 'dkNDVI']:
            continue
        values = package.mat(spec).ravel()[ids].astype('float32')
        values[(~np.isfinite(values)) | (np.abs(values) > 1e30) | np.isin(values, [-9999, -32768])] = np.nan
        table[spec['variable']] = values
        stats.append(dict(variable=spec['variable'], source=spec['path'], missing=int(np.isnan(values).sum())))
    with rasterio.open(green) as reader:
        g = reader.read(masked=True).filled(np.nan).reshape(6, -1)[:, ids]
    table['kNDVI'] = g[0]
    table['dkNDVI'] = g[1]
    table['QA0_missing_fraction'] = g[4]
    tables, qa0_tables = {}, {}
    counts = []
    for last, path in response.items():
        with rasterio.open(path) as reader:
            a = reader.read(masked=True).filled(np.nan).reshape(3, -1)[:, ids]
        result = table.copy()
        result['y_continuous'] = a[0]
        result['tac_pvalue'] = a[1]
        result['valid_TAC_windows'] = a[2]
        result['y'] = (a[0] >= 0).astype('int8')
        valid = np.isfinite(a[0]) & np.isfinite(a[1])
        significant = valid & (a[1] < .05)
        complete = np.isfinite(result[names]).all(axis=1).to_numpy()
        keep = significant & complete
        for veg in ['Forest', 'Non-forest']:
            group = result.vegetation.eq(veg).to_numpy()
            counts.append(dict(last_endpoint=last, vegetation=veg, base_pixels=int(group.sum()),
                               valid_trend=int((group & valid).sum()), significant=int((group & significant).sum()),
                               significant_percent_of_valid=100*float((group & significant).sum())/max(1, (group & valid).sum()),
                               complete22=int((group & keep).sum()), decline=int((group & keep & (a[0] >= 0)).sum()),
                               increase=int((group & keep & (a[0] < 0)).sum()),
                               zero_trends=int((group & keep & (a[0] == 0)).sum())))
        result = result.loc[keep].reset_index(drop=True)
        assert result.pixel_id.is_unique and (result.tac_pvalue < .05).all()
        assert np.isfinite(result[names].to_numpy()).all()
        np.testing.assert_array_equal(result.y.to_numpy(), (result.y_continuous >= 0).astype('int8'))
        target = DATA/f'analysis_endpoints_2005_{last}.parquet'
        result.to_parquet(target, index=False)
        tables[str(last)] = dict(path=str(target), sha256=package.digest(target), rows=len(result))
        log(f'Prepared endpoint {last}: {len(result):,} complete significant pixels')
        strict = result.loc[result.QA0_missing_fraction < .5].reset_index(drop=True)
        strict_target = DATA/f'analysis_endpoints_2005_{last}_QA0_MDNlt0p5.parquet'
        strict.to_parquet(strict_target, index=False)
        strict_counts = []
        for veg in ['Forest', 'Non-forest']:
            group = strict.loc[strict.vegetation.eq(veg)]
            strict_counts.append(dict(vegetation=veg, complete22=len(group),
                                      decline=int(group.y.eq(1).sum()), increase=int(group.y.eq(0).sum()),
                                      max_QA0_missing_fraction=float(group.QA0_missing_fraction.max())))
        qa0_tables[str(last)] = dict(path=str(strict_target), sha256=package.digest(strict_target),
                                    rows=len(strict), counts=strict_counts)
        log(f'New QA0 MDN<0.5 endpoint {last}: {len(strict):,} complete significant pixels')
    pd.DataFrame(counts).to_csv(DATA/'cohort_counts.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(stats).to_csv(DATA/'predictor_missingness.csv', index=False, encoding='utf-8-sig')
    dump(DATA/'source_manifest.json', dict(status='completed', manifest=str(MANIFEST), manifest_sha256=package.digest(MANIFEST),
        source_files_sha256=sources, candidates=names, tables=tables, qa0_tables=qa0_tables, counts=counts,
        new_response=str(DATA/'response_complete.json'), new_greenness=str(green),
        predictor_periods={s['variable']: s.get('period', 'legacy static/source-specific') for s in predictors},
        greenness_period='2000-2023 for both RF cohorts; equal-year average and annual-mean OLS trend',
        base_rule='Original stable MASK==1, IGBP nonzero, legacy MDN<0.5; no old Y or old p-value filtering',
        original_masks_retained=True, new_QA0_missingness_for_diagnostics_only=False,
        selected_training_mask='QA0_missing_fraction_2000_2023 < 0.5, additionally to original stable/landcover mask',
        significance='strict ordinary OLS P<0.05; not corrected for overlapping-window dependence',
        complete_case='all 22 candidate predictors finite before training-only VIF',
        forbidden_data_read=False))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    checked_fit()
    response = response_maps()
    green = greenness()
    build_tables(response, green)
    log('PREPARATION COMPLETED')


if __name__ == '__main__':
    main()
