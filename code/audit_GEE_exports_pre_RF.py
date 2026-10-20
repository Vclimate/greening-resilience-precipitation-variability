"""Archive GEE exports and audit rasters and the existing RF cohort; no fitting."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

DLL = os.add_dll_directory(str(Path(sys.base_prefix) / 'Library/bin'))
import pyarrow.parquet  # Load Anaconda Arrow before rasterio's DLLs.
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'input/GEE_recomputed_20260927'
OUT = DEST / 'pre_RF_audit'
HISTORICAL_RF_INPUTS = Path(os.environ.get('JRS_HISTORICAL_RF_INPUTS', ROOT / 'input/historical_rf_inputs'))
SOURCES = {
    'ERA5': Path('C:/Users/user/Downloads/JRS_ERA5_all_bilinear_005_recomputed'),
    'nonERA5': Path('C:/Users/user/Downloads/JRS_nonERA5_public_005_recomputed'),
}
# Bounds are arithmetic/physical checks, not statistical outlier filters.
SPECS = {
    'P': ('mm yr-1', 0, None), 'dP': ('mm yr-2', None, None),
    'P_SI': ('1', 0, 11/6), 'P_NIPV': ('1', 0, None),
    'Ta': ('degC', -100, 70), 'dTa': ('degC yr-1', None, None),
    'P_ET': ('mm yr-1', None, None), 'Rs': ('J m-2 day-1', 0, None),
    'SM': ('m3 m-3', 0, 1), 'VPD': ('hPa', 0, None),
    'Clay': ('g kg-1', 0, 1000), 'Sand': ('g kg-1', 0, 1000),
    'SOC': ('dg kg-1', 0, 10000), 'Nitrogen': ('cg kg-1', 0, 100000),
    'Elevation': ('m', -500, 9000), 'Slope': ('degree', 0, 90),
}
GRID = from_origin(-180, 80, .05, .05)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf-8')


def table(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def read(path, band=1, window=None):
    with rasterio.open(path) as src:
        a = src.read(band, masked=True, window=window).astype('float64').filled(np.nan)
    a[~np.isfinite(a)] = np.nan
    return a


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for group, source in SOURCES.items():
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            target = DEST / group / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            source_hash = digest(path)
            if target.exists():
                if digest(target) != source_hash:
                    raise RuntimeError(f'Refuse to replace different archived file: {target}')
            else:
                shutil.copy2(path, target)
            assert digest(target) == source_hash
            manifest.append(dict(source=str(path), archived=str(target),
                                 bytes=path.stat().st_size, sha256=source_hash))
    table('archive_manifest.csv', manifest)
    print(f'Archived and SHA256 verified: {len(manifest)} files', flush=True)

    old = pd.read_parquet(HISTORICAL_RF_INPUTS / 'analysis.parquet')
    candidate = old.copy()
    corrected = pd.read_parquet(ROOT / 'input/Species_nodata_corrected_20260926/analysis_species_corrected_all_rows.parquet')
    assert old.pixel_id.is_unique and corrected.pixel_id.is_unique
    candidate['Species'] = corrected.set_index('pixel_id').Species.reindex(old.pixel_id).to_numpy()
    ids = old.pixel_id.to_numpy(dtype=np.int64)
    global_rows, cols = ids // 7200, ids % 7200
    np.testing.assert_allclose(old.longitude, -180+(cols+.5)*.05, atol=1e-5)
    np.testing.assert_allclose(old.latitude, 90-(global_rows+.5)*.05, atol=1e-5)
    rows = global_rows-200
    inside = (rows >= 0) & (rows < 2800)
    assert inside.all(), 'Old RF has pixels outside the exported latitude extent'

    raster_meta, stats, qa_stats, comparisons, missing = [], [], [], [], []
    predictors, qa_paths = {}, {}
    smoke_path = None
    for item in manifest:
        path = Path(item['archived'])
        if path.suffix.lower() != '.tif':
            continue
        smoke = path.name.startswith('TEST_')
        if smoke:
            smoke_path = path
        with rasterio.open(path) as src:
            raster_meta.append(dict(file=str(path), shape=list(src.shape), crs=str(src.crs),
                                    transform=list(src.transform), nodata=src.nodata,
                                    descriptions=list(src.descriptions), tags=src.tags()))
            assert src.crs.to_epsg() == 4326
            if not smoke:
                assert src.shape == (2800, 7200) and src.transform.almost_equals(GRID)
            assert src.nodata == -9999
            descriptions = src.descriptions
            if src.count == 1 and descriptions[0] in SPECS and not smoke:
                assert descriptions[0] not in predictors
                predictors[descriptions[0]] = path
            elif not smoke:
                qa_paths[path.name] = path
        for band, name in enumerate(descriptions, 1):
            a = read(path, band)
            values = a[np.isfinite(a)]
            summary = dict(file=path.name, band=name, valid=int(values.size),
                           missing=int(a.size-values.size), zeros=int((values == 0).sum()))
            if values.size:
                qs = np.quantile(values, [0, .01, .5, .99, 1])
                summary.update(zip(['min', 'p01', 'median', 'p99', 'max'], qs.tolist()))
            if name in SPECS:
                unit, low, high = SPECS[name]
                bad = np.zeros(values.size, dtype=bool)
                if low is not None:
                    bad |= values < low-1e-5
                if high is not None:
                    bad |= values > high+1e-5
                summary.update(units=unit, out_of_bounds=int(bad.sum()))
                stats.append(summary)
                if not smoke:
                    x = a[rows, cols]
                    candidate[name] = x.astype(np.float32)
                    for vegetation in ['Forest', 'Non-forest']:
                        select = old.vegetation.eq(vegetation).to_numpy()
                        before = old[name].to_numpy(dtype=float)
                        match = select & np.isfinite(x) & np.isfinite(before)
                        delta = x[match]-before[match]
                        comparisons.append(dict(variable=name, vegetation=vegetation,
                            paired=int(match.sum()), pearson_r=float(np.corrcoef(before[match], x[match])[0, 1]),
                            old_median=float(np.median(before[match])), new_median=float(np.median(x[match])),
                            median_abs_difference=float(np.median(np.abs(delta))),
                            mean_difference=float(delta.mean()),
                            sign_changes=int(((before[match] >= 0) != (x[match] >= 0)).sum())))
                        missing.append(dict(variable=name, vegetation=vegetation,
                                            missing=int((select & ~np.isfinite(x)).sum()),
                                            total=int(select.sum())))
            elif not smoke:
                rf_values = a[rows, cols]
                finite = rf_values[np.isfinite(rf_values)]
                summary.update(rf_missing=int((~np.isfinite(rf_values)).sum()),
                               rf_min=float(finite.min()) if finite.size else None,
                               rf_max=float(finite.max()) if finite.size else None)
                if name.startswith('n_') or name.startswith('valid_') or name.startswith('all_'):
                    v, n = np.unique(values, return_counts=True)
                    summary['global_value_counts'] = json.dumps(dict(zip(map(str,v),map(int,n))))
                    v, n = np.unique(finite, return_counts=True)
                    summary['rf_value_counts'] = json.dumps(dict(zip(map(str,v),map(int,n))))
                if name.endswith('_coverage'):
                    summary.update(rf_below_90pct=int((finite < .9).sum()),
                                   rf_zero=int((finite == 0).sum()),
                                   out_of_bounds=int(((values < 0) | (values > 1+1e-6)).sum()))
                qa_stats.append(summary)
            del a, values
        print(f'Audited: {path.name}', flush=True)
    assert set(predictors) == set(SPECS), 'Incomplete predictor export set'
    dump('raster_metadata.json', raster_meta)
    table('raster_statistics.csv', stats)
    table('QA_statistics.csv', qa_stats)
    table('old_new_matched_RF_comparison.csv', comparisons)
    table('new_predictor_missing_in_old_RF.csv', missing)

    def qa(prefix, band):
        path = next(p for n,p in qa_paths.items() if n.startswith(prefix))
        with rasterio.open(path) as src:
            index = src.descriptions.index(band)+1
        return read(path, index)

    checks = {}
    if smoke_path:
        with rasterio.open(smoke_path) as src:
            col, row = (~GRID) * (src.transform.c, src.transform.f)
            assert abs(col-round(col)) < 1e-6 and abs(row-round(row)) < 1e-6
            window = Window(round(col), round(row), src.width, src.height)
            assert src.transform.almost_equals(rasterio.windows.transform(window, GRID))
        small, global_part = read(smoke_path), read(predictors['SOC'], window=window)
        both = np.isfinite(small) & np.isfinite(global_part)
        checks['SOC_smoke_vs_global'] = dict(paired=int(both.sum()),
            mask_disagreement=int((np.isfinite(small) != np.isfinite(global_part)).sum()),
            max_abs_difference=float(np.max(np.abs(small[both]-global_part[both]))))

    p, pet, et = read(predictors['P']), read(predictors['P_ET']), qa('ERA5Land_QA_study', 'ET_mean_mm_yr')
    valid = np.isfinite(p) & np.isfinite(pet) & np.isfinite(et)
    error = (p-et-pet)[valid]
    checks['P_minus_ET_identity'] = dict(paired=int(valid.sum()),
        max_abs_difference=float(np.abs(error).max()),
        absolute_error_above_0_02_mm=int((np.abs(error) > .02).sum()))
    del p, pet, et, valid, error
    years = qa('ERA5Land_QA_baseline', 'n_complete_years_P')
    positive = qa('ERA5Land_QA_baseline', 'n_positive_total_years')
    excluded = (years == 34) & (positive < 34)
    checks['old_baseline_mask_zero_years'] = dict(global_pixels=int(excluded.sum()),
        old_RF_pixels=int(excluded[rows, cols].sum()),
        note='Counts nearest-neighbor QA pixels with complete but not all-positive annual totals; does not exactly measure the bilinear footprint effect.')
    del years, positive, excluded
    sand, clay = read(predictors['Sand']), read(predictors['Clay'])
    checks['soil_texture_sum'] = dict(above_1000=int((sand+clay > 1000.001).sum()))
    del sand, clay

    variables = pd.read_csv(LEGACY / 'approved_22_predictors.csv').variable.tolist()
    assert len(variables) == 22
    pd.testing.assert_frame_equal(candidate[['pixel_id','longitude','latitude','vegetation','y_continuous','y','tac_pvalue']],
                                  old[['pixel_id','longitude','latitude','vegetation','y_continuous','y','tac_pvalue']])
    assert np.array_equal(candidate.y, (candidate.y_continuous >= 0).astype('int8'))
    complete = np.isfinite(candidate[variables].to_numpy()).all(axis=1)
    candidate['all_22_finite'] = complete
    candidate.to_parquet(OUT / 'candidate_22_on_legacy_RF_cohort.parquet', index=False)
    counts = []
    for vegetation in ['Forest', 'Non-forest']:
        select = candidate.vegetation.eq(vegetation).to_numpy()
        n = int(select.sum())
        counts.append(dict(vegetation=vegetation, old_rows=n,
            complete_new_22=int((select & complete).sum()), removed=int((select & ~complete).sum()),
            corrected_species_missing=int((select & candidate.Species.isna().to_numpy()).sum()),
            positive_labels=int((select & complete & (candidate.y == 1).to_numpy()).sum())))
    table('RF_cohort_counts.csv', counts)
    dictionary = []
    for name in variables:
        if name in predictors:
            period = '1990-2023' if name in ['P_SI','P_NIPV'] else ('2000-2023' if name in ['P','dP','Ta','dTa','Rs','SM','VPD','P_ET'] else 'static')
            dictionary.append(dict(variable=name, path=str(predictors[name]), units=SPECS[name][0], period=period,
                processing='native metric then bilinear' if period != 'static' else 'native metric then valid-area mean'))
        else:
            source = ROOT / 'input/Species_nodata_corrected_20260926/Species_sum_nodata_corrected.mat' if name == 'Species' else HISTORICAL_RF_INPUTS / 'analysis.parquet'
            dictionary.append(dict(variable=name, path=str(source), units='see approved source manifest', period='legacy retained', processing='no new resampling'))
    table('candidate_data_dictionary.csv', dictionary)
    checks['cohort'] = counts
    checks['all_predictor_out_of_bounds'] = sum(s['out_of_bounds'] for s in stats)
    checks['labels_unchanged'] = True
    checks['old_tac_pvalue_max'] = float(old.tac_pvalue.max())
    checks['exact_pvalue_0_05'] = int(old.tac_pvalue.eq(.05).sum())
    checks['zero_TAC_trends'] = int(old.y_continuous.eq(0).sum())
    checks['cohort_scope'] = 'Only legacy complete-case RF rows; cannot recover formerly excluded pixels. Rebuild population before final training.'
    checks['provenance_warnings'] = [
        'ERA5 metadata CSV is absent; old valid_native_baseline band confirms the earlier shared baseline QA schema.',
        'nonERA5 CSV reports v3 but TIFF filenames omit v3 and tags contain no script version; source job linkage cannot be proved from these files.',
        'Species preserves the previous placement; prior subpixel offset was not re-registered.',
        'New GMTED slope uses the native 7.5 arcsec DEM, not an exact reproduction of legacy 30 arcsec processing.',
        'This audit checks exported values and alignment, not a complete independent recomputation from monthly/native source observations.'
    ]
    dump('audit_summary.json', checks)
    print(json.dumps(checks, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
