"""Check all packaged predictors on the full legacy response mask, without fitting."""
import json
import argparse
import os
import sys
from pathlib import Path

DLL_HANDLE = os.add_dll_directory(str(Path(sys.base_prefix)/'Library/bin'))
import pyarrow.parquet
import h5py
from scipy.io import loadmat
from audit_GEE_exports_pre_RF import np, pd, ROOT, DEST, read, digest

OUT = DEST / 'full_package_check'


def mat(spec):
    if spec.get('format') == 'GeoTIFF':
        return read(spec['path'])
    path, key = spec['path'], spec['dataset']
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            assert f[key].shape == (7200, 3600)
            return f[key][:,200:3000].T
    a = loadmat(path, variable_names=[key])[key]
    assert a.shape == (3600,7200)
    return a[200:3000,:]


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=DEST/'RF_input_manifest_22.json')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output_dir
    OUT.mkdir(exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    expected_changes=manifest.get('expected_changed_predictors',[])
    specs = manifest['predictors']
    other = {s['variable']:s for s in manifest['response_and_masks']}
    hashes = {}
    for s in specs+list(other.values()):
        if s['path'] not in hashes:
            hashes[s['path']] = digest(Path(s['path']))
        assert hashes[s['path']] == s['sha256'], s['variable']

    y,p = mat(other['delta_TAC']),mat(other['TAC_pvalue'])
    lc,stable,mdn = mat(other['land_cover']),mat(other['stable_vegetation_mask']),mat(other['MDN'])
    base = (stable == 1) & np.isfinite(lc) & (lc != 0) & (mdn < .5) & np.isfinite(y)
    assert np.isfinite(mdn[base]).all() and (mdn[base]>=0).all()
    assert np.isin(lc[base], np.arange(1,18)).all()
    sig = base & np.isfinite(p) & (p <= .05)
    saved = mat(other['legacy_significant_TAC'])
    assert np.array_equal(np.isfinite(saved[base]), sig[base])
    np.testing.assert_array_equal(saved[sig], y[sig])
    assert ((p[base & np.isfinite(p)] >= 0) & (p[base & np.isfinite(p)] <= 1)).all()
    crop_ids = np.flatnonzero(sig)
    rows,cols = np.divmod(crop_ids,7200)
    ids = (rows+200)*7200+cols
    data = pd.DataFrame(dict(pixel_id=ids.astype('int32'), longitude=-180+(cols+.5)*.05,
        latitude=80-(rows+.5)*.05, vegetation=np.where(lc.ravel()[crop_ids]<=5,'Forest','Non-forest'),
        y_continuous=y.ravel()[crop_ids], y=(y.ravel()[crop_ids]>=0).astype('int8'),
        tac_pvalue=p.ravel()[crop_ids]))
    checks = dict(unique_source_files_sha256_verified=len(hashes),
        valid_response_base=int(base.sum()), significant_response_pixels=len(data),
        p_exactly_0_05=int((p[sig]==.05).sum()),zero_y=int((y[sig]==0).sum()),
        saved_significance_matches=True,classes_on_base=np.unique(lc[base]).tolist(),
        sample_rule='Legacy base + p<=0.05, no IQR filtering, no model fitting',
        grid_note='Comparison verifies approved matrix placement, not independent georeferencing of every historical MAT.')
    del y,p,lc,stable,mdn,base,sig,saved
    old = pd.read_parquet(DEST/'pre_RF_audit/candidate_22_on_legacy_RF_cohort.parquet')
    lookup = pd.Index(ids).get_indexer(old.pixel_id)
    assert (lookup>=0).all()
    for name in ['y_continuous','y','tac_pvalue']:
        np.testing.assert_array_equal(data[name].to_numpy()[lookup],old[name].to_numpy())
    stats = []
    for s in specs:
        a = mat(s) if s['format']=='MAT' else read(s['path'])
        assert a.shape == (2800,7200)
        x = a.ravel()[crop_ids].astype('float64')
        sentinel = (np.abs(x)>1e30) | np.isin(x,[-9999,-32768])
        finite_fill = np.isfinite(x) & sentinel
        x[sentinel] = np.nan
        x[~np.isfinite(x)] = np.nan
        vals = x[np.isfinite(x)]
        stat = dict(variable=s['variable'], finite=len(vals), missing=int((~np.isfinite(x)).sum()),
            finite_fill_encoded=int(finite_fill.sum()), zero=int((vals==0).sum()),
            min=float(vals.min()),median=float(np.median(vals)),max=float(vals.max()))
        data[s['variable']] = x.astype('float32')
        if s['variable'] not in expected_changes:
            np.testing.assert_array_equal(data[s['variable']].to_numpy()[lookup],old[s['variable']].to_numpy().astype('float32'))
        if s['variable'] in ['kNDVI','Forest_fraction']:
            stat['outside_0_1'] = int(((vals<0)|(vals>1)).sum())
        if s['variable'] in ['Species','WTD','HFP']:
            stat['negative_count'] = int((vals<0).sum())
        stats.append(stat)
        del a,x,vals
        print(json.dumps(stat),flush=True)
    kcz = mat(other['climate_zone']).ravel()[crop_ids]
    data['climate_zone'] = kcz
    checks['climate_zone_values'] = np.unique(kcz[np.isfinite(kcz)]).tolist()
    names=[s['variable'] for s in specs]
    complete = np.isfinite(data[names].to_numpy()).all(axis=1)
    checks['old_cohort_unchanged_X_and_Y_exactly_match'] = True
    checks['intentionally_changed_predictors'] = expected_changes
    checks['manifest_used'] = str(args.manifest)
    checks['counts'] = []
    for veg in ['Forest','Non-forest']:
        select=data.vegetation.eq(veg).to_numpy()
        keep=select & complete
        checks['counts'].append(dict(vegetation=veg,significant_before_complete=int(select.sum()),
            complete=int(keep.sum()),class1=int((keep & data.y.eq(1).to_numpy()).sum()),
            class0=int((keep & data.y.eq(0).to_numpy()).sum()),
            newly_recovered_vs_old_cohort=int((keep & ~np.isin(ids,old.pixel_id)).sum())))
    checks['complete_total']=int(complete.sum())
    checks['missing_predictor_rows']=int((~complete).sum())
    checks['complete_with_missing_or_nonpositive_climate_zone']=int((complete & (~np.isfinite(kcz)|(kcz<=0))).sum())
    pd.DataFrame(stats).to_csv(OUT/'predictor_checks_full_significant_population.csv',index=False,encoding='utf-8-sig')
    data['all_22_finite']=complete
    data.to_parquet(OUT/'full_significant_candidate_pixels.parquet',index=False)
    data.loc[complete].to_parquet(OUT/'complete22_candidate_pixels.parquet',index=False)
    # Verify the output table itself, not just the in-memory arrays.
    reread=pd.read_parquet(OUT/'complete22_candidate_pixels.parquet')
    assert reread.pixel_id.is_unique and np.isfinite(reread[names]).all().all()
    assert len(reread)==int(complete.sum())
    assert np.array_equal(reread.y,(reread.y_continuous>=0).astype('int8'))
    checks['parquet_roundtrip_passed']=True
    (OUT/'full_package_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    main()
