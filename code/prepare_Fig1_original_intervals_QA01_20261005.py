"""Restore Fig1 rowwise SEM and paired-baseline t intervals on current QA01 data."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import analyse_plot_QA01_unified22_20261005 as analysis
from scipy.io import loadmat, savemat
from scipy.stats import t

np, pd = analysis.np, analysis.pd
ROOT = analysis.OUT / 'figures_matlab_style'


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def movmean_include_nan(a, window=101):
    # MATLAB movmean defaults: centered odd window, shrink endpoints, include NaNs.
    index = np.arange(len(a))
    start, end = np.maximum(0, index-window//2), np.minimum(len(a), index+window//2+1)
    finite = np.isfinite(a)
    sums = np.r_[0., np.cumsum(np.where(finite, a, 0.))]
    counts = np.r_[0, np.cumsum(finite)]
    out = (sums[end]-sums[start])/(end-start)
    out[counts[end]-counts[start] != end-start] = np.nan
    return out


def main():
    table_path = analysis.ANALYSIS / 'kNDVI_pixel_statistics.parquet'
    table = pd.read_parquet(table_path)
    ids = table.pixel_id.to_numpy()-200*7200
    rows = ids//7200
    lat = 80-(rows+.5)*.05
    weights = np.cos(np.deg2rad(lat))
    forest = table.vegetation.eq('Forest').to_numpy()
    missing_valid = table.MDN_2000_2025.to_numpy() < .5
    slope = table.TAC_slope_22.to_numpy().astype(float)
    manifest = json.loads(Path('D:/papaer4/20260604返修补实验/input/GEE_recomputed_20260927/RF_input_manifest_22.json').read_text(encoding='utf-8'))
    lookup = {s['variable']: s for s in manifest['predictors']}
    finite_climate = np.ones(len(table), bool)
    for var in ['P', 'Ta']:
        finite_climate &= np.isfinite(analysis.read(lookup[var]['path']).ravel()[ids])
    lat_values = np.full((2800, 2, 3), np.nan)
    latitude = 80-(np.arange(2800)+.5)*.05
    latitude_rows = []
    for j, (veg, vm) in enumerate([('Forest', forest), ('Non-forest', ~forest)]):
        good = vm & missing_valid & np.isfinite(slope) & finite_climate
        r, x = rows[good], slope[good]
        n = np.bincount(r, minlength=2800)
        sums = np.bincount(r, weights=x, minlength=2800)
        squared = np.bincount(r, weights=x*x, minlength=2800)
        mean = np.divide(sums, n, out=np.full(2800, np.nan), where=n > 0)
        variance = np.divide(squared, n, out=np.full(2800, np.nan), where=n > 0)-mean**2
        std = np.sqrt(np.maximum(variance, 0))
        half = np.divide(1.96*std, np.sqrt(n), out=np.full(2800, np.nan), where=n > 0)
        smooth_mean, smooth_half = movmean_include_nan(mean), movmean_include_nan(half)
        lat_values[:, j] = np.column_stack([latitude, smooth_mean, smooth_half])
        latitude_rows.append(pd.DataFrame(dict(vegetation=veg, latitude=latitude, n_pixels=n,
            row_mean=mean, row_population_std=std, row_ci95_half=half,
            movmean101=smooth_mean, ci95_half_movmean101=smooth_half)))
        # Direct fixture for original nanstd(...,1) / sqrt(N) rule.
        check_row = int(np.argmax(n))
        fixture = x[r == check_row]
        np.testing.assert_allclose(mean[check_row], fixture.mean(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(half[check_row], 1.96*fixture.std(ddof=0)/np.sqrt(len(fixture)), rtol=1e-10)
    pd.concat(latitude_rows).to_csv(ROOT / 'Fig1b_original_method_QA01.csv', index=False)

    paths = [analysis.DATA / 'kNDVI/lag1_5year' / f'kNDVI_TAC_lag1_5year_{y}_01_01_QA01.tif' for y in range(2005, 2027)]

    def read_compact(path):
        x = analysis.read(path).ravel()[ids]
        x[np.abs(x) > 1] = np.nan
        return x

    with ThreadPoolExecutor(max_workers=4) as pool:
        stack = np.stack(list(pool.map(read_compact, paths)))
    cohort = missing_valid & np.isfinite(stack).all(axis=0)
    reference = pd.read_csv(analysis.ANALYSIS / 'kNDVI_regional_trajectories.csv')
    current = loadmat(ROOT / 'plot_inputs.mat')
    means, halves = np.zeros((22, 3)), np.zeros((22, 3))
    records, comparisons = [], []
    for j, (veg, vm) in enumerate([('Global', np.ones(len(ids), bool)), ('Forest', forest), ('Non-forest', ~forest)]):
        take = cohort & vm
        w = weights[take]
        n = int(take.sum())
        baseline = stack[0, take].astype(float)
        saved = reference.loc[reference.vegetation.eq(veg) & reference.zone.eq('Global')].sort_values('reference_year')
        assert (saved.n_pixels == n).all()
        critical = t.ppf(.975, n-1)
        for year in range(22):
            raw = stack[year, take].astype(float)
            mean_raw = np.dot(raw, w)/w.sum()
            np.testing.assert_allclose(mean_raw, saved.mean_TAC.iloc[year], rtol=0, atol=1e-10)
            delta = raw-baseline
            mean = np.dot(delta, w)/w.sum()
            variance = np.dot(w, (delta-mean)**2)/w.sum()
            half = critical*np.sqrt(variance)/np.sqrt(n)
            means[year, j], halves[year, j] = mean, half
            records.append(dict(vegetation=veg, reference_year=year+2004, endpoint_year=year+2005,
                n_pixels=n, mean_TAC_change_from_2004=mean, weighted_population_std=np.sqrt(variance),
                student_t_critical=critical, ci95_half=half, ci95_low=mean-half, ci95_high=mean+half))
        np.testing.assert_allclose(means[:, j], saved.mean_TAC.to_numpy()-saved.mean_TAC.iloc[0], atol=1e-10, rtol=0)
        before = np.polyfit(np.arange(22), saved.mean_TAC, 1)[0]
        after = np.polyfit(np.arange(22), means[:, j], 1)[0]
        np.testing.assert_allclose(before, after, atol=1e-14, rtol=0)
        comparisons.append(dict(vegetation=veg, current_raw_level_half_median=float(np.median(current['tac_half'][:, 0, j])),
            restored_paired_half_median=float(np.median(halves[1:, j])),
            restored_paired_half_max=float(halves[:, j].max()), slope_before=before, slope_after=after))
    pd.DataFrame(records).to_csv(ROOT / 'Fig1d_original_method_QA01.csv', index=False)
    savemat(ROOT / 'Fig1_original_method_intervals.mat', dict(lat_values_original=lat_values,
        tac_change_original=means, tac_half_original=halves), do_compression=True)
    report = dict(status='computed', original_plot= 'D:/papaer4/paper4全球植被恢复力/论文绘图所用数据/FIG1/Fig1_V5.m',
        original_trajectory_code='D:/papaer4/paper4全球植被恢复力/论文绘图所用数据/FIG2/森林非森林年际变化数据生成_IGBP_Xavg.mlx',
        current_plot_inputs_sha256=sha(ROOT / 'plot_inputs.mat'), source_table_sha256=sha(table_path),
        latitude_method='0.05-degree row: mean, population SD (ddof=0), half=1.96*SD/sqrt(N); movmean101 of mean and half separately; MATLAB include-NaN and shrink-edge defaults',
        trajectory_method='Per-pixel TAC_t-TAC_first; weighted population SD; t(0.975,N-1)*SD/sqrt(N)',
        cohort='Current fixed all22-valid QA01 cohort retained; do not restore old QA or change yearly sample membership',
        baseline_reference_year=2004, trend_slopes_unchanged=True, display_multiplier=1,
        interval_scope='Descriptive pixel-mean intervals assuming independent pixels, not spatially independent uncertainty or HAC trend intervals',
        old_results_not_reused=True, comparisons=comparisons)
    (ROOT / 'Fig1_original_intervals_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(comparisons), flush=True)
    print('Restored original Fig1 interval formulas; same current cohort; slopes checked identical.', flush=True)


if __name__ == '__main__':
    main()
