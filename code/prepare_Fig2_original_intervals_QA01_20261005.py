"""Original paired-baseline intervals for the five FIG2 regions, same fixed cohorts."""
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
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    table_path = analysis.ANALYSIS / 'kNDVI_pixel_statistics.parquet'
    table = pd.read_parquet(table_path)
    ids = table.pixel_id.to_numpy()-200*7200
    w_all = np.cos(np.deg2rad(80-(ids//7200+.5)*.05))
    forest = table.vegetation.eq('Forest').to_numpy()
    missing_valid = table.MDN_2000_2025.to_numpy() < .5
    zones = table.climate_zone.to_numpy()
    paths = [analysis.DATA / 'kNDVI/lag1_5year' / f'kNDVI_TAC_lag1_5year_{y}_01_01_QA01.tif'
             for y in range(2005, 2027)]

    def read_compact(path):
        values = analysis.read(path).ravel()[ids]
        values[np.abs(values) > 1] = np.nan
        return values

    with ThreadPoolExecutor(max_workers=4) as pool:
        stack = np.stack(list(pool.map(read_compact, paths)))
    cohort = missing_valid & np.isfinite(stack).all(axis=0)
    reference = pd.read_csv(analysis.ANALYSIS / 'kNDVI_regional_trajectories.csv')
    slopes = pd.read_csv(analysis.ANALYSIS / 'kNDVI_regional_OLS_HAC.csv')
    previous = loadmat(ROOT / 'Fig1_original_method_intervals.mat')
    means, halves = np.full((22, 5, 3), np.nan), np.full((22, 5, 3), np.nan)
    records = []
    for vi, (veg, vm) in enumerate([('Global', np.ones(len(ids), bool)), ('Forest', forest), ('Non-forest', ~forest)]):
        for zi, zone in analysis.ZONE.items():
            take = cohort & vm & ((zones == zi) if zi else True)
            weight, n = w_all[take], int(take.sum())
            baseline = stack[0, take].astype(float)
            saved = reference.loc[reference.vegetation.eq(veg) & reference.zone.eq(zone)].sort_values('reference_year')
            assert (saved.n_pixels == n).all() and len(saved) == 22
            critical = t.ppf(.975, n-1)
            for k in range(22):
                raw = stack[k, take].astype(float)
                np.testing.assert_allclose(np.dot(raw, weight)/weight.sum(), saved.mean_TAC.iloc[k], rtol=0, atol=1e-10)
                delta = raw-baseline
                mean = np.dot(delta, weight)/weight.sum()
                variance = np.dot(weight, (delta-mean)**2)/weight.sum()
                half = critical*np.sqrt(variance/n)
                means[k, zi, vi], halves[k, zi, vi] = mean, half
                records.append(dict(vegetation=veg, zone=zone, reference_year=2004+k,
                    n_pixels=n, TAC_change=mean, ci95_half=half))
            fit = slopes.loc[slopes.vegetation.eq(veg) & slopes.zone.eq(zone)
                             & slopes.period.eq('22_periods') & slopes.method.eq('OLS')].iloc[0]
            np.testing.assert_allclose(np.polyfit(np.arange(22), means[:, zi, vi], 1)[0], fit.slope, atol=1e-14, rtol=0)
    np.testing.assert_allclose(means[:, 0, :], previous['tac_change_original'], atol=1e-14, rtol=0)
    np.testing.assert_allclose(halves[:, 0, :], previous['tac_half_original'], atol=1e-14, rtol=0)
    savemat(ROOT / 'Fig2_original_method_intervals.mat', dict(tac_change=means, tac_half=halves), do_compression=True)
    pd.DataFrame(records).to_csv(ROOT / 'Fig2_original_method_QA01.csv', index=False)
    (ROOT / 'Fig2_original_intervals_audit.json').write_text(json.dumps(dict(
        status='verified', years=[2004, 2025], source_table_sha256=sha(table_path),
        original_fig1_intervals_sha256=sha(ROOT / 'Fig1_original_method_intervals.mat'),
        means_and_slopes_agree_with_saved_results=True, global_row_identical_to_fig1=True,
        formula='weighted population SD of pixel TAC_t-TAC_first; half=t(.975,N-1)*SD/sqrt(N)',
        cohort='same all22-valid QA01 fixed cohort within each vegetation/region',
        interval_scope='Independent-pixel descriptive intervals, not HAC trend confidence intervals',
        actual_interval_multiplier=1), indent=2), encoding='utf-8')
    print('FIG2: all15 trajectories verified; global matches FIG1; OLS slopes unchanged.', flush=True)


if __name__ == '__main__':
    main()
