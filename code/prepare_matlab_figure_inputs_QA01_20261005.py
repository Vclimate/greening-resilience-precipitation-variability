"""Export saved QA01 results, not old figure caches, for native MATLAB plots."""
import hashlib
import json
from pathlib import Path

import analyse_plot_QA01_unified22_20261005 as analysis
from scipy.io import savemat
from scipy.ndimage import uniform_filter

np, pd = analysis.np, analysis.pd
ROOT = analysis.OUT
TARGET = ROOT / 'figures_matlab_style'
ORIGINAL = Path('D:/papaer4/paper4全球植被恢复力/论文绘图所用数据')


def cells(items):
    return np.array(items, dtype=object)


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    TARGET.mkdir(exist_ok=True)
    inputs = []

    def record(path):
        path = Path(path)
        inputs.append(dict(path=str(path), sha256=digest(path)))
        return path

    table = pd.read_parquet(record(analysis.ANALYSIS / 'kNDVI_pixel_statistics.parquet'))
    ids = table.pixel_id.to_numpy() - 200 * 7200
    assert ids.min() >= 0 and ids.max() < 2800 * 7200
    assert len(np.unique(ids)) == len(ids)
    forest = table.vegetation.eq('Forest').to_numpy()
    slope = table.TAC_slope_22.to_numpy()
    mean_map = analysis.dense(ids, slope)
    valid = np.isfinite(mean_map)
    numerator = uniform_filter(np.nan_to_num(mean_map), size=20, mode='constant')
    denominator = uniform_filter(valid.astype('float32'), size=20, mode='constant')
    smoothed = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                         where=denominator > 0)
    smoothed[~valid] = np.nan
    category = analysis.read(record(analysis.ANALYSIS / 'kNDVI_decoupling_category.tif'))
    expected = np.full(len(ids), np.nan)
    joint = (table.TAC_pvalue_22.to_numpy() < .1) & (table.greenness_pvalue.to_numpy() < .1)
    green = table.greenness_slope.to_numpy()
    for code, mask in [(1, (green > 0) & (slope > 0)), (2, (green < 0) & (slope > 0)),
                       (3, (green < 0) & (slope < 0)), (4, (green > 0) & (slope < 0))]:
        expected[joint & mask] = code
    np.testing.assert_equal(category.ravel()[ids], expected)
    veg_map = analysis.dense(ids, forest.astype('float32'))
    cf = np.where(veg_map == 1, category, np.nan)[::4, ::4]
    cn = np.where(veg_map == 0, category, np.nan)[::4, ::4]
    cmap256 = analysis.plt.get_cmap('PiYG_r')(np.linspace(0, 1, 256))[:, :3]
    cmap = np.concatenate([cmap256[:109], cmap256[146:]])
    times = pd.read_csv(record(analysis.ANALYSIS / 'kNDVI_regional_trajectories.csv'))
    fits = pd.read_csv(record(analysis.ANALYSIS / 'kNDVI_regional_OLS_HAC.csv'))
    lat = pd.read_csv(record(analysis.ANALYSIS / 'Fig1_latitude.csv'))
    categories = pd.read_csv(record(analysis.ANALYSIS / 'kNDVI_decoupling.csv'))
    vegetation = ['Global', 'Forest', 'Non-forest']
    zones = list(analysis.ZONE.values())
    years = np.arange(2004, 2026)
    means = np.full((22, 5, 3), np.nan)
    half = np.full_like(means, np.nan)
    fit_values = np.full((5, 3, 4), np.nan)
    fractions = np.zeros((5, 2, 4))
    for i, veg in enumerate(vegetation):
        for j, zone in enumerate(zones):
            group = times.loc[times.vegetation.eq(veg) & times.zone.eq(zone)].sort_values('reference_year')
            np.testing.assert_array_equal(group.reference_year, years)
            means[:, j, i], half[:, j, i] = group.mean_TAC, group.ci95_half
            fit = fits.loc[fits.vegetation.eq(veg) & fits.zone.eq(zone)
                           & fits.period.eq('22_periods') & fits.method.eq('HAC4')].iloc[0]
            fit_values[j, i] = fit[['slope', 'ci95_low', 'ci95_high', 'p_value']].to_numpy()
            if i:
                group = categories.loc[categories.vegetation.eq(veg) & categories.zone.eq(zone)].sort_values('category_code')
                np.testing.assert_array_equal(group.category_code, [1, 2, 3, 4])
                assert abs(group.percent.sum() - 100) < 1e-8
                fractions[j, i-1] = group.percent
    lat_values = np.full((28, 2, 3), np.nan)
    for i, veg in enumerate(vegetation[1:]):
        group = lat.loc[lat.vegetation.eq(veg)].sort_values('latitude')
        lat_values[:len(group), i] = group[['latitude', 'mean', 'ci95_half']].to_numpy()
    manifest_path = record(Path('D:/papaer4/20260604返修补实验/input/GEE_recomputed_20260927/RF_input_manifest_22.json'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    lookup = {r['variable']: r for r in manifest['predictors']}
    precip, temp = [analysis.read(record(lookup[v]['path'])).ravel()[ids] for v in ['P', 'Ta']]
    latitude = 80 - (ids // 7200 + .5) * .05
    bins, climate, counts = analysis.climate_grid(slope, precip, temp, np.cos(np.deg2rad(latitude)))
    coast = np.empty((1, len(analysis.COAST)), dtype=object)
    for i, line in enumerate(analysis.COAST):
        coast[0, i] = line
    rank_values = np.empty((2, 1), dtype=object)
    rank_labels = np.empty_like(rank_values)
    curve_values = np.empty((2, 6), dtype=object)
    curve_labels = np.empty_like(curve_values)
    curve_units = np.empty_like(curve_values)
    importance_values = np.zeros((2, 6, 3))
    performance = np.zeros((2, 3))
    tex = {'P_SI': 'P.SI', 'P_NIPV': 'P.NIPV', 'dTa': r'\deltaT_a',
           'dkNDVI': r'\deltakNDVI', 'P_ET': 'P - ET', 'dP': r'\deltaP',
           'Forest_fraction': 'Forest fraction', 'Rs': 'R_s'}
    units = {'dTa': r'(^\circC year^{-1})', 'dkNDVI': '(year^{-1})',
             'P_SI': '(-)', 'P_NIPV': '(-)', 'Sand': '(g kg^{-1})',
             'Nitrogen': '(cg kg^{-1})', 'Elevation': '(m)', 'Rs': '(MJ m^{-2} day^{-1})'}
    tops = []
    for i, veg in enumerate(vegetation[1:]):
        model = ROOT / 'RF_endpoints_2005_2023_QA01_repeated30' / veg / 'summary'
        ranks = pd.read_csv(record(model / 'importance_summary.csv')).dropna(subset=['standardized_importance'])
        protocol = json.loads(record(ROOT / 'PDP' / veg / 'protocol.json').read_text(encoding='utf-8'))
        assert ranks.variable.head(6).tolist() == protocol['top']
        tops.append(protocol['top'])
        columns = ['standardized_importance', 'standardized_importance_q025', 'standardized_importance_q975']
        rank_values[i, 0] = ranks.head(10)[columns].to_numpy()
        rank_labels[i, 0] = cells([tex.get(v, v) for v in ranks.variable.head(10)])
        metrics = pd.read_csv(record(model / 'performance_summary.csv')).set_index('metric')
        performance[i] = metrics.loc['accuracy', ['median', 'q025', 'q975']].to_numpy() * 100
        curves = pd.read_csv(record(ROOT / 'PDP' / veg / 'PDP_summary.csv'))
        for j, var in enumerate(protocol['top']):
            c = curves.loc[curves.variable.eq(var) & curves.zone.eq('Global')].sort_values('x')
            assert (c.n_models == 30).all() and (c.n_background == 5000).all()
            assert (c.q025 <= c['median']).all() and (c['median'] <= c.q975).all()
            a = c[['x', 'median', 'q025', 'q975']].to_numpy()
            if var == 'Rs':
                a[:, 0] *= 1e-6
            a[:, 1:] *= 100
            curve_values[i, j] = a
            curve_labels[i, j] = tex.get(var, var)
            curve_units[i, j] = units[var]
            importance_values[i, j] = ranks.set_index('variable').loc[var, columns].to_numpy()
    payload = dict(trend_map=smoothed[::4, ::4], category_forest=cf, category_nonforest=cn,
        longitude=-180+(np.arange(0, 7200, 4)+.5)*.05,
        latitude=80-(np.arange(0, 2800, 4)+.5)*.05,
        cmap=cmap, cmap20=analysis.plt.get_cmap('PiYG_r')(np.linspace(0, 1, 20))[:, :3],
        curve_colors=analysis.plt.get_cmap('PiYG_r')([1/9, 8/9])[:, :3],
        category_colors=np.array([[227, 26, 28], [251, 154, 153], [31, 120, 180], [166, 206, 227]])/255,
        climate_colors=np.array([[150, 150, 150], [0, 176, 0], [255, 87, 20], [235, 215, 0], [105, 219, 255]])/255,
        coast=coast, years=years, tac_mean=means, tac_half=half, hac4=fit_values,
        vegetation=cells(vegetation), zones=cells(zones), lat_values=lat_values,
        climate_precip_edges=bins[0], climate_temp_edges=bins[1], climate_grid=climate.T,
        climate_count=counts.T, fractions=fractions, rank_values=rank_values, rank_labels=rank_labels,
        curve_values=curve_values, curve_labels=curve_labels, curve_units=curve_units,
        importance_values=importance_values, performance=performance)
    savemat(TARGET / 'plot_inputs.mat', payload, do_compression=True)
    templates = [ORIGINAL / 'FIG1/Fig1_V5.m',
                 ORIGINAL / 'FIG2/不同气候分区森林和非森林时间变化曲线绘图.m',
                 ORIGINAL / 'FIG3/KNDVI_TAC变化加空心饼图方向绘图v3.m',
                 ORIGINAL / 'FIG4/分类模型_land_cover_PDP绘图.m']
    report = dict(status='prepared', output=str(TARGET), inputs=inputs,
        templates=[dict(path=str(p), sha256=digest(p)) for p in templates],
        new_plot_input_sha256=digest(TARGET / 'plot_inputs.mat'), model_retraining=False,
        inherited='MATLAB layouts, PiYG_r, Paired categories, original climate colors, Arial',
        main_TAC_reference_years=[2004, 2025], RF_reference_years=[2004, 2022], RF_top6=tops,
        uncertainty='Unscaled saved intervals. Regional pixel-mean intervals are descriptive; RF/PDP ranges are refit percentiles.',
        changes_from_legacy=['No 10x interval inflation', 'No hard-coded old slopes or significance',
                             'Standardized OOB scores are not percentages', 'No uncomputed climate-bin significance dots'],
        map_display='Same existing 20-cell continuous smoothing and stride-4 preview; categories never averaged',
        original_files_modified=False)
    (TARGET / 'provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared {TARGET}; unchanged saved science; checked categories, years, PDPs and ranks.', flush=True)


if __name__ == '__main__':
    main()
