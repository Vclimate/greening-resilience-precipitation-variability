"""Traceable main/sensitivity statistics and figures from unified local TAC."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from pathlib import Path

import recompute_QA01_22periods_20261005 as source
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from cartopy.io.shapereader import Reader
from scipy.ndimage import uniform_filter
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

np, pd, rasterio = source.np, source.pd, source.rasterio
OUT, DATA = source.OUT, source.DATA
FIG = OUT/'figures'
ANALYSIS = OUT/'analysis'
GREEN, PINK = plt.get_cmap('PiYG_r')(1/9), plt.get_cmap('PiYG_r')(8/9)
COLORS = {'Global': '#666666', 'Forest': GREEN, 'Non-forest': PINK}
ZONE = {0: 'Global', 1: 'Tropical', 2: 'Arid', 3: 'Temperate', 4: 'Boreal'}
CATEGORY = {1: 'Greening + resilience decline', 2: 'Browning + resilience decline',
            3: 'Browning + resilience increase', 4: 'Greening + resilience increase'}
CATCOLORS = ['#E31A1C', '#FB9A99', '#1F78B4', '#A6CEE3']
COAST = []
for directory in [Path('C:/Users/user/.local/share/cartopy/shapefiles/natural_earth/physical'),
                  Path('C:/Users/user/AppData/Local/cartopy/shapefiles/natural_earth/physical')]:
    path = directory/'ne_110m_coastline.shp'
    if path.exists():
        for geom in Reader(path).geometries():
            for line in getattr(geom, 'geoms', [geom]):
                COAST.append(np.column_stack(line.xy))
        break
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
                     'axes.titlesize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def read(path, bands=1):
    with rasterio.open(path) as f:
        assert f.crs.to_epsg() == 4326 and f.shape == (2800, 7200)
        np.testing.assert_allclose(list(f.transform), list(source.core.GRID), atol=0, rtol=0)
        return f.read(bands, masked=True).filled(np.nan)


def compact_stack(variant, w, lag, ids):
    years = np.arange(max(2005, 2000+w), 2027)
    result = np.empty((len(years), len(ids)), np.float32)
    for j, y in enumerate(years):
        path = DATA/variant/f'lag{lag}_{w}year'/f'{variant}_TAC_lag{lag}_{w}year_{y}_01_01_QA01.tif'
        result[j] = read(path).ravel()[ids]
    result[np.abs(result) > 1] = np.nan
    return years-1, result


def weighted_mean(a, w):
    good = np.isfinite(a)
    return float(np.dot(a[good], w[good])/w[good].sum()) if good.any() else np.nan


def moments(a, w):
    good = np.isfinite(a)
    if good.sum() < 2:
        return weighted_mean(a, w), np.nan
    v, weight = a[good].astype(float), w[good]
    mean = np.dot(v, weight)/weight.sum()
    variance = np.dot((v-mean)**2, weight)/weight.sum()
    neff = weight.sum()**2/np.dot(weight, weight)
    return float(mean), float(1.96*np.sqrt(variance/neff))


def fit_trajectory(years, values, variant, vegetation, zone, length, period):
    good = np.isfinite(values)
    y, v = years[good], values[good]
    model = sm.OLS(v, sm.add_constant(y-y.mean())).fit()
    independent = stats.linregress(y, v)
    np.testing.assert_allclose(model.params[1], independent.slope, rtol=1e-10, atol=1e-14)
    rows = []
    for lag in range(5):
        fit = model if lag == 0 else model.get_robustcov_results(
            cov_type='HAC', maxlags=lag, kernel='bartlett', use_correction=True, use_t=True)
        np.testing.assert_array_equal(fit.params, model.params)
        low, high = fit.conf_int()[1]
        rows.append(dict(variant=variant, vegetation=vegetation, zone=zone,
                         window_years=length, period=period, n_years=len(y),
                         first_reference_year=int(y[0]), last_reference_year=int(y[-1]),
                         method='OLS' if lag == 0 else f'HAC{lag}', slope=float(fit.params[1]),
                         stderr=float(fit.bse[1]), ci95_low=float(low), ci95_high=float(high),
                         p_value=float(fit.pvalues[1])))
    lb = acorr_ljungbox(model.resid, lags=[1, 2, 3, 4], return_df=True)
    diagnostic = dict(variant=variant, vegetation=vegetation, zone=zone, window_years=length,
                      period=period, residual_lag1=float(np.corrcoef(model.resid[:-1], model.resid[1:])[0, 1]),
                      durbin_watson=float(durbin_watson(model.resid)),
                      ljung_box_lag4_p=float(lb.loc[4, 'lb_pvalue']))
    return rows, diagnostic


def raster(name, ids, values):
    a = np.full((2800, 7200), -9999, np.float32)
    a.ravel()[ids] = np.where(np.isfinite(values), values, -9999)
    path = ANALYSIS/f'{name}.tif'
    with rasterio.open(path, 'w', **source.prep.profile(1)) as f:
        f.write(a, 1)
        f.update_tags(qa='0/1', preprocessing='2000-2025', source=str(DATA))
    return path


def dense(ids, values):
    a = np.full((2800, 7200), np.nan, np.float32)
    a.ravel()[ids] = values
    return a


def map_panel(ax, ids, values, title, limit=None, categorical=False):
    a = dense(ids, values)
    if categorical:
        cmap, norm = ListedColormap(CATCOLORS), None
        kwargs = dict(vmin=.5, vmax=4.5, interpolation='nearest')
    else:
        good = np.isfinite(a)
        num = uniform_filter(np.nan_to_num(a), size=20, mode='constant')
        den = uniform_filter(good.astype(np.float32), size=20, mode='constant')
        a = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        a[~good] = np.nan
        cmap, norm = 'PiYG_r', TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
        kwargs = dict(norm=norm, interpolation='nearest')
    im = ax.imshow(a[::4, ::4], extent=[-180, 180, -60, 80], origin='upper',
                   cmap=cmap, aspect='auto', **kwargs)
    ax.set(xlim=(-180, 180), ylim=(-60, 80), xticks=[-120, 0, 120], yticks=[-40, 0, 40, 80])
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_facecolor('#F5F5F5')
    if COAST:
        ax.add_collection(LineCollection(COAST, colors='#777777', linewidths=.3))
    return im


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(FIG/f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def climate_grid(values, precipitation, temperature, weights):
    good = np.isfinite(values) & np.isfinite(precipitation) & np.isfinite(temperature)
    x, y, z, w = [a[good] for a in [precipitation, temperature, values, weights]]
    bins = [np.linspace(0, 5000, 51), np.linspace(-10, 30, 41)]
    sums = np.histogram2d(x, y, bins=bins, weights=z*w)[0]
    denominator = np.histogram2d(x, y, bins=bins, weights=w)[0]
    count = np.histogram2d(x, y, bins=bins)[0]
    mean = np.divide(sums, denominator, out=np.full_like(sums, np.nan), where=denominator > 0)
    mean[count < 10] = np.nan
    return bins, mean, count


def latitudes(values, lat, weights, forest):
    records = []
    for veg, mask in [('Forest', forest), ('Non-forest', ~forest)]:
        for low in np.arange(-60, 80, 5):
            take = mask & (lat >= low) & (lat < low+5) & np.isfinite(values)
            if take.sum() < 2:
                continue
            mean, half = moments(values[take], weights[take])
            records.append(dict(vegetation=veg, latitude=low+2.5, mean=mean, ci95_half=half,
                                n_pixels=int(take.sum())))
    return pd.DataFrame(records)


def three_panels(ids, values, forest, lat, weights, p, t, title, stem, limit, compact=False):
    fig = plt.figure(figsize=(10, 6) if compact else (10, 5))
    grid = (fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=.65, wspace=.4) if compact else
            fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[2.4, .9, 1.1], hspace=.5, wspace=.55))
    ax = fig.add_subplot(grid[0, :] if compact else grid[0, :2])
    im = map_panel(ax, ids, values, f'a  {title}', limit)
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.07, pad=.16, label=r'yr$^{-1}$' if 'trend' in title else 'TAC')
    ax = fig.add_subplot(grid[1, 0] if compact else grid[0, 2])
    bins = latitudes(values, lat, weights, forest)
    for veg in ['Forest', 'Non-forest']:
        g = bins.loc[bins.vegetation.eq(veg)]
        ax.plot(g['mean'], g.latitude, color=COLORS[veg], label=veg)
        ax.fill_betweenx(g.latitude, g['mean']-g.ci95_half, g['mean']+g.ci95_half, color=COLORS[veg], alpha=.2)
    ax.axvline(0, color='#777777', lw=.5)
    ax.set(title='b  Latitude', ylabel='Latitude (degrees)', ylim=(-60, 80))
    if compact:
        ax.legend(frameon=False, fontsize=7)
    bins.to_csv(ANALYSIS/f'{stem}_latitude.csv', index=False)
    ax = fig.add_subplot(grid[1, 1] if compact else grid[1, 0])
    bins2, mean, count = climate_grid(values, p, t, weights)
    im2 = ax.pcolormesh(*bins2, mean.T, cmap='PiYG_r', vmin=-limit, vmax=limit, shading='flat')
    ax.set(title='c  Climate gradients', xlabel='Precipitation (mm yr$^{-1}$)', ylabel='Temperature (degrees C)')
    fig.colorbar(im2, ax=ax, fraction=.045, pad=.04)
    return fig, grid


def analyse(variant):
    assert rf_read(DATA/variant/'complete.json')['status'] == 'completed'
    manifest, support, lc, old, base = source.old.masks()
    ids = np.flatnonzero(base)
    rows = ids//7200
    lat = 80-(rows+.5)*.05
    weights = np.cos(np.deg2rad(lat))
    forest = lc.ravel()[ids] <= 5
    zone = source.prep.package.mat(support['climate_zone']).ravel()[ids]
    g = read(DATA/variant/'greenness_2000_2025.tif', list(range(1, 7))).reshape(6, -1)[:, ids]
    valid_missing = np.isfinite(g[5]) & (g[5] < .5)
    annual = read(DATA/variant/'annual_means_2000_2025.tif', list(range(1, 27))).reshape(26, -1)[:, ids]
    annual[:, ~valid_missing] = np.nan
    greening_slope, greening_p, annual_n = source.prep.fit_pixels(annual, np.arange(2000, 2026))
    years, tac = compact_stack(variant, 5, 1, ids)
    tac[:, ~valid_missing] = np.nan
    slope, pvalue, ntac = source.prep.fit_pixels(tac, years)
    mean_tac = np.divide(np.nansum(tac, axis=0), np.isfinite(tac).sum(axis=0),
                         out=np.full(len(ids), np.nan), where=np.isfinite(tac).sum(axis=0) > 0)
    for name, value in [('TAC_trend_22', slope), ('TAC_p_22', pvalue), ('mean_TAC_22', mean_tac),
                        ('greenness_trend_26', greening_slope), ('greenness_p_26', greening_p)]:
        raster(f'{variant}_{name}', ids, value)
    pd.DataFrame(dict(pixel_id=((rows+200)*7200+ids%7200).astype('int32'),
                      vegetation=np.where(forest, 'Forest', 'Non-forest'), climate_zone=zone,
                      MDN_2000_2025=g[5], TAC_slope_22=slope, TAC_pvalue_22=pvalue,
                      greenness_slope=greening_slope, greenness_pvalue=greening_p)).to_parquet(
                          ANALYSIS/f'{variant}_pixel_statistics.parquet', index=False)
    categories = np.full(len(ids), np.nan)
    joint = np.isfinite(slope) & np.isfinite(greening_slope) & (pvalue < .1) & (greening_p < .1)
    joint &= (slope != 0) & (greening_slope != 0)
    for code, sel in [(1, (greening_slope > 0)&(slope > 0)), (2, (greening_slope < 0)&(slope > 0)),
                      (3, (greening_slope < 0)&(slope < 0)), (4, (greening_slope > 0)&(slope < 0))]:
        categories[joint & sel] = code
    raster(f'{variant}_decoupling_category', ids, categories)
    catrows, greeningrows = [], []
    for veg, mask in [('Forest', forest), ('Non-forest', ~forest)]:
        good = mask & np.isfinite(greening_slope) & valid_missing
        den = weights[good].sum()
        greeningrows.append(dict(variant=variant, vegetation=veg, n_valid=int(good.sum()),
            significant_greening_pct=100*weights[good & (greening_slope > 0) & (greening_p < .1)].sum()/den,
            significant_browning_pct=100*weights[good & (greening_slope < 0) & (greening_p < .1)].sum()/den))
        for zn, zname in ZONE.items():
            take = mask & joint & ((zone == zn) if zn else True)
            denominator = weights[take].sum()
            for code in CATEGORY:
                sel = take & (categories == code)
                catrows.append(dict(variant=variant, vegetation=veg, zone=zname, category_code=code,
                                    category=CATEGORY[code], n_pixels=int(sel.sum()),
                                    denominator_pixels=int(take.sum()),
                                    percent=100*weights[sel].sum()/denominator if denominator else np.nan))
    pd.DataFrame(catrows).to_csv(ANALYSIS/f'{variant}_decoupling.csv', index=False)
    pd.DataFrame(greeningrows).to_csv(ANALYSIS/f'{variant}_greening.csv', index=False)
    trajectories, fits, diagnostics = [], [], []
    for veg, vm in [('Global', np.ones(len(ids), bool)), ('Forest', forest), ('Non-forest', ~forest)]:
        for zn, zname in ZONE.items():
            mask = vm & valid_missing & np.isfinite(tac).all(axis=0) & ((zone == zn) if zn else True)
            if mask.sum() < 3:
                continue
            means = []
            for j, yr in enumerate(years):
                mean, half = moments(tac[j, mask], weights[mask])
                means.append(mean)
                trajectories.append(dict(variant=variant, vegetation=veg, zone=zname, reference_year=int(yr),
                                         endpoint_year=int(yr+1), mean_TAC=mean, ci95_half=half,
                                         n_pixels=int(mask.sum())))
            for last, label in [(22, '22_periods'), (19, 'first19')]:
                fitted, diagnostic = fit_trajectory(years[:last], np.array(means[:last]), variant, veg, zname, 5, label)
                fits.extend(fitted)
                diagnostics.append(diagnostic)
    ts = pd.DataFrame(trajectories)
    ts.to_csv(ANALYSIS/f'{variant}_regional_trajectories.csv', index=False)
    pd.DataFrame(fits).to_csv(ANALYSIS/f'{variant}_regional_OLS_HAC.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(ANALYSIS/f'{variant}_residual_diagnostics.csv', index=False)
    prep_record = dict(variant=variant, main_TAC_endpoints=[2005, 2026], main_reference_years=[2004, 2025],
        greenness_period=[2000, 2025], RF_endpoints=[2005, 2023], RF_reference_years=[2004, 2022],
        original_stable_domain=int(base.sum()), main_missing_lt50=int(valid_missing.sum()),
        same_pixels_each_regional_trajectory=True,
        spatial_CI='Approximate mean intervals treating pixels as independent; do not represent spatially independent uncertainty',
        joint_categories='Both greenness and TAC trend OLS P<0.1; denominator is joint-significant area',
        end_effect='Last years remain edge affected; all22 displayed and first19 reported as sensitivity',
        greening=greeningrows, files_are_new_not_overwrites=True)
    source.prep.dump(ANALYSIS/f'{variant}_summary.json', prep_record)
    if variant == 'kNDVI':
        p_spec = next(s for s in manifest['predictors'] if s['variable'] == 'P')
        t_spec = next(s for s in manifest['predictors'] if s['variable'] == 'Ta')
        precip, temp = [source.prep.package.mat(s).ravel()[ids] for s in [p_spec, t_spec]]
        limit = max(.003, float(np.nanquantile(np.abs(slope), .97)))
        fig, grid = three_panels(ids, slope, forest, lat, weights, precip, temp,
                                'kNDVI TAC trend', 'Fig1', limit)
        ax = fig.add_subplot(grid[1, 1:])
        for veg in ['Global', 'Forest', 'Non-forest']:
            group = ts.loc[ts.vegetation.eq(veg) & ts.zone.eq('Global')]
            ax.plot(group.reference_year, group.mean_TAC, color=COLORS[veg], label=veg, lw=1.2)
            ax.fill_between(group.reference_year, group.mean_TAC-group.ci95_half,
                            group.mean_TAC+group.ci95_half, color=COLORS[veg], alpha=.15)
        ax.axvspan(2015, 2016, color='#CC4444', alpha=.12)
        ax.set(title='d  Regional TAC', xlabel='Reference year', ylabel='TAC', xlim=(2004, 2025))
        ax.set_xticks([2004, 2008, 2012, 2016, 2020, 2025])
        ax.legend(frameon=False, fontsize=7)
        fig.subplots_adjust(bottom=.14)
        save(fig, 'Fig1_QA01_unified22')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for ax, veg, letter in zip(axes, ['Forest', 'Non-forest'], ['a', 'b']):
            for zn, color in zip(ZONE.values(), ['#555555', '#D95F02', '#E6AB02', '#1B9E77', '#7570B3']):
                group = ts.loc[ts.vegetation.eq(veg) & ts.zone.eq(zn)].sort_values('reference_year')
                if group.empty:
                    continue
                baseline = group.mean_TAC.iloc[0]
                ax.plot(group.reference_year, group.mean_TAC-baseline, color=color, label=zn, lw=1.2)
                ax.fill_between(group.reference_year, group.mean_TAC-baseline-group.ci95_half,
                                group.mean_TAC-baseline+group.ci95_half, color=color, alpha=.1)
            ax.axhline(0, color='#888888', lw=.6)
            ax.axvspan(2015, 2016, color='#CC4444', alpha=.1)
            ax.set(title=f'{letter}  {veg}', xlabel='Reference year', ylabel='TAC change from 2004', xlim=(2004, 2025))
            ax.set_xticks([2004, 2008, 2012, 2016, 2020, 2025])
        axes[1].legend(frameon=False, ncol=2, fontsize=7)
        fig.tight_layout()
        save(fig, 'Fig2_QA01_unified22')
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], wspace=.25, hspace=.35)
        for j, veg in enumerate(['Forest', 'Non-forest']):
            mask = forest if veg == 'Forest' else ~forest
            values = np.where(mask, categories, np.nan)
            map_panel(fig.add_subplot(gs[j, 0]), ids, values, f'{chr(97+j)}  {veg}', categorical=True)
            group = pd.DataFrame(catrows).loc[lambda a: a.vegetation.eq(veg) & a.zone.eq('Global')]
            ax = fig.add_subplot(gs[j, 1])
            if group.percent.notna().all() and group.percent.sum() > 0:
                ax.pie(group.percent, colors=CATCOLORS, autopct='%1.1f%%', pctdistance=.77,
                       wedgeprops=dict(width=.45, edgecolor='white'), textprops=dict(fontsize=8))
            ax.set_title(veg)
        fig.legend(handles=[Patch(facecolor=c, label=CATEGORY[k]) for k, c in zip(CATEGORY, CATCOLORS)],
                   loc='lower center', ncol=2, frameon=False, fontsize=7)
        save(fig, 'Fig3_QA01_unified22')
        for v, title, stem, lim in [(mean_tac, 'Long-term mean TAC', 'FigS3_QA01_unified22', .8),
                                   (greening_slope, 'kNDVI greenness trend', 'FigS4_QA01_unified22',
                                    max(.0005, float(np.nanquantile(np.abs(greening_slope), .97))))]:
            fig, grid = three_panels(ids, v, forest, lat, weights, precip, temp, title, stem, lim, compact=True)
            save(fig, stem)
        support_figures(ids, forest, zone, weights, manifest, valid_missing)
    source.prep.log(f'{variant} main statistics and figures COMPLETED')


def support_figures(ids, forest, zone, weights, manifest, valid):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    cmap = ListedColormap([GREEN, PINK])
    values = dense(ids, np.where(valid, np.where(forest, 1, 2), np.nan))
    axes[0].imshow(values[::4, ::4], origin='upper', extent=[-180, 180, -60, 80], cmap=cmap, vmin=.5, vmax=2.5, aspect='auto')
    axes[0].set(title='a  Stable vegetation analysis domain', xlabel='Longitude', ylabel='Latitude')
    if COAST:
        axes[0].add_collection(LineCollection(COAST, colors='#777777', linewidths=.3))
    for i, veg in enumerate(['Forest', 'Non-forest']):
        mask = valid & (forest if veg == 'Forest' else ~forest)
        shares = np.array([weights[mask & (zone == z)].sum() for z in range(1, 5)])
        denominator = weights[mask & np.isin(zone, [1, 2, 3, 4])].sum()
        axes[1].bar(np.arange(4)+(i-.5)*.36, 100*shares/denominator, width=.36, label=veg, color=COLORS[veg])
    axes[1].set(xticks=np.arange(4), xticklabels=list(ZONE.values())[1:], ylabel='Area fraction (%)', title='b  Climate-zone composition')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, 'FigS1_QA01_unified22')
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    for ax, variable in zip(axes, ['P_SI', 'P_NIPV']):
        spec = next(s for s in manifest['predictors'] if s['variable'] == variable)
        v = source.prep.package.mat(spec).ravel()[ids]
        a = dense(ids, np.where(valid, v, np.nan))
        im = ax.imshow(a[::4, ::4], extent=[-180, 180, -60, 80], origin='upper', aspect='auto',
                       cmap='BrBG', vmin=0, vmax=float(np.nanquantile(v, .99)))
        ax.set_title(variable.replace('_', '.')+' (-)', loc='left')
        if COAST:
            ax.add_collection(LineCollection(COAST, colors='#777777', linewidths=.3))
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.06, pad=.08)
    save(fig, 'FigS2_QA01_unified22')


def spatial_comparison(a, ap, b, bp, weights, mask):
    good = mask & np.isfinite(a) & np.isfinite(b)
    w, x, y = weights[good], a[good], b[good]
    xm, ym = np.dot(w, x)/w.sum(), np.dot(w, y)/w.sum()
    corr = np.dot(w, (x-xm)*(y-ym))/np.sqrt(np.dot(w, (x-xm)**2)*np.dot(w, (y-ym)**2))
    sig = good & (ap < .1) & (bp < .1)
    agree = np.sign(a) == np.sign(b)
    return dict(n_common=int(good.sum()), area_weighted_correlation=float(corr),
                sign_agreement_pct=100*float(weights[good & agree].sum()/weights[good].sum()),
                both_significant_n=int(sig.sum()),
                both_significant_direction_agreement_pct=100*float(weights[sig & agree].sum()/weights[sig].sum())
                                                        if sig.any() else None)


def sensitivity():
    manifest, support, lc, old, base = source.old.masks()
    ids = np.flatnonzero(base)
    weights = np.cos(np.deg2rad(80-(ids//7200+.5)*.05))
    g = read(DATA/'kNDVI/greenness_2000_2025.tif', 6).ravel()[ids]
    valid = np.isfinite(g) & (g < .5)
    forest = lc.ravel()[ids] <= 5
    common_years = np.arange(2009, 2026)
    slopes, pvalues, series = {}, {}, {}
    for w in [5, 7, 10]:
        years, values = compact_stack('kNDVI', w, 1, ids)
        values = values[np.isin(years, common_years)]
        assert values.shape[0] == len(common_years)
        series[w] = values
        slopes[w], pvalues[w], n = source.prep.fit_pixels(values, common_years)
    common = valid & np.logical_and.reduce([np.isfinite(v).all(axis=0) for v in series.values()])
    metrics = {str(w): spatial_comparison(slopes[5], pvalues[5], slopes[w], pvalues[w], weights, common) for w in [7, 10]}
    source.prep.dump(ANALYSIS/'window_sensitivity.json', dict(reference_period=[2009, 2025], comparisons=metrics,
                                                          common_pixels=int(common.sum())))
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    lim = max(.005, float(np.nanquantile(np.abs(slopes[5][common]), .98)))
    for ax, w in zip(axes[0], [5, 7, 10]):
        im = map_panel(ax, ids, np.where(common, slopes[w], np.nan), f'{w}-year window', lim)
    fig.colorbar(im, ax=list(axes[0]), orientation='horizontal', fraction=.06, label=r'TAC trend (yr$^{-1}$)')
    for ax, w in zip(axes[1, :2], [7, 10]):
        ax.hexbin(slopes[5][common][::10], slopes[w][common][::10], gridsize=55, cmap='Greys', mincnt=1,
                  extent=(-lim, lim, -lim, lim))
        ax.plot([-lim, lim], [-lim, lim], ':', color='#777777')
        ax.set(xlabel='5-year TAC trend', ylabel=f'{w}-year TAC trend', xlim=(-lim, lim), ylim=(-lim, lim),
               title=f"r = {metrics[str(w)]['area_weighted_correlation']:.3f}")
    ax = axes[1, 2]
    for i, w in enumerate([5, 7, 10]):
        for j, (veg, mask) in enumerate([('Global', np.ones(len(ids), bool)), ('Forest', forest), ('Non-forest', ~forest)]):
            take = common & mask
            means = np.array([weighted_mean(a[take], weights[take]) for a in series[w]])
            fitted, diagnostic = fit_trajectory(common_years, means, 'kNDVI', veg, 'Global', w, 'common17')
            rows.extend(fitted)
            row = fitted[4]
            ax.errorbar(j+(i-1)*.19, row['slope']*1000,
                yerr=np.array([[row['slope']-row['ci95_low']], [row['ci95_high']-row['slope']]])*1000,
                fmt=['o', 's', '^'][i], color=COLORS[veg], capsize=2, label=f'{w} years' if j == 0 else None)
    ax.axhline(0, color='#777777', lw=.6)
    ax.set(xticks=range(3), xticklabels=['Global', 'Forest', 'Non-forest'], ylabel=r'Trend ($10^{-3}$ yr$^{-1}$)')
    ax.legend(frameon=False, fontsize=7)
    pd.DataFrame(rows).to_csv(ANALYSIS/'window_regional_OLS_HAC.csv', index=False)
    save(fig, 'FigS8_QA01_unified22')
    years, main = compact_stack('kNDVI', 5, 1, ids)
    lag_metrics = {}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
    for ax, lag in zip(axes, [2, 3, 5]):
        y, a = compact_stack('kNDVI', 5, lag, ids)
        assert np.array_equal(y, years)
        s, p, n = source.prep.fit_pixels(a, years)
        s0, p0, n0 = source.prep.fit_pixels(main, years)
        same = valid & np.isfinite(a).all(axis=0) & np.isfinite(main).all(axis=0)
        lag_metrics[str(lag)] = spatial_comparison(s0, p0, s, p, weights, same)
        im = map_panel(ax, ids, np.where(same, s, np.nan), f'Lag {lag}', lim)
    fig.colorbar(im, ax=list(axes), orientation='horizontal', fraction=.08, label=r'TAC trend (yr$^{-1}$)')
    save(fig, 'FigS10_QA01_lag_sensitivity')
    source.prep.dump(ANALYSIS/'lag_sensitivity.json', dict(reference_period=[2004, 2025], comparisons=lag_metrics))
    del series, main
    evi_good = read(DATA/'EVI/greenness_2000_2025.tif', 6).ravel()[ids]
    common = valid & np.isfinite(evi_good) & (evi_good < .5)
    slopes = {}
    for variant in ['kNDVI', 'EVI']:
        y, a = compact_stack(variant, 5, 1, ids)
        assert np.array_equal(y, years)
        s, p, n = source.prep.fit_pixels(a, years)
        slopes[variant] = (s, p)
        common &= np.isfinite(a).all(axis=0)
    metric = spatial_comparison(*slopes['kNDVI'], *slopes['EVI'], weights, common)
    source.prep.dump(ANALYSIS/'EVI_robustness.json', dict(reference_period=[2004, 2025], comparisons=metric,
                                                      common_pixels=int(common.sum())))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    annualtrend = read(ANALYSIS/'EVI_greenness_trend_26.tif').ravel()[ids]
    map_panel(axes[0, 0], ids, np.where(common, annualtrend, np.nan), 'a  Annual EVI trend',
              max(.0005, float(np.nanquantile(np.abs(annualtrend[common]), .98))))
    map_panel(axes[0, 1], ids, np.where(common, slopes['EVI'][0], np.nan), 'b  EVI TAC trend', lim)
    cats = pd.concat([pd.read_csv(ANALYSIS/f'{v}_decoupling.csv') for v in ['kNDVI', 'EVI']])
    ax = axes[1, 0]
    labels = []
    for i, (v, veg) in enumerate([(v, veg) for veg in ['Forest', 'Non-forest'] for v in ['kNDVI', 'EVI']]):
        g = cats.loc[cats.variant.eq(v) & cats.vegetation.eq(veg) & cats.zone.eq('Global')]
        bottom = 0
        for row, color in zip(g.itertuples(), CATCOLORS):
            ax.bar(i, row.percent, bottom=bottom, color=color)
            bottom += row.percent
        labels.append(f'{veg}\n{v}')
    ax.set(xticks=range(4), xticklabels=labels, ylabel='Joint-significant area (%)', ylim=(0, 100))
    ax = axes[1, 1]
    ax.hexbin(slopes['kNDVI'][0][common][::10], slopes['EVI'][0][common][::10], gridsize=55, mincnt=1,
              cmap='Greys', extent=(-lim, lim, -lim, lim))
    ax.plot([-lim, lim], [-lim, lim], ':', color='#777777')
    ax.set(xlabel='kNDVI TAC trend', ylabel='EVI TAC trend', xlim=(-lim, lim), ylim=(-lim, lim),
           title=f"d  r = {metric['area_weighted_correlation']:.3f}")
    save(fig, 'FigS9_QA01_unified22')
    source.prep.log('Window, lag and EVI sensitivity COMPLETED')


def rf_read(path):
    return source.rf.read(path)


def redraw_compact():
    manifest, _, _, _, _ = source.old.masks()
    table = pd.read_parquet(ANALYSIS/'kNDVI_pixel_statistics.parquet')
    ids = table.pixel_id.to_numpy()-200*7200
    lat = 90-(table.pixel_id.to_numpy()//7200+.5)*.05
    weights = np.cos(np.deg2rad(lat))
    forest = table.vegetation.eq('Forest').to_numpy()
    lookup = {r['variable']: r for r in manifest['predictors']}
    precip, temp = [read(lookup[v]['path']).ravel()[ids] for v in ['P', 'Ta']]
    mean_tac = read(ANALYSIS/'kNDVI_mean_TAC_22.tif').ravel()[ids]
    green = table.greenness_slope.to_numpy()
    for values, title, stem, limit in [(mean_tac, 'Long-term mean TAC', 'FigS3_QA01_unified22', .8),
                                      (green, 'kNDVI greenness trend', 'FigS4_QA01_unified22',
                                       max(.0005, float(np.nanquantile(np.abs(green), .97))))]:
        fig, _ = three_panels(ids, values, forest, lat, weights, precip, temp, title, stem, limit, compact=True)
        save(fig, stem)
    source.prep.log('S3/S4 layout redrawn from unchanged saved statistics')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['kNDVI', 'EVI', 'sensitivity', 'all', 'compact'], default='all')
    args = p.parse_args()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    if args.stage in ['all', 'kNDVI']:
        analyse('kNDVI')
    if args.stage in ['all', 'EVI']:
        analyse('EVI')
    if args.stage in ['all', 'sensitivity']:
        sensitivity()
    if args.stage == 'compact':
        redraw_compact()


if __name__ == '__main__':
    main()
