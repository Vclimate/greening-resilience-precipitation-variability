"""Redraw existing SI spatial results at a consistent geographic aspect ratio."""
import hashlib
import json
from pathlib import Path

import analyse_plot_QA01_unified22_20261005 as a

np, pd, plt = a.np, a.pd, a.plt
OUT = a.source.prep.ROOT / 'v15 - JRS/QA01_integrated_results_20261006/figures/spatial_ratio_v2'
RECORDS = []
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9})


def finish_map(ax):
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=(-180, 180), ylim=(-60, 80), xticks=[-120, 0, 120], yticks=[-40, 0, 40, 80])
    ax.tick_params(length=3, width=.7, pad=3)
    for edge in ax.spines.values():
        edge.set_linewidth(.7)


def map_panel(ax, ids, values, title, limit):
    im = a.map_panel(ax, ids, values, title, limit)
    finish_map(ax)
    return im


def colourbar(fig, image, ax, label='', shrink=.55):
    cb = fig.colorbar(image, ax=ax, orientation='horizontal', fraction=.035,
                     pad=.085, shrink=shrink, aspect=30)
    cb.set_label(label, labelpad=3)
    cb.ax.tick_params(length=3, pad=2, labelsize=8.5)
    return cb


def save(fig, name):
    fig.canvas.draw()
    ratios = []
    for ax in fig.axes:
        if tuple(ax.get_xlim()) == (-180., 180.) and tuple(ax.get_ylim()) == (-60., 80.):
            box = ax.get_window_extent()
            ratio = box.width / box.height
            np.testing.assert_allclose(ratio, 360/140, rtol=1e-6)
            ratios.append(ratio)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=350, bbox_inches='tight', facecolor='white')
    RECORDS.append(dict(figure=name, map_panel_width_height_ratios=ratios,
                        png=str(OUT/f'{name}.png'), pdf=str(OUT/f'{name}.pdf')))
    plt.close(fig)
    print(f'Saved {name}', flush=True)


def overview(ids, forest, zone, weights, manifest, valid):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), width_ratios=[1.55, 1], layout='constrained')
    values = a.dense(ids, np.where(valid, np.where(forest, 1, 2), np.nan))
    axes[0].imshow(values[::4, ::4], origin='upper', extent=[-180, 180, -60, 80],
                   cmap=a.ListedColormap([a.GREEN, a.PINK]), vmin=.5, vmax=2.5, interpolation='nearest')
    finish_map(axes[0])
    axes[0].set_title('a  Retained vegetation domain', loc='left', fontweight='bold')
    if a.COAST:
        axes[0].add_collection(a.LineCollection(a.COAST, colors='#777777', linewidths=.3))
    for i, veg in enumerate(['Forest', 'Non-forest']):
        take = valid & (forest if veg == 'Forest' else ~forest)
        shares = np.array([weights[take & (zone == z)].sum() for z in range(1, 5)])
        denominator = weights[take & np.isin(zone, [1, 2, 3, 4])].sum()
        axes[1].bar(np.arange(4)+(i-.5)*.36, 100*shares/denominator,
                    width=.36, label=veg, color=a.COLORS[veg])
    axes[1].set(xticks=range(4), xticklabels=list(a.ZONE.values())[1:], ylabel='Area fraction (%)')
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].set_title('b  Climate-zone composition', loc='left', fontweight='bold')
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, 'FigS1_map_ratio_v2')

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.2), layout='constrained')
    for ax, variable, panel in zip(axes, ['P_SI', 'P_NIPV'], ['a', 'b']):
        spec = next(s for s in manifest['predictors'] if s['variable'] == variable)
        v = a.source.prep.package.mat(spec).ravel()[ids]
        image = a.dense(ids, np.where(valid, v, np.nan))
        im = ax.imshow(image[::4, ::4], extent=[-180, 180, -60, 80], origin='upper',
                       cmap='BrBG', vmin=0, vmax=float(np.nanquantile(v, .99)), interpolation='nearest')
        finish_map(ax)
        ax.set_title(panel+'  '+variable.replace('_', '.')+' (-)', loc='left', fontweight='bold')
        if a.COAST:
            ax.add_collection(a.LineCollection(a.COAST, colors='#777777', linewidths=.3))
        colourbar(fig, im, ax, shrink=.55)
    save(fig, 'FigS2_map_ratio_v2')


def three_panels(ids, values, forest, weights, precip, temp, title, stem, limit, number):
    fig = plt.figure(figsize=(8.5, 7), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=.07, wspace=.06)
    ax = fig.add_subplot(grid[0, :])
    im = map_panel(ax, ids, values, 'a  '+title, limit)
    colourbar(fig, im, ax, r'yr$^{-1}$' if 'trend' in title else 'TAC', shrink=.5)
    bins = pd.read_csv(a.ANALYSIS/f'{stem}_latitude.csv')
    ax = fig.add_subplot(grid[1, 0])
    for veg in ['Forest', 'Non-forest']:
        g = bins.loc[bins.vegetation.eq(veg)]
        ax.plot(g['mean'], g.latitude, color=a.COLORS[veg], lw=1.3, label=veg)
        ax.fill_betweenx(g.latitude, g['mean']-g.ci95_half, g['mean']+g.ci95_half,
                         color=a.COLORS[veg], alpha=.2)
    ax.axvline(0, color='#777777', lw=.5)
    ax.set(ylabel='Latitude (degrees)', xlabel='TAC' if number == 3 else r'kNDVI trend (yr$^{-1}$)', ylim=(-60, 80))
    ax.set_title('b  Latitudinal profile', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='best')
    ax = fig.add_subplot(grid[1, 1])
    bins2, mean, _ = a.climate_grid(values, precip, temp, weights)
    im2 = ax.pcolormesh(*bins2, mean.T, cmap='PiYG_r', vmin=-limit, vmax=limit, shading='flat')
    ax.set(xlabel=r'Precipitation (mm yr$^{-1}$)', ylabel='Temperature (degrees C)')
    ax.set_title('c  Climate gradients', loc='left', fontweight='bold')
    cb = fig.colorbar(im2, ax=ax, fraction=.04, pad=.025)
    cb.ax.tick_params(labelsize=8)
    save(fig, f'FigS{number}_map_ratio_v2')


def window(ids, forest, weights, valid):
    years = np.arange(2009, 2026)
    series, slopes, pvalues = {}, {}, {}
    for w in [5, 7, 10]:
        print(f'Reading saved {w}-year annual TAC for unchanged map statistics', flush=True)
        y, values = a.compact_stack('kNDVI', w, 1, ids)
        values = values[np.isin(y, years)]
        series[w] = values
        slopes[w], pvalues[w], _ = a.source.prep.fit_pixels(values, years)
    common = valid & np.logical_and.reduce([np.isfinite(v).all(axis=0) for v in series.values()])
    saved = json.loads((a.ANALYSIS/'window_sensitivity.json').read_text(encoding='utf-8'))
    assert common.sum() == saved['common_pixels']
    for w in [7, 10]:
        checked = a.spatial_comparison(slopes[5], pvalues[5], slopes[w], pvalues[w], weights, common)
        np.testing.assert_allclose(checked['area_weighted_correlation'], saved['comparisons'][str(w)]['area_weighted_correlation'], atol=1e-12, rtol=0)
    limit = max(.005, float(np.nanquantile(np.abs(slopes[5][common]), .98)))
    fig = plt.figure(figsize=(9, 8.3), layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], hspace=.08, wspace=.05)
    maps=[]
    for i,w in enumerate([5, 7, 10]):
        ax=fig.add_subplot(grid[i, 0])
        im=map_panel(ax, ids, np.where(common, slopes[w], np.nan), f'{chr(97+i)}  {w}-year window', limit)
        maps.append(ax)
    colourbar(fig, im, maps, r'TAC trend (yr$^{-1}$)', shrink=.7)
    for i,w in enumerate([7, 10]):
        ax=fig.add_subplot(grid[i, 1])
        ax.hexbin(slopes[5][common][::10], slopes[w][common][::10], gridsize=55,
                  cmap='Greys', mincnt=1, extent=(-limit, limit, -limit, limit))
        ax.plot([-limit, limit], [-limit, limit], ':', color='#777777')
        ax.set(xlabel='5-year TAC trend', ylabel=f'{w}-year TAC trend', xlim=(-limit, limit), ylim=(-limit, limit))
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{chr(100+i)}  r = {saved['comparisons'][str(w)]['area_weighted_correlation']:.3f}", loc='left', fontweight='bold')
    ax=fig.add_subplot(grid[2, 1])
    regional=pd.read_csv(a.ANALYSIS/'window_regional_OLS_HAC.csv')
    for i,w in enumerate([5,7,10]):
        for j,veg in enumerate(['Global','Forest','Non-forest']):
            r=regional.loc[regional.window_years.eq(w) & regional.vegetation.eq(veg) & regional.method.eq('HAC4')].iloc[0]
            ax.errorbar(j+(i-1)*.19, r.slope*1000,
                        yerr=[[1000*(r.slope-r.ci95_low)],[1000*(r.ci95_high-r.slope)]],
                        fmt=['o','s','^'][i], color=a.COLORS[veg], capsize=2,
                        label=f'{w} years' if j == 0 else None)
    ax.axhline(0,color='#777777',lw=.6)
    ax.set(xticks=range(3),xticklabels=['Global','Forest','Non-forest'],ylabel=r'Trend ($10^{-3}$ yr$^{-1}$)')
    ax.tick_params(axis='x',labelsize=8)
    ax.legend(frameon=False,fontsize=8)
    ax.set_title('f  Regional trends',loc='left',fontweight='bold')
    save(fig,'FigS8_map_ratio_v2')
    del series
    return limit


def lag_and_evi(ids, valid, weights, limit):
    series, slopes = {}, {}
    for lag in [1,2,3,5]:
        print(f'Reading saved lag-{lag} TAC',flush=True)
        years,values=a.compact_stack('kNDVI',5,lag,ids)
        series[lag]=values
        slopes[lag]=a.source.prep.fit_pixels(values,years)[:2]
    common=valid & np.logical_and.reduce([np.isfinite(v).all(axis=0) for v in series.values()])
    lagfolder=a.OUT/'lag_sensitivity_integrated_20261006'
    saved=json.loads((lagfolder/'verification.json').read_text(encoding='utf-8'))
    assert common.sum()==saved['n_common']
    expected=pd.read_csv(lagfolder/'spatial_comparison.csv')
    for lag in [2,3,5]:
        checked=a.spatial_comparison(*slopes[1],*slopes[lag],weights,common)
        r=expected.loc[expected.vegetation.eq('Global') & expected.zone.eq('Global') & expected.TAC_lag.eq(lag)].iloc[0]
        np.testing.assert_allclose(checked['area_weighted_correlation'],r.area_weighted_correlation,atol=1e-12,rtol=0)
    fig,axes=plt.subplots(2,2,figsize=(9,4.8),layout='constrained')
    for ax,lag,panel in zip(axes.ravel(),[1,2,3,5],['a','b','c','d']):
        im=map_panel(ax,ids,np.where(common,slopes[lag][0],np.nan),f'{panel}  Lag {lag}',.012)
    colourbar(fig,im,list(axes.ravel()),r'TAC trend (yr$^{-1}$)',shrink=.55)
    save(fig,'FigS10_map_ratio_v2')

    evi_missing=a.read(a.DATA/'EVI/greenness_2000_2025.tif',6).ravel()[ids]
    print('Reading saved EVI lag-1 TAC',flush=True)
    eviy,evi=a.compact_stack('EVI',5,1,ids)
    assert np.array_equal(eviy,years)
    es,ep,_=a.source.prep.fit_pixels(evi,years)
    matched=valid & np.isfinite(evi_missing) & (evi_missing<.5)
    matched &= np.isfinite(evi).all(axis=0) & np.isfinite(series[1]).all(axis=0)
    saved=json.loads((a.ANALYSIS/'EVI_robustness.json').read_text(encoding='utf-8'))
    assert matched.sum()==saved['common_pixels']
    checked=a.spatial_comparison(*slopes[1],es,ep,weights,matched)
    np.testing.assert_allclose(checked['area_weighted_correlation'],saved['comparisons']['area_weighted_correlation'],atol=1e-12,rtol=0)
    fig=plt.figure(figsize=(9,6.7),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[.85,1],hspace=.07,wspace=.08)
    annual=a.read(a.ANALYSIS/'EVI_greenness_trend_26.tif').ravel()[ids]
    for j,(values,title,scale) in enumerate([(annual,'a  Annual EVI trend',max(.0005,float(np.nanquantile(np.abs(annual[matched]),.98)))),
                                            (es,'b  EVI TAC trend',limit)]):
        ax=fig.add_subplot(grid[0,j])
        im=map_panel(ax,ids,np.where(matched,values,np.nan),title,scale)
        colourbar(fig,im,ax,r'yr$^{-1}$',shrink=.72)
    cats=pd.concat([pd.read_csv(a.ANALYSIS/f'{v}_decoupling.csv') for v in ['kNDVI','EVI']])
    ax=fig.add_subplot(grid[1,0])
    labels=[]
    for i,(v,veg) in enumerate([(v,veg) for veg in ['Forest','Non-forest'] for v in ['kNDVI','EVI']]):
        g=cats.loc[cats.variant.eq(v) & cats.vegetation.eq(veg) & cats.zone.eq('Global')]
        bottom=0
        for row,color in zip(g.itertuples(),a.CATCOLORS):
            ax.bar(i,row.percent,bottom=bottom,color=color)
            bottom+=row.percent
        labels.append(f'{veg}\n{v}')
    ax.set(xticks=range(4),xticklabels=labels,ylabel='Joint-significant area (%)',ylim=(0,100))
    ax.tick_params(axis='x',labelsize=8)
    ax.set_title('c  Greening-resilience classes',loc='left',fontweight='bold')
    ax=fig.add_subplot(grid[1,1])
    ax.hexbin(slopes[1][0][matched][::10],es[matched][::10],gridsize=55,mincnt=1,cmap='Greys',extent=(-limit,limit,-limit,limit))
    ax.plot([-limit,limit],[-limit,limit],':',color='#777777')
    ax.set(xlabel='kNDVI TAC trend',ylabel='EVI TAC trend',xlim=(-limit,limit),ylim=(-limit,limit))
    ax.set_aspect('equal',adjustable='box')
    ax.set_title(f"d  r = {checked['area_weighted_correlation']:.3f}",loc='left',fontweight='bold')
    save(fig,'FigS9_map_ratio_v2')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_paths=list(a.ANALYSIS.glob('*.csv'))+list(a.ANALYSIS.glob('*.json'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    manifest,support,lc,old,base=a.source.old.masks()
    ids=np.flatnonzero(base)
    weights=np.cos(np.deg2rad(80-(ids//7200+.5)*.05))
    forest=lc.ravel()[ids]<=5
    zone=a.source.prep.package.mat(support['climate_zone']).ravel()[ids]
    missing=a.read(a.DATA/'kNDVI/greenness_2000_2025.tif',6).ravel()[ids]
    valid=np.isfinite(missing) & (missing<.5)
    overview(ids,forest,zone,weights,manifest,valid)
    lookup={r['variable']:r for r in manifest['predictors']}
    precip,temp=[a.read(lookup[v]['path']).ravel()[ids] for v in ['P','Ta']]
    values=a.read(a.ANALYSIS/'kNDVI_mean_TAC_22.tif').ravel()[ids]
    three_panels(ids,values,forest,weights,precip,temp,'Long-term mean TAC','FigS3_QA01_unified22',.8,3)
    table=pd.read_parquet(a.ANALYSIS/'kNDVI_pixel_statistics.parquet',columns=['pixel_id','greenness_slope'])
    np.testing.assert_array_equal(table.pixel_id.to_numpy()-200*7200,ids)
    values=table.greenness_slope.to_numpy()
    limit=max(.0005,float(np.nanquantile(np.abs(values),.97)))
    three_panels(ids,values,forest,weights,precip,temp,'kNDVI greenness trend','FigS4_QA01_unified22',limit,4)
    limit=window(ids,forest,weights,valid)
    lag_and_evi(ids,valid,weights,limit)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'map_ratio_verification.json').write_text(json.dumps(dict(status='passed',
        source_statistics_unchanged=True,no_TAC_recomputation=True,no_RF_retraining=True,
        map_coordinate_aspect='Equal longitude/latitude coordinate units; width:height = 360:140',
        display_smoothing_and_colour_limits_unchanged=True,figures=RECORDS),indent=2),encoding='utf-8')
    print('All seven spatial figures redrawn; cached comparison metrics match exactly.',flush=True)


if __name__=='__main__':
    main()
