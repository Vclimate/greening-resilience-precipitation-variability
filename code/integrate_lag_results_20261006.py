"""Compare four existing QA01 TAC lags on one fixed spatial support."""
import json
import analyse_plot_QA01_unified22_20261005 as a

np, pd, plt = a.np, a.pd, a.plt
OUT = a.OUT / 'lag_sensitivity_integrated_20261006'


def main():
    OUT.mkdir(exist_ok=True)
    manifest, support, lc, old, base = a.source.old.masks()
    ids = np.flatnonzero(base)
    weights = np.cos(np.deg2rad(80 - (ids // 7200 + .5) * .05))
    forest = lc.ravel()[ids] <= 5
    zone = a.source.prep.package.mat(support['climate_zone']).ravel()[ids]
    missing = a.read(a.DATA / 'kNDVI/greenness_2000_2025.tif', 6).ravel()[ids]
    series = {}
    for lag in [1, 2, 3, 5]:
        a.source.prep.log(f'Loading existing lag {lag}')
        years, series[lag] = a.compact_stack('kNDVI', 5, lag, ids)
    common = np.isfinite(missing) & (missing < .5)
    for values in series.values():
        common &= np.isfinite(values).all(axis=0)
    trends = {lag: a.source.prep.fit_pixels(values, years)[:2] for lag, values in series.items()}
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.7), layout='constrained')
    for ax, lag in zip(axes.ravel(), [1, 2, 3, 5]):
        image = a.map_panel(ax, ids, np.where(common, trends[lag][0], np.nan), f'Lag {lag}', .012)
    fig.colorbar(image, ax=list(axes.ravel()), orientation='horizontal', fraction=.04,
                 label='TAC trend (year^-1)')
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'Fig_lag_spatial_common_support.{ext}', dpi=350, bbox_inches='tight')
    plt.close(fig)
    comparisons, regional, diagnostics, trajectories, fractions = [], [], [], [], []
    for vegetation, vm in [('Global', np.ones(len(ids), bool)), ('Forest', forest), ('Non-forest', ~forest)]:
        for code, name in a.ZONE.items():
            take = common & vm & ((zone == code) if code else True)
            for lag, values in series.items():
                means = np.array([a.weighted_mean(v[take], weights[take]) for v in values])
                for year, mean in zip(years, means):
                    trajectories.append(dict(vegetation=vegetation, zone=name, TAC_lag=lag,
                                             reference_year=int(year), endpoint_year=int(year+1),
                                             mean_TAC=mean, n_pixels=int(take.sum())))
                for n, period in [(22, '22_periods'), (19, '19_periods')]:
                    rows, diag = a.fit_trajectory(years[:n], means[:n], 'kNDVI', vegetation, name, 5, period)
                    regional.extend([dict(**r, TAC_lag=lag) for r in rows])
                    diagnostics.append(dict(**diag, TAC_lag=lag))
                slope, p = trends[lag]
                fractions.append(dict(vegetation=vegetation, zone=name, TAC_lag=lag,
                                      n_pixels=int(take.sum()),
                                      positive_trend_area_pct=100*float(weights[take & (slope > 0)].sum()/weights[take].sum()),
                                      negative_trend_area_pct=100*float(weights[take & (slope < 0)].sum()/weights[take].sum())))
                if lag != 1:
                    comparisons.append(dict(vegetation=vegetation, zone=name, TAC_lag=lag,
                                            **a.spatial_comparison(*trends[1], slope, p, weights, take)))
    for name, records in [('regional_OLS_HAC', regional), ('regional_trajectories', trajectories),
                          ('residual_diagnostics', diagnostics), ('spatial_comparison', comparisons),
                          ('direction_area_fractions', fractions)]:
        pd.DataFrame(records).to_csv(OUT / f'{name}.csv', index=False)
    data = pd.DataFrame(trajectories)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    styles = {1: '-', 2: '--', 3: '-.', 5: ':'}
    for row, veg in enumerate(['Forest', 'Non-forest']):
        for lag in [1, 2, 3, 5]:
            d = data.loc[data.vegetation.eq(veg) & data.zone.eq('Global') & data.TAC_lag.eq(lag)].sort_values('reference_year')
            axes[row, 0].plot(d.reference_year, d.mean_TAC - d.mean_TAC.iloc[0],
                              styles[lag], color=a.COLORS[veg], lw=1.4, label=f'Lag {lag}')
            r = next(r for r in regional if r['vegetation'] == veg and r['zone'] == 'Global'
                     and r['TAC_lag'] == lag and r['period'] == '22_periods' and r['method'] == 'HAC4')
            axes[row, 1].errorbar(lag, r['slope']*1000,
                                yerr=[[1000*(r['slope']-r['ci95_low'])], [1000*(r['ci95_high']-r['slope'])]],
                                fmt='o', color=a.COLORS[veg], capsize=3)
        axes[row, 0].set(title=veg, ylabel='TAC change from 2004', xlabel='Reference year')
        axes[row, 0].legend(frameon=False, ncol=4, fontsize=8)
        axes[row, 1].axhline(0, color='#777777', lw=.7)
        axes[row, 1].set(title=f'{veg}: OLS slope with HAC4 interval', xticks=[1, 2, 3, 5],
                         xlabel='TAC lag (collection positions)', ylabel='Trend (10^-3 year^-1)')
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'Fig_lag_regional_sensitivity.{ext}', dpi=350, bbox_inches='tight')
    plt.close(fig)
    checks = dict(status='completed', n_common=int(common.sum()),
                  forest_common=int((common & forest).sum()), nonforest_common=int((common & ~forest).sum()),
                  all_lags_use_identical_pixels=True, reference_years=[2004, 2025],
                  endpoint_years=[2005, 2026], QA=[0, 1],
                  no_TAC_recomputation=True, no_RF_retraining=True,
                  pixel_significance='OLS; conditional significant-only agreement is secondary',
                  regional_significance='HAC 1-4; HAC4 primary')
    (OUT / 'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2), flush=True)
    print(pd.DataFrame(comparisons).loc[lambda d: d.zone.eq('Global')].to_string(index=False), flush=True)
    print(pd.DataFrame(regional).loc[lambda d: d.zone.eq('Global') & d.period.eq('22_periods') & d.method.eq('HAC4'),
                                       ['vegetation', 'TAC_lag', 'slope', 'p_value']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
