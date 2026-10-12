"""Exploratory monthly-update TAC/ENSO analysis; original QA01 inputs unchanged."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

import recompute_QA01_22periods_20261005 as source
import analyse_plot_QA01_unified22_20261005 as analysis
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from numba import set_num_threads
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

np, pd, rasterio = source.np, source.pd, source.rasterio
OUT = source.OUT / 'ENSO_monthly_20261006'
ONI_URL = 'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v5/'
SEASONS = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
ZONES = ['Global', 'Tropical', 'Arid', 'Temperate', 'Boreal']
COLORS = {'Forest': '#276419', 'Non-forest': '#8e0152'}


def dump(name, value):
    source.prep.dump(OUT / name, value)


def log(message):
    print(time.strftime('%H:%M:%S') + ' ' + message, flush=True)


def oni():
    path = OUT / 'NOAA_ONI_ERSSTv5_snapshot.html'
    if not path.exists():
        request = urllib.request.Request(ONI_URL, headers={'User-Agent': 'JRS-research-analysis/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    candidates = []
    for frame in pd.read_html(io.BytesIO(raw)):
        if frame.shape[1] != 13:
            continue
        year = pd.to_numeric(frame.iloc[:, 0], errors='coerce')
        selected = frame.loc[year.between(1950, 2100)].copy()
        if len(selected) >= 70:
            selected.columns = ['year'] + SEASONS
            selected['year'] = pd.to_numeric(selected.year).astype(int)
            candidates.append(selected)
    assert candidates, 'NOAA ONI table not found; do not substitute another index silently'
    frame = max(candidates, key=len).drop_duplicates('year').sort_values('year')
    records = []
    for row in frame.itertuples(index=False, name=None):
        for month, value in enumerate(row[1:], 1):
            value = pd.to_numeric(value, errors='coerce')
            if np.isfinite(value):
                records.append(dict(month=pd.Timestamp(int(row[0]), month, 1), ONI=float(value)))
    table = pd.DataFrame(records).sort_values('month').reset_index(drop=True)
    assert not table.month.duplicated().any()
    phase = np.zeros(len(table), np.int8)
    events = []
    for sign, label in [(1, 'El Nino'), (-1, 'La Nina')]:
        selected = sign * table.ONI.to_numpy() >= .5
        first = np.flatnonzero(np.diff(np.r_[False, selected].astype(int)) == 1)
        last = np.flatnonzero(np.diff(np.r_[selected, False].astype(int)) == -1)
        for start, end in zip(first, last):
            dates = table.month.iloc[start:end+1]
            assert dates.iloc[-1].to_period('M').ordinal - dates.iloc[0].to_period('M').ordinal == end-start
            if end-start+1 >= 5:
                phase[start:end+1] = sign
                events.append(dict(phase=label, onset=dates.iloc[0], end=dates.iloc[-1],
                                   n_months=end-start+1, peak_ONI=table.ONI.iloc[start:end+1].iloc[
                                       np.argmax(sign * table.ONI.iloc[start:end+1].to_numpy())]))
    table['phase_code'] = phase
    table['phase'] = np.choose(phase+1, ['La Nina', 'Neutral', 'El Nino'])
    table.to_csv(OUT / 'ONI_center_month_and_phase.csv', index=False)
    pd.DataFrame(events).sort_values('onset').to_csv(OUT / 'ONI_events.csv', index=False)
    dump('ONI_source.json', dict(url=ONI_URL, index='ONI based on ERSSTv5; archived historical definition',
        snapshot_sha256=hashlib.sha256(raw).hexdigest(), season_assignment='Assign each overlapping 3-month season to its central month; DJF -> January',
        phase_rule='ONI >= +0.5 or <= -0.5 for at least 5 consecutive overlapping seasons',
        realtime_warning='Centered ONI includes one month after its central month; this is retrospective, not an operational forecast'))
    return table


def reference_cohort():
    cached = OUT / 'pixel_metadata.parquet'
    if cached.exists():
        return pd.read_parquet(cached)
    table = pd.read_parquet(analysis.ANALYSIS / 'kNDVI_pixel_statistics.parquet')
    table = table.loc[table.MDN_2000_2025.lt(.5)].copy()
    ids = table.pixel_id.to_numpy(dtype=np.int64) - 200*7200
    assert np.all(np.diff(ids) > 0) and ids.min() >= 0 and ids.max() < 2800*7200
    valid = np.ones(len(ids), bool)
    for year in range(2005, 2027):
        a = analysis.read(source.DATA / 'kNDVI/lag1_5year' / f'kNDVI_TAC_lag1_5year_{year}_01_01_QA01.tif').ravel()[ids]
        valid &= np.isfinite(a) & (np.abs(a) <= 1)
    table = table.loc[valid].copy().reset_index(drop=True)
    ids = ids[valid]
    table['grid_id'] = ids
    table['latitude'] = 80-(ids//7200+.5)*.05
    table['longitude'] = -180+(ids%7200+.5)*.05
    table['area_weight'] = np.cos(np.deg2rad(table.latitude))
    reference = pd.read_csv(analysis.ANALYSIS / 'kNDVI_regional_trajectories.csv')
    for veg in ['Global', 'Forest', 'Non-forest']:
        n = len(table) if veg == 'Global' else int(table.vegetation.eq(veg).sum())
        rows = reference.loc[reference.zone.eq('Global') & reference.vegetation.eq(veg)]
        assert (rows.n_pixels == n).all(), (veg, n, rows.n_pixels.iloc[0])
    table.to_parquet(cached, index=False)
    log(f'Annual22 fixed reference cohort: {len(table):,} pixels')
    return table


def monthly(table, threads, strip_height):
    packs, dates, base_args = source.configuration()
    endpoints = pd.date_range('2005-01-01', '2026-01-01', freq='MS')
    months = endpoints-pd.offsets.MonthBegin(1)
    assert len(endpoints) == 253 and months[0] == pd.Timestamp('2004-12-01')
    starts = np.searchsorted(dates, (endpoints-pd.DateOffset(years=5)).to_numpy(dtype='datetime64[D]'))[None, :]
    ends = np.searchsorted(dates, endpoints.to_numpy(dtype='datetime64[D]'))
    assert np.array_equal(starts[0, ::12], base_args[4][0])
    assert np.array_equal(ends[::12], base_args[5])
    pd.DataFrame(dict(month=months, window_start=endpoints-pd.DateOffset(years=5),
                      window_end_exclusive=endpoints)).to_csv(OUT / 'monthly_window_dates.csv', index=False)
    path = OUT / 'monthly_TAC_pixel_stack.npy'
    state_path = OUT / 'compute_status.json'
    ids = table.grid_id.to_numpy()
    shape = (len(table), len(endpoints))
    resume = 0
    if path.exists():
        assert state_path.exists()
        state = json.loads(state_path.read_text(encoding='utf-8'))
        assert state['shape'] == list(shape) and state['strip_height'] == strip_height
        if state['status'] == 'completed':
            return np.load(path, mmap_mode='r'), months
        resume = state['completed_rows']
        assert resume % strip_height == 0 or resume == 2800
        cache = np.load(path, mmap_mode='r+')
    else:
        cache = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=shape)
        cache[:] = np.nan
        cache.flush()
        dump('compute_status.json', dict(status='running', completed_rows=0, total_rows=2800,
                                         shape=shape, strip_height=strip_height))
    stamps = {t['path']: (Path(t['path']).stat().st_size, Path(t['path']).stat().st_mtime_ns)
              for p in packs for t in p['tiles']}
    assert source.rf.read(source.OUT / 'preflight.json')['passed']
    set_num_threads(threads)
    annual_errors = []
    started = time.monotonic()
    with threadpool_limits(limits=1), rasterio.Env(GDAL_CACHEMAX=512*1024**2):
        for start in range(resume, 2800, strip_height):
            height = min(strip_height, 2800-start)
            a, b = np.searchsorted(ids, [start*7200, (start+height)*7200])
            if b > a:
                full = source.read_strip(packs, start, height, 'kNDVI', stamps)
                local = ids[a:b]-start*7200
                values = np.ascontiguousarray(full.reshape(len(dates), -1)[:, local].T)
                del full
                result = source.calculate(values, *base_args[:4], starts, ends,
                                          np.array([1]), np.array([0]))
                tac = result[:, 0, :, 0].copy()
                tac[np.abs(tac) > 1] = np.nan
                # Same calendar endpoints and same pixels must reproduce the published annual calculation.
                for j, year in enumerate(range(2005, 2027)):
                    old_path = source.DATA / 'kNDVI/lag1_5year' / f'kNDVI_TAC_lag1_5year_{year}_01_01_QA01.tif'
                    with rasterio.open(old_path) as src:
                        expected = src.read(1, window=source.Window(0, start, 7200, height)).ravel()[local]
                    np.testing.assert_allclose(tac[:, 12*j], expected, rtol=0, atol=2e-6)
                    annual_errors.append(float(np.max(np.abs(tac[:, 12*j]-expected))))
                cache[a:b] = tac
                del values, result, tac
                cache.flush()
            elapsed = time.monotonic()-started
            dump('compute_status.json', dict(status='running', completed_rows=start+height, total_rows=2800,
                shape=shape, strip_height=strip_height, elapsed_seconds=elapsed,
                annual_max_absolute_error=max(annual_errors, default=0)))
            log(f'Monthly TAC rows {start+height}/2800; pixels {b:,}/{len(table):,}; {elapsed:.0f}s')
    dump('compute_status.json', dict(status='completed', completed_rows=2800, total_rows=2800,
        shape=shape, strip_height=strip_height, elapsed_seconds=time.monotonic()-started,
        annual_max_absolute_error=max(annual_errors, default=0), annual_endpoints_conformance=True))
    return np.load(path, mmap_mode='r'), months


def aggregate(table, cache, months, index):
    fixed = np.ones(len(table), bool)
    for start in range(0, len(table), 200000):
        fixed[start:start+200000] = np.isfinite(cache[start:start+200000]).all(axis=1)
    ids = table.grid_id.to_numpy()
    forest = table.vegetation.eq('Forest').to_numpy()
    zone = table.climate_zone.to_numpy()
    weights = table.area_weight.to_numpy()
    records, cohort_records = [], []
    for name, cohort in [('monthly_fixed', fixed), ('annual22_reference', np.ones(len(table), bool))]:
        for z, zname in enumerate(ZONES):
            for veg, mask in [('Global', np.ones(len(table), bool)), ('Forest', forest), ('Non-forest', ~forest)]:
                selected = cohort & mask & ((zone == z) if z else True)
                positions = np.flatnonzero(selected)
                w = weights[selected]
                sums = np.zeros(len(months))
                denominators = np.zeros(len(months))
                counts = np.zeros(len(months), dtype=np.int64)
                for start in range(0, len(positions), 100000):
                    take = positions[start:start+100000]
                    a = np.asarray(cache[take], dtype=float)
                    valid = np.isfinite(a)
                    weight = w[start:start+len(take), None]
                    sums += np.nansum(a*weight, axis=0)
                    denominators += np.sum(valid*weight, axis=0)
                    counts += valid.sum(axis=0)
                mean = np.divide(sums, denominators, out=np.full(len(months), np.nan), where=denominators > 0)
                records.extend(dict(cohort=name, vegetation=veg, zone=zname, month=m,
                    mean_TAC=float(v), n_pixels=int(n), area_weight_sum=float(d))
                    for m, v, n, d in zip(months, mean, counts, denominators))
                cohort_records.append(dict(cohort=name, vegetation=veg, zone=zname, candidate_pixels=len(positions),
                    min_monthly_pixels=int(counts.min()), max_monthly_pixels=int(counts.max())))
    regional = pd.DataFrame(records).merge(index, on='month', how='left', validate='many_to_one')
    assert regional.ONI.notna().all(), 'NOAA index does not cover the analysis months'
    regional.to_csv(OUT / 'monthly_regional_TAC_ONI.csv', index=False)
    pd.DataFrame(cohort_records).to_csv(OUT / 'cohort_coverage.csv', index=False)
    main = regional.loc[regional.cohort.eq('monthly_fixed')].copy()
    assert main.groupby(['vegetation', 'zone']).n_pixels.nunique().eq(1).all()
    # The sensitivity cohort reproduces the annual means in FIG1/FIG2, not just individual pixels.
    before = pd.read_csv(analysis.ANALYSIS / 'kNDVI_regional_trajectories.csv')
    available = regional.loc[regional.cohort.eq('annual22_reference') & regional.month.dt.month.eq(12)].copy()
    available['reference_year'] = available.month.dt.year
    joined = available.merge(before, on=['vegetation', 'zone', 'reference_year'], validate='one_to_one')
    np.testing.assert_allclose(joined.mean_TAC_x, joined.mean_TAC_y, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(joined.n_pixels_x, joined.n_pixels_y)
    pd.DataFrame(dict(pixel_id=table.pixel_id, grid_id=ids, monthly_all_valid=fixed)).to_parquet(
        OUT / 'monthly_fixed_membership.parquet', index=False)
    dump('cohort_audit.json', dict(annual22_reference_pixels=len(table), monthly_fixed_pixels=int(fixed.sum()),
        removed_pixels=int((~fixed).sum()), removed_percent=100*float((~fixed).mean()),
        annual_regional_means_match=True, monthly_fixed_membership_constant=True,
        membership_warning='All-253-month cohort may be smaller than annual22 cohort; both summaries are saved'))
    return main


def regress(main):
    results, diagnoses, detrended = [], [], []
    for (veg, zone), group in main.groupby(['vegetation', 'zone'], sort=False):
        group = group.sort_values('month').copy()
        trend = np.arange(len(group), dtype=float)/12
        trend -= trend.mean()
        seasonal = pd.get_dummies(group.month.dt.month, drop_first=True, dtype=float).to_numpy()
        base = np.column_stack([np.ones(len(group)), trend, seasonal])
        y = group.mean_TAC.to_numpy()
        assert np.isfinite(y).all()
        background = sm.OLS(y, base).fit()
        group['TAC_adjusted'] = background.resid
        detrended.append(group)
        for outcome in ['level', 'monthly_change']:
            response = y if outcome == 'level' else np.r_[np.nan, np.diff(y)]
            for lag in [0, 3, 6, 12]:
                x = group.ONI.shift(lag).to_numpy()
                valid = np.isfinite(x) & np.isfinite(response)
                X = np.column_stack([base, x])[valid]
                model = sm.OLS(response[valid], X).fit()
                controls = base[valid]
                projection = np.linalg.pinv(controls)
                adjusted_y = response[valid]-controls@(projection@response[valid])
                adjusted_x = x[valid]-controls@(projection@x[valid])
                partial_r = np.dot(adjusted_y, adjusted_x)/(np.linalg.norm(adjusted_y)*np.linalg.norm(adjusted_x))
                shifted = np.column_stack([np.roll(x[valid], k) for k in range(1, int(valid.sum()))])
                shifted -= controls@(projection@shifted)
                shift_r = adjusted_y@shifted/(np.linalg.norm(adjusted_y)*np.linalg.norm(shifted, axis=0))
                p_shift = (1+np.count_nonzero(np.abs(shift_r) >= abs(partial_r)-1e-12))/int(valid.sum())
                for bandwidth in [12, 24, 60]:
                    fit = model.get_robustcov_results(cov_type='HAC', maxlags=bandwidth,
                        kernel='bartlett', use_correction=True, use_t=True)
                    low, high = fit.conf_int()[-1]
                    results.append(dict(vegetation=veg, zone=zone, outcome=outcome,
                        ONI_lead_months=lag, HAC_lag_months=bandwidth, n_months=int(valid.sum()),
                        coefficient_per_ONI_degC=float(fit.params[-1]), ci95_low=float(low),
                        ci95_high=float(high), p_HAC=float(fit.pvalues[-1]),
                        partial_r_adjusted=float(partial_r), p_circular_shift=float(p_shift),
                        Pearson_r_raw=float(pearsonr(response[valid], x[valid]).statistic),
                        adjusted_R_squared=float(model.rsquared_adj)))
                if lag == 0:
                    lb = acorr_ljungbox(model.resid, lags=[12, 24, 60], return_df=True)
                    diagnoses.append(dict(vegetation=veg, zone=zone, outcome=outcome,
                        residual_ACF1=float(np.corrcoef(model.resid[:-1], model.resid[1:])[0, 1]),
                        **{f'Ljung_Box_{l}_p': float(lb.loc[l, 'lb_pvalue']) for l in [12, 24, 60]}))
    results = pd.DataFrame(results)
    results['q_BH_exploratory_40_tests'] = np.nan
    results['q_BH_contemporaneous_10_tests'] = np.nan
    results['q_BH_shift_exploratory_40_tests'] = np.nan
    results['q_BH_shift_contemporaneous_10_tests'] = np.nan
    # Global-vegetation summaries overlap forest/non-forest and are descriptive only.
    for outcome in ['level', 'monthly_change']:
        for bandwidth in [12, 24, 60]:
            selected = results.outcome.eq(outcome) & results.HAC_lag_months.eq(bandwidth) & results.vegetation.ne('Global')
            results.loc[selected, 'q_BH_exploratory_40_tests'] = multipletests(results.loc[selected, 'p_HAC'], method='fdr_bh')[1]
            results.loc[selected, 'q_BH_shift_exploratory_40_tests'] = multipletests(results.loc[selected, 'p_circular_shift'], method='fdr_bh')[1]
            zero = selected & results.ONI_lead_months.eq(0)
            results.loc[zero, 'q_BH_contemporaneous_10_tests'] = multipletests(results.loc[zero, 'p_HAC'], method='fdr_bh')[1]
            results.loc[zero, 'q_BH_shift_contemporaneous_10_tests'] = multipletests(results.loc[zero, 'p_circular_shift'], method='fdr_bh')[1]
    results.to_csv(OUT / 'ONI_regressions_HAC_and_FDR.csv', index=False)
    pd.DataFrame(diagnoses).to_csv(OUT / 'temporal_dependence_diagnostics.csv', index=False)
    adjusted = pd.concat(detrended, ignore_index=True)
    adjusted.to_csv(OUT / 'monthly_TAC_adjusted_for_trend_and_season.csv', index=False)
    return results, adjusted


def phases(ax, index):
    for row in index.itertuples():
        if row.phase_code:
            ax.axvspan(row.month, row.month+pd.offsets.MonthBegin(1),
                color='#d6604d' if row.phase_code > 0 else '#4393c3', alpha=.12, lw=0, zorder=0)


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT/f'{name}.{ext}', dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plots(main, results, adjusted, index):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    subset = index.loc[index.month.between(main.month.min(), main.month.max())]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1.6, 1.6]})
    axes[0].plot(subset.month, subset.ONI, color='#444444', lw=1)
    axes[0].axhline(.5, color='#b2182b', ls=':', lw=.7)
    axes[0].axhline(-.5, color='#2166ac', ls=':', lw=.7)
    axes[0].set(ylabel='ONI (degrees C)', title='a  ENSO')
    for ax, veg, letter in zip(axes[1:], ['Forest', 'Non-forest'], ['b', 'c']):
        for zone, style in [('Global', '-'), ('Tropical', '--')]:
            a = main.loc[main.vegetation.eq(veg) & main.zone.eq(zone)].sort_values('month')
            ax.plot(a.month, a.mean_TAC, style, color=COLORS[veg], lw=1.3, alpha=1 if zone == 'Global' else .7, label=zone)
        ax.set(ylabel='Lag-1 TAC', title=f'{letter}  {veg}')
        ax.legend(loc='upper left', ncol=2, frameon=False)
    for ax in axes:
        phases(ax, subset)
        ax.set_xlim(main.month.min(), main.month.max()+pd.offsets.MonthBegin(1))
    axes[0].legend(handles=[Patch(color='#d6604d', alpha=.3, label='El Nino'),
        Patch(color='#4393c3', alpha=.3, label='La Nina')], loc='upper left', ncol=2, frameon=False)
    axes[-1].set_xlabel('Month represented by the window end (end date minus one month)')
    fig.tight_layout()
    save(fig, 'Fig_ENSO_monthly_TAC_global_tropical')
    fig, axes = plt.subplots(5, 2, figsize=(10, 9), sharex=True)
    for j, veg in enumerate(['Forest', 'Non-forest']):
        for i, zone in enumerate(ZONES):
            ax = axes[i, j]
            a = adjusted.loc[adjusted.vegetation.eq(veg) & adjusted.zone.eq(zone)].sort_values('month')
            phases(ax, subset)
            ax.plot(a.month, a.TAC_adjusted*1000, color=COLORS[veg], lw=1)
            ax.axhline(0, color='#999999', lw=.5, ls=':')
            ax.set_ylabel(f'{zone}\nTAC anomaly ($10^{{-3}}$)')
            if not i:
                ax.set_title(veg, fontweight='bold')
            if i == 4:
                ax.set_xlabel('Month')
    fig.tight_layout()
    save(fig, 'Fig_ENSO_monthly_TAC_climate_anomalies')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    source_rows = results.loc[results.outcome.eq('level') & results.ONI_lead_months.eq(0) & results.HAC_lag_months.eq(60)]
    for ax, veg in zip(axes, ['Forest', 'Non-forest']):
        data = source_rows.loc[source_rows.vegetation.eq(veg)].set_index('zone').loc[ZONES]
        y = np.arange(5)
        x = data.coefficient_per_ONI_degC.to_numpy()*1000
        low, high = data.ci95_low.to_numpy()*1000, data.ci95_high.to_numpy()*1000
        ax.errorbar(x, y, xerr=np.vstack([x-low, high-x]), fmt='o', color=COLORS[veg], capsize=3, ms=4)
        ax.axvline(0, color='#888888', ls=':', lw=.8)
        ax.set(yticks=y, yticklabels=ZONES, title=veg, xlabel='ONI coefficient ($10^{-3}$ TAC / degree C)')
        for k, q in enumerate(data.q_BH_contemporaneous_10_tests):
            ax.text(.98, y[k]-.12, f'q={q:.3f}', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=6)
    axes[0].invert_yaxis()
    fig.suptitle('Trend/season-adjusted association; 95% HAC(60-month) intervals', fontsize=9)
    fig.tight_layout()
    save(fig, 'Fig_ENSO_ONI_association_HAC60')


def event_composites(adjusted):
    events = pd.read_csv(OUT / 'ONI_events.csv', parse_dates=['onset', 'end'])
    offsets = np.arange(-12, 25)
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    for i, phase in enumerate(['El Nino', 'La Nina']):
        for j, veg in enumerate(['Forest', 'Non-forest']):
            ax = axes[i, j]
            for zone, style in [('Global', '-'), ('Tropical', '--')]:
                group = adjusted.loc[adjusted.vegetation.eq(veg) & adjusted.zone.eq(zone)].set_index('month')
                curves = []
                for event in events.loc[events.phase.eq(phase)].itertuples():
                    dates = pd.DatetimeIndex([event.onset+pd.DateOffset(months=int(k)) for k in offsets])
                    if not dates.isin(group.index).all():
                        continue
                    curve = group.loc[dates, 'TAC_adjusted'].to_numpy()
                    curve -= curve[offsets < 0].mean()
                    curves.append(curve)
                    records.extend(dict(phase=phase, vegetation=veg, zone=zone, onset=event.onset,
                        relative_month=int(k), TAC_anomaly_from_pre_event=float(v)) for k, v in zip(offsets, curve))
                if curves:
                    curves = np.asarray(curves)*1000
                    color = COLORS[veg] if zone == 'Global' else '#777777'
                    ax.plot(offsets, curves.mean(axis=0), style, color=color, lw=1.2, label=f'{zone} (n={len(curves)} events)')
                    if zone == 'Global':
                        ax.fill_between(offsets, *np.percentile(curves, [25, 75], axis=0), color=color, alpha=.15)
            ax.axvline(0, color='#777777', ls=':', lw=.8)
            ax.axhline(0, color='#aaaaaa', lw=.5)
            ax.set(title=f'{phase}: {veg}', ylabel='TAC anomaly ($10^{-3}$)')
            ax.legend(frameon=False, fontsize=7)
            if i:
                ax.set_xlabel('Months relative to ENSO onset')
    fig.suptitle('Retrospective event composites; shading = between-event 25th-75th percentiles', fontsize=9)
    fig.tight_layout()
    save(fig, 'Fig_ENSO_event_composites')
    pd.DataFrame(records).to_csv(OUT / 'ENSO_event_composite_source.csv', index=False)


def report(results):
    cohort = json.loads((OUT / 'cohort_audit.json').read_text(encoding='utf-8'))
    selected = results.loc[results.vegetation.ne('Global') & results.outcome.eq('level')
        & results.ONI_lead_months.eq(0) & results.HAC_lag_months.eq(60)]
    lines = ['# ENSO 与逐月更新 TAC：探索性分析', '',
        '## 主要判断',
        f'- 10 个同期关联中，HAC60 与 BH 校正后 q<0.05 的结果为 {int(selected.q_BH_contemporaneous_10_tests.lt(.05).sum())} 项；循环平移后 q<0.05 的结果为 {int(selected.q_BH_shift_contemporaneous_10_tests.lt(.05).sum())} 项。',
        '- 热带出现森林正关联、非森林负关联的候选信号，但显著性受 HAC 带宽影响，且循环平移未提供稳健支持。',
        '- 若干滞后和月际变化在 HAC 检验中显著，不能只据此选取最有利结果；应结合 40 项校正、时间平移与事件差异解释。',
        '- 当前结果不支持写成 ENSO 统一或因果性地降低全球植被韧性，也不等于证明不存在局地 ENSO 响应。', '',
        '## 口径',
        '- 使用原始 QA=0/1 MOD13C1 NDVI、kNDVI=tanh(NDVI²)，复用 2000–2025 年移动去趋势和三阶谐波去季节流程。',
        '- TAC 为像元残差的一阶稳健自回归斜率；保留原 16 天合成序列 lag-1 定义，不是月平均序列的 lag-1。',
        '- 五年窗口每月滑动。窗口末端不含当日；2025-01-01 对应 2024 年 12 月。253 个窗口覆盖 2004-12 至 2025-12。',
        '- 日期范围和定义不变的每年 1 月 1 日窗口逐像元核对原年度 TAC；另外核对年度区域均值。',
        f'- 年度固定参考样本 {cohort["annual22_reference_pixels"]:,}；253 个月均有效的固定样本 {cohort["monthly_fixed_pixels"]:,}，减少 {cohort["removed_percent"]:.3f}%。',
        '- 森林/非森林与五个区域分别汇总，按纬度面积权重计算。没有按 TAC 趋势显著性筛选。',
        '- ENSO 使用 NOAA 归档的 ERSSTv5 ONI，三个月季节分配到中心月份；±0.5°C 持续至少五个重叠季节划分事件。',
        '', '## 同期关联结果',
        'TAC ~ ONI + 线性时间趋势 + 月份固定效应。下表使用 HAC 60 个月，BH q 值在森林/非森林×五区域的 10 个同期检验中校正。',
        '', '| 植被 | 区域 | ONI 系数（TAC/°C） | 95% HAC CI | P | BH q |',
        '|---|---|---:|---|---:|---:|']
    for row in selected.itertuples():
        lines.append(f'| {row.vegetation} | {row.zone} | {row.coefficient_per_ONI_degC:.5f} | [{row.ci95_low:.5f}, {row.ci95_high:.5f}] | {row.p_HAC:.4g} | {row.q_BH_contemporaneous_10_tests:.4g} |')
    lines += ['', '## 时间依赖敏感性',
        '循环平移检验对 ONI 做所有非零循环平移，每次重新控制时间趋势和月份，比较偏相关绝对值。它保留指数的循环时间结构，但依赖平稳性与周期边界假设，是辅助检验而不是因果检验。', '',
        '| 植被 | 区域 | 同期循环平移 P | 10项 BH q |', '|---|---|---:|---:|']
    for row in selected.itertuples():
        lines.append(f'| {row.vegetation} | {row.zone} | {row.p_circular_shift:.4g} | {row.q_BH_shift_contemporaneous_10_tests:.4g} |')
    lines += ['', 'HAC 带宽敏感性（未经多重校正的 P）：', '',
        '| 植被 | 区域 | HAC12 | HAC24 | HAC60 |', '|---|---|---:|---:|---:|']
    for veg in ['Forest', 'Non-forest']:
        for zone in ['Global', 'Tropical']:
            group = results.loc[results.vegetation.eq(veg) & results.zone.eq(zone)
                & results.outcome.eq('level') & results.ONI_lead_months.eq(0)].set_index('HAC_lag_months')
            lines.append(f'| {veg} | {zone} | {group.loc[12, "p_HAC"]:.4g} | {group.loc[24, "p_HAC"]:.4g} | {group.loc[60, "p_HAC"]:.4g} |')
    lines += ['', '## 12个月领先的探索性结果',
        '下列仅用于展示预先设置的一个时滞；完整 0/3/6/12 个月结果与 40 项多重校正见 CSV，不应只选择显著时滞作结论。', '',
        '| 植被 | 区域 | 系数 | HAC60 P | 40项 HAC BH q | 循环平移 P | 40项平移 BH q |',
        '|---|---|---:|---:|---:|---:|---:|']
    lagged = results.loc[results.vegetation.ne('Global') & results.zone.isin(['Global', 'Tropical'])
        & results.outcome.eq('level') & results.ONI_lead_months.eq(12) & results.HAC_lag_months.eq(60)]
    for row in lagged.itertuples():
        lines.append(f'| {row.vegetation} | {row.zone} | {row.coefficient_per_ONI_degC:.5f} | {row.p_HAC:.4g} | {row.q_BH_exploratory_40_tests:.4g} | {row.p_circular_shift:.4g} | {row.q_BH_shift_exploratory_40_tests:.4g} |')
    lines += ['', '## 解释边界',
        '- TAC 增加表示恢复速度下降的代理信号，并不是直接测量的恢复时间。',
        '- 五年窗口使 ENSO 事件前后的影响被混合和平滑；逐月更新不等于一个月尺度的独立韧性估计。',
        '- HAC 60 个月是与窗口长度相称的探索性设置，不保证完全消除时间依赖。另提供 12/24 个月带宽、月际 TAC 差分和 0/3/6/12 个月 ONI 领先结果，不按最小 P 值挑选。',
        '- 主模型的同期 q 值与探索性 40 个滞后组合的 q 值分开保存，不能混用。',
        '- 事件图阴影表示事件间 25–75% 分位范围，不是 95% CI。事件数有限，且五年 TAC 窗口跨事件重叠。',
        '- 预处理移动去趋势使用中心窗口、谐波使用全时段；ONI 也是中心三个月指数。本分析是回顾性描述，不能作为实时预警或 ENSO 的因果证明。',
        '- 未纳入同期降水、温度等混杂变量，因此不能据此断言 ENSO 独立驱动韧性变化。',
        '- 原始数据、主论文年度 TAC、RF 及文档均未修改。', '',
        f'NOAA 数据与定义：[ONI ERSSTv5]({ONI_URL})。']
    (OUT / 'ENSO_monthly_results_CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    dump('analysis_methods.json', dict(status='completed', exploratory=True,
        raw_source_audit=str(source.core.AUDIT), preprocessing_source=str(Path(source.__file__)),
        qa=[0, 1], preprocessing_period=[2000, 2025], detrend_half_window_days=912, harmonic_order=3,
        window_years=5, window_updates='monthly', lag_definition='lag1 in original MODIS collection positions',
        primary_HAC_bandwidth_months=60, sensitivity_HAC_bandwidths=[12, 24],
        predictors='ONI, centered linear time, eleven calendar-month dummy variables',
        primary_multiple_testing_family=10, exploratory_lag_multiple_testing_family=40,
        temporal_shift_test='All nonzero circular shifts of ONI, controls re-fitted after each shift; two-sided partial-correlation statistic; approximate stationary periodic null',
        ONI_leads_months=[0, 3, 6, 12], main_files_unchanged=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--strip-height', type=int, default=256)
    parser.add_argument('--postprocess-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = oni()
    if args.postprocess_only:
        assert json.loads((OUT / 'compute_status.json').read_text(encoding='utf-8'))['status'] == 'completed'
        regional = pd.read_csv(OUT / 'monthly_regional_TAC_ONI.csv', parse_dates=['month'])
        main_table = regional.loc[regional.cohort.eq('monthly_fixed')].copy()
    else:
        table = reference_cohort()
        cache, months = monthly(table, args.threads, args.strip_height)
        main_table = aggregate(table, cache, months, index)
    results, adjusted = regress(main_table)
    plots(main_table, results, adjusted, index)
    event_composites(adjusted)
    report(results)
    print(results.loc[results.vegetation.ne('Global') & results.zone.isin(['Global', 'Tropical'])
        & results.outcome.eq('level') & results.ONI_lead_months.eq(0) & results.HAC_lag_months.eq(60)].to_string(index=False), flush=True)
    log('ENSO monthly-update analysis COMPLETED')


if __name__ == '__main__':
    main()
