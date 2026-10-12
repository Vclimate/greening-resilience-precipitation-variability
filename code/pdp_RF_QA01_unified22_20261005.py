"""Fixed 5000-pixel background PDPs over all 30 verified refits, including zones."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import recompute_QA01_22periods_20261005 as data
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

np, pd, rf = data.np, data.pd, data.rf
OUT = data.OUT/'PDP'
FIG = data.OUT/'figures'
GREEN, PINK = plt.get_cmap('PiYG_r')(1/9), plt.get_cmap('PiYG_r')(8/9)
LABEL = {'P_SI': 'P.SI', 'P_NIPV': 'P.NIPV', 'P_ET': 'P - ET', 'dTa': r'$\delta T_a$',
         'dP': r'$\delta P$', 'dkNDVI': r'$\delta$kNDVI', 'Forest_fraction': 'Forest fraction'}
XLABEL = {'P_SI': 'P.SI (-)', 'P_NIPV': 'P.NIPV (-)', 'P_ET': 'P - ET (mm yr$^{-1}$)',
          'dTa': r'$\delta T_a$ ($^\circ$C yr$^{-1}$)', 'dP': r'$\delta P$ (mm yr$^{-2}$)',
          'dkNDVI': r'$\delta$kNDVI (yr$^{-1}$)', 'Rs': r'Rs (MJ m$^{-2}$ day$^{-1}$)',
          'SM': 'SM (m$^3$ m$^{-3}$)', 'VPD': 'VPD (hPa)', 'WTD': 'WTD (m)',
          'Elevation': 'Elevation (m)', 'Slope': 'Slope (degrees)', 'HFP': 'HFP (-)',
          'Clay': 'Clay (g kg$^{-1}$)', 'Sand': 'Sand (g kg$^{-1}$)', 'SOC': 'SOC (dg kg$^{-1}$)',
          'Nitrogen': 'Nitrogen (cg kg$^{-1}$)', 'Species': 'Species (number)',
          'kNDVI': 'kNDVI (-)', 'Forest_fraction': 'Forest fraction (-)',
          'P': 'P (mm yr$^{-1}$)', 'Ta': r'Ta ($^\circ$C)'}
ZONE = {0: 'Global', 1: 'Tropical', 2: 'Arid', 3: 'Temperate', 4: 'Boreal'}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7,
                     'axes.titlesize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})


def prepare():
    manifest = rf.read(data.DATA/'source_manifest.json')
    df = pd.read_parquet(manifest['table'])
    for veg in ['Forest', 'Non-forest']:
        folder = OUT/veg
        folder.mkdir(parents=True, exist_ok=True)
        group = df.loc[df.vegetation.eq(veg)].reset_index(drop=True)
        with np.load(data.MODELS/veg/'runs/00/split_indices.npz') as split:
            rng = np.random.default_rng(20261005+(veg == 'Non-forest'))
            idx = np.sort(rng.choice(split['test'], size=min(5000, len(split['test'])), replace=False))
        background = group.iloc[idx].copy()
        background['group_row_index'] = idx
        background.to_parquet(folder/'background.parquet', index=False)
        ranking = pd.read_csv(data.MODELS/veg/'summary/importance_summary.csv').dropna(subset=['standardized_importance'])
        top = ranking.variable.head(6).tolist()
        grids = {}
        for var in top:
            x = background[var].to_numpy()
            grids[var] = np.linspace(*np.quantile(x, [.05, .95]), 41)
            if np.unique(x).size < 41:
                grids[var] = np.unique(x)
        data.prep.dump(folder/'protocol.json', dict(background_n=len(idx), top=top,
            source_table_sha256=manifest['table_sha256'], background_sha256=rf.digest(folder/'background.parquet'),
            grids=grids, method='brute-force probability PDP', repeats=30,
            uncertainty='pointwise 2.5/97.5 percentiles across retained-feature refits, not independent CI',
            zones='Stratified backgrounds in the same vegetation model, not separately fitted zone models',
            background_scope='seed0 test pixels; may enter training in later refits; explanation only'))


def worker(job):
    veg, rep = job
    folder, run = OUT/veg, data.MODELS/veg/'runs'/f'{rep:02d}'
    target = folder/f'run_{rep:02d}.parquet'
    if target.exists():
        return job
    protocol, design = rf.read(folder/'protocol.json'), rf.read(run/'design.json')
    names = design['selected_features']
    bg = pd.read_parquet(folder/'background.parquet')
    X = bg[names].to_numpy('float32')
    model = joblib.load(run/'model.joblib')
    model.n_jobs = 8
    np.testing.assert_array_equal(model.classes_, [0, 1])
    records = []
    for var in protocol['top']:
        if var not in names:
            continue
        j = names.index(var)
        grid = protocol['grids'][var]
        for q, (x, p) in enumerate(predict_grid(model, X, j, grid)):
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            if q == len(grid)//2:
                work = X.copy()
                work[:, j] = x
                manual = np.mean([t.predict_proba(work, check_input=False)[:, 1].mean() for t in model.estimators_])
                np.testing.assert_allclose(p.mean(), manual, rtol=0, atol=1e-12)
            for z, name in ZONE.items():
                mask = np.ones(len(bg), bool) if z == 0 else bg.climate_zone.eq(z).to_numpy()
                if mask.sum() >= 20:
                    records.append(dict(vegetation=veg, repeat=rep, variable=var, zone=name,
                                        x=float(x), probability_decline=float(p[mask].mean()), n_background=int(mask.sum())))
    pd.DataFrame(records).to_parquet(target, index=False)
    return job


def predict_grid(model, X, feature, grid, batch=8):
    # Exact brute-force PDP; batching avoids rebuilding the thread pool per x.
    for start in range(0, len(grid), batch):
        values = np.asarray(grid[start:start+batch], dtype='float32')
        work = np.tile(X, (len(values), 1))
        work.reshape(len(values), len(X), X.shape[1])[:, :, feature] = values[:, None]
        probabilities = model.predict_proba(work)[:, 1].reshape(len(values), len(X))
        for x, p in zip(grid[start:start+batch], probabilities):
            yield x, p


def summaries():
    df = pd.read_parquet(rf.read(data.DATA/'source_manifest.json')['table'])
    performance = []
    for veg in ['Forest', 'Non-forest']:
        folder = OUT/veg
        all_curves = pd.concat([pd.read_parquet(folder/f'run_{r:02d}.parquet') for r in range(30)])
        all_curves.to_csv(folder/'PDP_all30.csv', index=False)
        rows = []
        for (var, zone, x), group in all_curves.groupby(['variable', 'zone', 'x']):
            low, median, high = np.quantile(group.probability_decline, [.025, .5, .975])
            rows.append(dict(variable=var, zone=zone, x=x, median=median, q025=low, q975=high,
                             n_models=group.repeat.nunique(), n_background=group.n_background.iloc[0]))
        summary = pd.DataFrame(rows)
        summary.to_csv(folder/'PDP_summary.csv', index=False)
        contrasts = []
        for (var, zone), group in summary.groupby(['variable', 'zone']):
            c = group.sort_values('x')
            curves = all_curves.loc[all_curves.variable.eq(var) & all_curves.zone.eq(zone)]
            pivot = curves.pivot(index='repeat', columns='x', values='probability_decline')
            contrasts.append(dict(variable=var, zone=zone, x_low=float(c.x.iloc[0]), x_high=float(c.x.iloc[-1]),
                probability_low=float(c['median'].iloc[0]), probability_high=float(c['median'].iloc[-1]),
                low_high_change_pp=100*float(c['median'].iloc[-1]-c['median'].iloc[0]),
                curve_range_pp=100*float(np.ptp(c['median'])),
                x_at_min=float(c.x.iloc[int(np.argmin(c['median']))]),
                x_at_max=float(c.x.iloc[int(np.argmax(c['median']))]),
                n_positive_endpoint=int((pivot.iloc[:, -1] > pivot.iloc[:, 0]).sum()),
                n_negative_endpoint=int((pivot.iloc[:, -1] < pivot.iloc[:, 0]).sum()),
                n_models=len(pivot)))
        pd.DataFrame(contrasts).to_csv(folder/'PDP_contrasts.csv', index=False)
        group = df.loc[df.vegetation.eq(veg)].set_index('pixel_id')
        for rep in range(30):
            prediction = pd.read_parquet(data.MODELS/veg/'runs'/f'{rep:02d}'/'predictions.parquet')
            prediction['climate_zone'] = group.loc[prediction.pixel_id, 'climate_zone'].to_numpy()
            for z, name in ZONE.items():
                p = prediction if z == 0 else prediction.loc[prediction.climate_zone.eq(z)]
                if len(p) < 2 or p.y.nunique() != 2:
                    continue
                y, probability = p.y.to_numpy(), p.probability_decline.to_numpy()
                performance.append(dict(vegetation=veg, repeat=rep, zone=name, n_test=len(p),
                    accuracy=accuracy_score(y, probability >= .5), balanced_accuracy=balanced_accuracy_score(y, probability >= .5),
                    macro_f1=f1_score(y, probability >= .5, average='macro'), roc_auc=roc_auc_score(y, probability)))
    pd.DataFrame(performance).to_csv(OUT/'zone_holdout_metrics_all30.csv', index=False)


def importance(ax, veg):
    rank = pd.read_csv(data.MODELS/veg/'summary/importance_summary.csv').dropna(subset=['standardized_importance']).head(10)
    med = rank.standardized_importance.to_numpy()
    err = np.array([med-rank.standardized_importance_q025, rank.standardized_importance_q975-med])
    color = GREEN if veg == 'Forest' else PINK
    ax.barh(np.arange(len(rank)), med, color=color, height=.65)
    ax.errorbar(med, np.arange(len(rank)), xerr=err, fmt='none', ecolor='#555555', capsize=2, lw=.7)
    ax.set_yticks(np.arange(len(rank)), [LABEL.get(v, v) for v in rank.variable])
    ax.invert_yaxis()
    ax.set_xlim(0, max(rank.standardized_importance_q975)*1.08)
    ax.set_xlabel('Standardized OOB importance')
    perf = pd.read_csv(data.MODELS/veg/'summary/performance_summary.csv').set_index('metric').loc['accuracy']
    ax.set_title(f'{veg}\nAccuracy {perf["median"]:.2%} [{perf["q025"]:.2%}, {perf["q975"]:.2%}]', loc='left', pad=14)


def curve(ax, veg, var, zone='Global'):
    summary = pd.read_csv(OUT/veg/'PDP_summary.csv')
    c = summary.loc[summary.variable.eq(var) & summary.zone.eq(zone)].sort_values('x')
    if c.empty:
        ax.set_visible(False)
        return
    scale = 1e-6 if var == 'Rs' else 1
    x, median, low, high = [c[k].to_numpy() for k in ['x', 'median', 'q025', 'q975']]
    x *= scale
    for m, lo, hi, color, style in [(median, low, high, PINK, '-'), (1-median, 1-high, 1-low, GREEN, '--')]:
        ax.fill_between(x, lo, hi, color=color, alpha=.2, lw=0)
        ax.plot(x, m, color=color, ls=style, lw=1.1)
    bounds = np.r_[low, high, 1-low, 1-high]
    pad = max(.002, .1*np.ptp(bounds))
    ax.set_ylim(max(0, bounds.min()-pad), min(1, bounds.max()+pad))
    ax.set_xlim(x[0], x[-1])
    label = XLABEL.get(var, var)
    ax.set_xlabel(label.replace(' (', '\n('))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_title(LABEL.get(var, var), loc='left', pad=18)
    rank = pd.read_csv(data.MODELS/veg/'summary/importance_summary.csv').set_index('variable')
    value = rank.loc[var]
    ax.text(0, 1.035, f'I = {value.standardized_importance:.2f} [{value.standardized_importance_q025:.2f}, {value.standardized_importance_q975:.2f}]',
            transform=ax.transAxes, fontsize=6, va='bottom')
    ax.grid(axis='y', color='#EEEEEE', lw=.5)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(FIG/f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def figures():
    fig = plt.figure(figsize=(7.4, 8.5))
    outer = fig.add_gridspec(2, 2, width_ratios=[1, 2.9], hspace=.55, wspace=.33)
    for i, veg in enumerate(['Forest', 'Non-forest']):
        bar = fig.add_subplot(outer[i, 0])
        importance(bar, veg)
        bar.text(-.12, 1.17, ['a', 'c'][i], transform=bar.transAxes,
                 fontsize=10, fontweight='bold')
        inner = outer[i, 1].subgridspec(2, 3, hspace=.95, wspace=.48)
        top = rf.read(OUT/veg/'protocol.json')['top']
        for j, (axspec, var) in enumerate(zip(inner, top)):
            ax = fig.add_subplot(axspec)
            curve(ax, veg, var)
            ax.text(.98, 1.16, f'{["b", "d"][i]}{j+1}', transform=ax.transAxes,
                    ha='right', fontsize=8, fontweight='bold')
            if j%3 == 0:
                ax.set_ylabel('Classification\nprobability')
    fig.legend(handles=[Line2D([0], [0], color=GREEN, ls='--', label='Resilience increase'),
                        Line2D([0], [0], color=PINK, label='Resilience decline'),
                        Patch(facecolor='#999999', alpha=.2, label='2.5-97.5% across 30 refits')],
               loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=.1, top=.93)
    save(fig, 'Fig4_QA01_unified22_RF_PDP')
    for veg, name in [('Forest', 'FigS5'), ('Non-forest', 'FigS6')]:
        top = rf.read(OUT/veg/'protocol.json')['top']
        fig, axes = plt.subplots(4, 6, figsize=(7.4, 9))
        fig.subplots_adjust(hspace=1.0, wspace=.6, left=.08, right=.98, top=.89, bottom=.12)
        for i, zone in enumerate(list(ZONE.values())[1:]):
            for j, var in enumerate(top):
                curve(axes[i, j], veg, var, zone)
                axes[i, j].tick_params(labelsize=6)
                axes[i, j].xaxis.set_major_locator(MaxNLocator(2))
                axes[i, j].yaxis.set_major_locator(MaxNLocator(3))
                axes[i, j].xaxis.label.set_fontsize(6)
                axes[i, j]._left_title.set_fontsize(7)
                for annotation in axes[i, j].texts:
                    annotation.set_visible(False)
                if j == 0:
                    axes[i, j].set_ylabel(zone+'\nClassification probability')
        fig.suptitle(f'{veg}: climate-stratified PDPs', fontsize=9, y=.98)
        fig.legend(handles=[Line2D([0], [0], color=GREEN, ls='--', label='Resilience increase'),
                            Line2D([0], [0], color=PINK, label='Resilience decline')],
                   loc='lower center', ncol=2, frameon=False, fontsize=7,
                   bbox_to_anchor=(.5, .01))
        save(fig, f'{name}_QA01_unified22_climate_PDP')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, veg, color in zip(axes, ['Forest', 'Non-forest'], [GREEN, PINK]):
        ranking = pd.read_csv(data.MODELS/veg/'summary/importance_summary.csv').dropna(subset=['raw_delta_error_pp'])
        ranking = ranking.sort_values('raw_delta_error_pp', ascending=False).head(10)
        med = ranking.raw_delta_error_pp.to_numpy()
        ax.barh(np.arange(len(ranking)), med, color=color)
        ax.errorbar(med, np.arange(len(ranking)), xerr=[med-ranking.raw_delta_error_pp_q025,
                     ranking.raw_delta_error_pp_q975-med], fmt='none', capsize=2, ecolor='#555555')
        ax.set_yticks(np.arange(len(ranking)), [LABEL.get(n, n) for n in ranking.variable])
        ax.invert_yaxis()
        ax.set(title=veg, xlabel='Raw OOB error increase (percentage points)')
    save(fig, 'FigS7_QA01_unified22_raw_OOB')


def main():
    assert rf.read(data.OUT/'RF_verified.json')['status'] == 'verified'
    prepare()
    pending = [(veg, rep) for rep in range(30) for veg in ['Forest', 'Non-forest']
               if not (OUT/veg/f'run_{rep:02d}.parquet').exists()]
    completed = 60-len(pending)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, job) for job in pending]
        for f in as_completed(futures):
            job = f.result()
            completed += 1
            data.prep.dump(OUT/'status.json', dict(status='running', completed=completed, total=60, last=list(job)))
            data.prep.log(f'Unified22 PDP {completed}/60 completed')
    summaries()
    figures()
    data.prep.dump(OUT/'status.json', dict(status='completed', models=60, background_pixels_each=5000,
                                        grids=41, main_factors_each=6, all_runs_included=True))


if __name__ == '__main__':
    main()
