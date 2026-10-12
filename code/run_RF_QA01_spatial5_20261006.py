"""Nested RF spatial-block five-fold sensitivity analysis on the current QA01 table."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import run_RF_local_QA0_repeated30_20261004 as helpers

ROOT = Path(r'D:\papaer4\20260604返修补实验')
SOURCE = ROOT / 'output/QA01_unified22_20261005'
OUT = SOURCE / 'RF_spatial5_500km_no_buffer_20261006'
SEED = 2026100600
CONFIG = dict(version='QA01_nested_spatial_RF_5fold_v1', outer_folds=5,
              block_size_projected_m=500000, block_crs='EPSG:6933', buffer_km=0,
              same_block_same_fold_across_vegetation=True, seed=SEED,
              concurrent_models=4, threads_per_model=8, trees_final=300,
              trees_tuning=150, TPE_trials_per_outer_fold=12, TPE_startup_trials=6,
              sampled_inner_training_cap=60000, sampled_inner_validation_cap=20000,
              inner_validation='One of five spatial group folds within outer training',
              tuning_search={'min_samples_leaf': [1, 64], 'max_features': [2, 'retained_count']},
              tuning_objective='Inner validation accuracy; AUC tie-break',
              final_threshold=.5, VIF_threshold=10, protected_variables=[],
              VIF_and_response_IQR='Fit on each inner/outer training set only',
              eligibility='Unchanged current QA0/1 P<0.05 complete-case first19 table',
              test_response_filter=False, block_scale_selected_using_performance=False,
              spatial_independence_guaranteed=False, bootstrap_blocks=1000,
              bootstrap_scope='Conditional OOF sample uncertainty, not retraining uncertainty',
              main_analysis_preserved=True, OOB_importance_or_SHAP_not_requested=True)


def dump(path, value):
    helpers.dump(path, value)


def log(text):
    print(time.strftime('%H:%M:%S') + ' ' + text, flush=True)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def make_blocks(df):
    project = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True)
    x, y = project.transform(df.longitude.to_numpy(), df.latitude.to_numpy())
    bx = np.floor(np.asarray(x) / 500000).astype('int32')
    by = np.floor(np.asarray(y) / 500000).astype('int32')
    return (bx.astype('int64') + 100000) * 200000 + (by + 100000)


def group_folds(labels, blocks, seed):
    folds = np.full(len(labels), -1, dtype='int8')
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(labels)), labels, blocks)):
        folds[test] = fold
    assert (folds >= 0).all()
    assert pd.DataFrame({'block': blocks, 'fold': folds}).groupby('block').fold.nunique().max() == 1
    return folds


def prepare():
    manifest_path = SOURCE / 'data/source_manifest.json'
    manifest = helpers.read(manifest_path)
    assert manifest['unified22'] and manifest['forbidden_data_read'] is False
    table = Path(manifest['table'])
    assert digest(table) == manifest['table_sha256']
    df = pd.read_parquet(table)
    names = manifest['candidates']
    assert len(names) == 22 and df.pixel_id.is_unique
    assert (df.tac_pvalue < .05).all() and (df.QA01_missing_fraction < .5).all()
    assert np.isfinite(df[names].to_numpy()).all()
    np.testing.assert_array_equal(df.y, (df.y_continuous >= 0).astype('int8'))
    row, col = np.divmod(df.pixel_id.to_numpy(), 7200)
    np.testing.assert_allclose(df.longitude, -180 + (col + .5) * .05, atol=1e-8, rtol=0)
    np.testing.assert_allclose(df.latitude, 90 - (row + .5) * .05, atol=1e-8, rtol=0)
    for item in manifest['counts']:
        assert df.vegetation.eq(item['vegetation']).sum() == item['n_eligible']
    df['block500'] = make_blocks(df)
    strata = 2 * df.vegetation.eq('Non-forest').to_numpy('int8') + df.y.to_numpy()
    df['fold'] = group_folds(strata, df.block500.to_numpy(), SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(configuration=CONFIG, source_manifest=str(manifest_path),
                    source_manifest_sha256=digest(manifest_path), table=str(table),
                    table_sha256=manifest['table_sha256'], script_sha256=digest(__file__),
                    helper_sha256=digest(helpers.__file__),
                    versions={k: importlib.metadata.version(k) for k in
                              ['numpy', 'pandas', 'scikit-learn', 'pyproj', 'optuna']})
    if (OUT / 'protocol.json').exists():
        assert helpers.read(OUT / 'protocol.json') == protocol, 'Changed protocol; do not mix runs'
    else:
        dump(OUT / 'protocol.json', protocol)
    assignment = df[['pixel_id', 'longitude', 'latitude', 'vegetation', 'y', 'block500', 'fold']]
    assignment.to_parquet(OUT / 'fold_assignments.parquet', index=False)
    summary = df.groupby(['vegetation', 'fold']).agg(
        n_test=('pixel_id', 'size'), test_blocks=('block500', 'nunique'),
        decline_fraction=('y', 'mean')).reset_index()
    summary.to_csv(OUT / 'fold_sample_counts.csv', index=False)
    log(f'Verified current table: {len(df):,} pixels, {df.block500.nunique()} spatial blocks')
    return df, names, protocol


def model(params, seed, features, trees):
    return RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=8,
        bootstrap=True, criterion='gini', max_depth=None, class_weight=None,
        min_samples_leaf=int(params['min_samples_leaf']),
        max_features=min(int(params['max_features']), features))


def nested_tune(df, X, y, names, outer, folder, seed):
    blocks = df.block500.to_numpy()
    folds = group_folds(y[outer], blocks[outer], seed + 11)
    inner0, validation = outer[folds != 0], outer[folds == 0]
    inner, iqr = helpers.legacy.iqr(df, inner0)
    assert not np.intersect1d(blocks[inner0], blocks[validation]).size
    selected, history = helpers.vif_select(X[inner], names)
    history.to_csv(folder / 'inner_VIF_history.csv', index=False)
    fit = helpers.legacy.sample(inner, y, 60000, seed + 12)
    val = helpers.legacy.sample(validation, y, 20000, seed + 13)
    np.savez_compressed(folder / 'inner_split.npz', training_before_iqr=inner0,
                        training=inner, validation=validation,
                        sampled_training=fit, sampled_validation=val)
    dump(folder / 'inner_design.json', dict(selected_features=[names[j] for j in selected],
         IQR=iqr, n_training=len(inner), n_validation=len(validation),
         n_sampled_training=len(fit), n_sampled_validation=len(val),
         spatial_blocks_disjoint=True, outer_test_used=False))
    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=6))
    study.enqueue_trial(dict(min_samples_leaf=5, max_features=min(4, len(selected))))
    fit_x = np.ascontiguousarray(X[fit][:, selected])
    val_x = np.ascontiguousarray(X[val][:, selected])

    def objective(trial):
        params = dict(min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 64, log=True),
                      max_features=trial.suggest_int('max_features', 2, len(selected)))
        duplicate = next((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                          and t.params == params), None)
        if duplicate is not None:
            trial.set_user_attr('auc', duplicate.user_attrs['auc'])
            trial.set_user_attr('duplicate_of', duplicate.number)
            return duplicate.value
        trained = model(params, seed, len(selected), 150).fit(fit_x, y[fit])
        probability = trained.predict_proba(val_x)[:, 1]
        trial.set_user_attr('auc', float(roc_auc_score(y[val], probability)))
        return float(accuracy_score(y[val], probability >= .5))

    def callback(study, trial):
        study.trials_dataframe().to_csv(folder / 'tuning_trials.csv', index=False)
        log(f'{df.vegetation.iloc[0]} {folder.name}: '
            f'TPE {trial.number + 1}/12 inner accuracy={trial.value:.2%}')

    study.optimize(objective, n_trials=12, callbacks=[callback])
    ranked = sorted(study.trials, key=lambda t: (-t.value, -t.user_attrs['auc'], t.number))
    chosen = ranked[0]
    dump(folder / 'tuning_selection.json', dict(params=chosen.params,
         inner_accuracy=chosen.value, inner_roc_auc=chosen.user_attrs['auc'],
         trial=chosen.number, outer_test_used=False, tuning_trees=150, final_trees=300))
    return chosen.params


def nearest_distance(df, train, test):
    lon, lat = np.deg2rad(df.longitude.to_numpy()), np.deg2rad(df.latitude.to_numpy())
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    distance, _ = cKDTree(xyz[train]).query(xyz[test], workers=8)
    km = 2 * 6371.0088 * np.arcsin(np.clip(distance / 2, 0, 1))
    return dict(min_km=float(km.min()), q05_km=float(np.quantile(km, .05)),
                median_km=float(np.median(km)), fraction_within_10km=float((km < 10).mean()),
                fraction_within_50km=float((km < 50).mean()), fraction_within_100km=float((km < 100).mean()))


def run_fold(group, names, fold):
    veg = group.vegetation.iloc[0]
    folder = OUT / veg / f'fold_{fold + 1}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        marker = helpers.read(folder / 'complete.json')
        assert digest(folder / 'predictions.parquet') == marker['predictions_sha256']
        assert digest(folder / 'model.joblib') == marker['model_sha256']
        return helpers.read(folder / 'metrics.json')
    started = time.monotonic()
    X, y = group[names].to_numpy('float32'), group.y.to_numpy('int8')
    outer = np.flatnonzero(group.fold.to_numpy() != fold)
    test = np.flatnonzero(group.fold.to_numpy() == fold)
    blocks = group.block500.to_numpy()
    assert not np.intersect1d(blocks[outer], blocks[test]).size
    assert np.array_equal(np.sort(np.r_[outer, test]), np.arange(len(group)))
    seed = SEED + (1000 if veg == 'Non-forest' else 0) + fold
    log(f'{veg} fold {fold + 1}/5: start nested spatial tuning')
    params = nested_tune(group, X, y, names, outer, folder, seed)
    train, iqr = helpers.legacy.iqr(group, outer)
    selected, history = helpers.vif_select(X[train], names)
    history.to_csv(folder / 'outer_VIF_history.csv', index=False)
    features = [names[j] for j in selected]
    np.savez_compressed(folder / 'outer_split.npz', training_before_iqr=outer, training=train, test=test)
    max_vif = float(history.loc[history.iteration == history.iteration.max(), 'vif'].max())
    assert max_vif < 10
    dump(folder / 'design.json', dict(seed=seed, fold=fold + 1, vegetation=veg,
         selected_features=features, removed_features=[n for n in names if n not in features],
         n_training_before_iqr=len(outer), n_training=len(train), n_test=len(test),
         training_IQR=iqr, test_filtered=False, max_VIF=max_vif,
         training_blocks=int(np.unique(blocks[train]).size), test_blocks=int(np.unique(blocks[test]).size),
         train_test_blocks_disjoint=True, hyperparameters=params))
    log(f'{veg} fold {fold + 1}/5: refit 300 trees, {len(train):,} pixels; params={params}')
    trained = model(params, seed, len(selected), 300).fit(X[train][:, selected], y[train])
    assert np.array_equal(trained.classes_, [0, 1])
    probability = trained.predict_proba(X[test][:, selected])[:, 1]
    baseline = np.full(len(test), int(y[train].mean() >= .5), dtype='int8')
    result = dict(vegetation=veg, fold=fold + 1, n_training=len(train),
                  **helpers.legacy.metrics(y[test], probability, baseline))
    distance = nearest_distance(group, train, test)
    dump(folder / 'nearest_training_distance.json', distance)
    prediction = group.iloc[test][['pixel_id', 'longitude', 'latitude', 'y', 'block500', 'fold']].copy()
    prediction['probability_decline'] = probability
    prediction['baseline_prediction'] = baseline
    prediction.to_parquet(folder / 'predictions.parquet', index=False)
    joblib.dump(trained, folder / 'model.joblib', compress=3)
    dump(folder / 'metrics.json', result)
    dump(folder / 'complete.json', dict(seed=seed, fold=fold + 1,
         elapsed_seconds=time.monotonic() - started,
         predictions_sha256=digest(folder / 'predictions.parquet'),
         model_sha256=digest(folder / 'model.joblib'), all_checks_passed=True))
    log(f'{veg} fold {fold + 1}/5 COMPLETE: accuracy={result["accuracy"]:.2%}, '
        f'balanced={result["balanced_accuracy"]:.2%}, AUC={result["roc_auc"]:.4f}')
    return result


def block_intervals(prediction, seed):
    y = prediction.y.to_numpy()
    p = (prediction.probability_decline >= .5).to_numpy('int8')
    work = pd.DataFrame(dict(block=prediction.block500, tn=(y == 0) & (p == 0),
                            fp=(y == 0) & (p == 1), fn=(y == 1) & (p == 0),
                            tp=(y == 1) & (p == 1)))
    counts = work.groupby('block')[['tn', 'fp', 'fn', 'tp']].sum().to_numpy('float64')
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=1000)
    tn, fp, fn, tp = (weights @ counts).T
    values = dict(accuracy=(tn + tp) / (tn + tp + fn + fp),
                  balanced_accuracy=.5 * (tn / (tn + fp) + tp / (tp + fn)),
                  macro_f1=.5 * (2 * tn / (2 * tn + fp + fn) + 2 * tp / (2 * tp + fp + fn)))
    return {name: dict(ci_low=float(np.quantile(v, .025)), ci_high=float(np.quantile(v, .975)))
            for name, v in values.items()}


def summarize(df, names, fold_results):
    results = pd.DataFrame(fold_results).sort_values(['vegetation', 'fold'])
    results.to_csv(OUT / 'performance_by_fold.csv', index=False)
    summary, retention = [], []
    for i, veg in enumerate(['Forest', 'Non-forest']):
        parts = [pd.read_parquet(OUT / veg / f'fold_{f + 1}/predictions.parquet') for f in range(5)]
        prediction = pd.concat(parts).sort_values('pixel_id').reset_index(drop=True)
        expected = df.loc[df.vegetation.eq(veg)].sort_values('pixel_id')
        assert prediction.pixel_id.is_unique and len(prediction) == len(expected)
        np.testing.assert_array_equal(prediction.pixel_id, expected.pixel_id)
        np.testing.assert_array_equal(prediction.y, expected.y)
        prediction.to_parquet(OUT / veg / 'OOF_predictions.parquet', index=False)
        pooled = helpers.legacy.metrics(prediction.y.to_numpy(), prediction.probability_decline.to_numpy(),
                                        prediction.baseline_prediction.to_numpy())
        ci = block_intervals(prediction, SEED + i)
        dump(OUT / veg / 'pooled_OOF_metrics.json', dict(**pooled, block_bootstrap=ci))
        reference = pd.read_csv(SOURCE / 'RF_endpoints_2005_2023_QA01_repeated30' / veg /
                                'summary/performance_summary.csv').set_index('metric')
        for metric in ['accuracy', 'macro_f1', 'balanced_accuracy', 'roc_auc', 'majority_baseline_accuracy']:
            values = results.loc[results.vegetation.eq(veg), metric]
            summary.append(dict(vegetation=veg, metric=metric, pooled_OOF=pooled[metric],
                 fold_mean=float(values.mean()), fold_SD=float(values.std(ddof=1)),
                 fold_min=float(values.min()), fold_max=float(values.max()),
                 n_OOF=len(prediction), **ci.get(metric, dict(ci_low=None, ci_high=None)),
                 existing_random30_median=float(reference.loc[metric, 'median']),
                 random_comparison_scope='Context only: 70/30 vs 80/20 and different tuning budgets'))
        for fold in range(5):
            design = helpers.read(OUT / veg / f'fold_{fold + 1}/design.json')
            retention.extend(dict(vegetation=veg, fold=fold + 1, variable=n,
                                  retained=n in design['selected_features']) for n in names)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'performance_summary.csv', index=False)
    pd.DataFrame(retention).to_csv(OUT / 'VIF_retention_by_fold.csv', index=False)
    percentage = summary.copy()
    for column in ['pooled_OOF', 'fold_mean', 'fold_SD', 'fold_min', 'fold_max',
                   'ci_low', 'ci_high', 'existing_random30_median']:
        percentage[column] *= 100
    percentage.to_csv(OUT / 'performance_summary_percent.csv', index=False)
    plot(df, results, summary)
    lines = ['# QA01 RF spatial five-fold validation', '',
             'Current QA0/1 first19 significant-trend cohort; 500 km projected equal-area blocks; no buffer.',
             'All VIF, response IQR filtering and parameter optimization used training folds only.',
             'The inner 12-trial TPE used up to 60,000/20,000 pixels and 150 trees; final fits used 300 trees.',
             'This is a bounded optimization trial, not an exhaustive search.', '',
             '| Vegetation | Pixels | OOF accuracy (95% block-bootstrap CI) | Balanced accuracy | ROC-AUC | Majority baseline |',
             '|---|---:|---:|---:|---:|---:|']
    for veg in ['Forest', 'Non-forest']:
        s = summary.loc[summary.vegetation.eq(veg)].set_index('metric')
        a = s.loc['accuracy']
        lines.append(f'| {veg} | {int(a.n_OOF):,} | {100*a.pooled_OOF:.2f}% '
            f'({100*a.ci_low:.2f}-{100*a.ci_high:.2f}%) | '
            f'{100*s.loc["balanced_accuracy", "pooled_OOF"]:.2f}% | '
            f'{s.loc["roc_auc", "pooled_OOF"]:.4f} | '
            f'{100*s.loc["majority_baseline_accuracy", "pooled_OOF"]:.2f}% |')
    lines += ['', 'Block-bootstrap intervals describe OOF sample variation conditional on the fitted models, not complete training uncertainty.',
              'Spatial blocking reduces leakage but does not eliminate short-distance dependence across block boundaries or the date line.',
              'Nearest-training-distance diagnostics are reported separately for all test pixels.',
              'Existing random30 metrics are contextual references, not a matched experiment isolating the spatial effect.',
              'Source data and the manuscript attribution models were not modified.', '',
              'Method references: https://doi.org/10.1111/ecog.02881 ; https://scikit-learn.org/stable/modules/cross_validation.html']
    (OUT / 'results_readme.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def plot(df, results, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from cartopy.io.shapereader import Reader
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'pdf.fonttype': 42})
    folder = OUT / 'figures'
    folder.mkdir(exist_ok=True)
    coasts = []
    for directory in [Path('C:/Users/user/.local/share/cartopy/shapefiles/natural_earth/physical'),
                      Path('C:/Users/user/AppData/Local/cartopy/shapefiles/natural_earth/physical')]:
        path = directory / 'ne_110m_coastline.shp'
        if path.exists():
            for geom in Reader(path).geometries():
                coasts.extend(np.column_stack(line.xy) for line in getattr(geom, 'geoms', [geom]))
            break
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
    cmap = ListedColormap(colors)
    cmap.set_bad('white')
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    for ax, veg in zip(axes, ['Forest', 'Non-forest']):
        g = df.loc[df.vegetation.eq(veg)]
        grid = np.full((3600, 7200), np.nan, dtype='float32')
        grid.ravel()[g.pixel_id.to_numpy()] = g.fold.to_numpy() + 1
        ax.imshow(grid[200:3000], extent=[-180, 180, -60, 80], vmin=.5, vmax=5.5,
                  origin='upper', interpolation='nearest', cmap=cmap, aspect='equal', rasterized=True)
        ax.set(xlim=(-180, 180), ylim=(-60, 80), xticks=[-120, 0, 120], yticks=[-40, 0, 40, 80])
        ax.set_title(f'{veg}: spatial test-fold assignments', loc='left', fontweight='bold')
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.7)
        if coasts:
            ax.add_collection(LineCollection(coasts, colors='#888888', linewidths=.35))
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=c) for c in colors], [f'Fold {f + 1}' for f in range(5)],
               loc='outside lower center', ncol=5, frameon=False)
    for ext in ['png', 'pdf']:
        fig.savefig(folder / f'Fig_RF_spatial5_fold_map.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
    for ax, veg, color in zip(axes, ['Forest', 'Non-forest'], ['#4D9221', '#C51B7D']):
        g = results.loc[results.vegetation.eq(veg)]
        for i, metric in enumerate(['accuracy', 'balanced_accuracy', 'roc_auc']):
            ax.scatter(i + np.linspace(-.07, .07, 5), 100 * g[metric], color=color, s=35)
            row = summary.loc[summary.vegetation.eq(veg) & summary.metric.eq(metric)].iloc[0]
            ax.plot([i - .2, i + .2], [100*row.fold_mean]*2, color='#222222', linewidth=1.5)
        ax.axhline(50, color='#999999', linestyle=':', linewidth=.8)
        ax.set(xticks=range(3), xticklabels=['Accuracy', 'Balanced\naccuracy', 'ROC-AUC'],
               ylabel='Fold score (%)', ylim=(40, 100), title=veg)
        ax.spines[['top', 'right']].set_visible(False)
    for ext in ['png', 'pdf']:
        fig.savefig(folder / f'Fig_RF_spatial5_performance.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df, names, protocol = prepare()
    if args.check_only:
        print(pd.read_csv(OUT / 'fold_sample_counts.csv').to_string(index=False), flush=True)
        return
    started = time.monotonic()
    groups = {veg: df.loc[df.vegetation.eq(veg)].reset_index(drop=True)
              for veg in ['Forest', 'Non-forest']}
    jobs = [(veg, fold) for fold in range(5) for veg in groups]
    results = []
    dump(OUT / 'status.json', dict(status='running', completed_folds=0, planned_folds=10))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_fold, groups[veg], names, fold) for veg, fold in jobs]
            for future in as_completed(futures):
                results.append(future.result())
                dump(OUT / 'status.json', dict(status='running', completed_folds=len(results), planned_folds=10))
        summary = summarize(df, names, results)
        assert digest(Path(protocol['table'])) == protocol['table_sha256']
        dump(OUT / 'status.json', dict(status='completed', completed_folds=10,
              OOF_pixels=len(df), elapsed_seconds=time.monotonic() - started,
              verification='Original input hash; disjoint outer/inner blocks; VIF<10; exactly one OOF prediction per pixel',
              spatial_independence_guaranteed=False))
        log('ALL 10 SPATIAL RF MODELS COMPLETED AND VERIFIED')
        print(summary.to_string(index=False), flush=True)
    except Exception as error:
        dump(OUT / 'status.json', dict(status='failed', error=str(error), completed_folds=len(results)))
        raise


if __name__ == '__main__':
    main()
