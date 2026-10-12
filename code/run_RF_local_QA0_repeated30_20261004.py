"""Training-only VIF, TPE tuning, 30 random RF refits and exact tree OOB importance."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import prepare_RF_local_QA0_20261004 as preparation
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = preparation.OUT
LEGACY_SCRIPT = preparation.ROOT/'xgboost_shap_spatial_20260924/rf_legacy_inputs_20260924/rf_drop_P_Ta_20260925/code/run_rf.py'
spec = importlib.util.spec_from_file_location('rf_helpers', LEGACY_SCRIPT)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
THREADS = 16
TREES = 300
N_RUNS = 30
FIRST_SEED = 2026100400
CONFIG = dict(version='local_QA0_rebuilt_RF_30_v1', trees=TREES, repeats=N_RUNS,
              first_seed=FIRST_SEED, threads_per_model=THREADS, concurrent_groups=2,
              outer_test_fraction=.3, inner_validation_fraction=.25, TPE_trials=24,
              tuning_training_cap=80000, tuning_validation_cap=30000,
              tuning='Seed-0 inner validation only; top three candidates verified on full inner split',
              objective='accuracy, then AUC tie-break; test never used for parameter selection',
              VIF='training only, iteratively remove highest VIF until <10; no protected predictors',
              response='class 1: delta_TAC >=0 (decline); class 0: delta_TAC <0 (increase)',
              sample='strict ordinary OLS P<0.05; all 22 candidates finite',
              missing_mask='QA0 missing fraction over 2000-2023 <0.5, in addition to retained original stable mask',
              training_outliers='1.5 IQR of training response only; tests unfiltered',
              bootstrap='standard RF with-replacement bootstrap, full training size per tree',
              importance='per-tree OOB permutation error change; three permutations; mean across trees divided by across-tree SD, then average permutations',
              parameter_policy='Retune once per vegetation; fix parameters across 30 refits',
              performance_scope='Seed-0 test is excluded from tuning; later splits describe internal stability, not fresh nested independent validation',
              intervals='2.5th and 97.5th percentiles of overlapping refits; conditional on retention',
              spatially_independent_validation=False)


def dump(path, value):
    preparation.dump(path, value)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return preparation.package.digest(Path(path))


def log(veg, message):
    preparation.log(f'{veg}: {message}')


def vif_select(X, names):
    std = np.std(X, axis=0, dtype=np.float64)
    retained = np.flatnonzero(std > 1e-12).tolist()
    history = [dict(iteration=-1, variable=names[j], vif=None, removed=True, reason='constant')
               for j in np.flatnonzero(std <= 1e-12)]
    assert retained, 'No varying predictors'
    correlation = np.atleast_2d(np.corrcoef(X[:, retained].astype('float64'), rowvar=False))
    iteration = 0
    while retained:
        regularized = bool(np.linalg.eigvalsh(correlation).min() < 1e-10)
        matrix = correlation + np.eye(len(retained))*1e-10 if regularized else correlation
        values = np.diag(np.linalg.inv(matrix))
        worst = int(np.argmax(values))
        remove = bool(values[worst] >= 10)
        for j, value in enumerate(values):
            history.append(dict(iteration=iteration, variable=names[retained[j]], vif=float(value),
                                removed=bool(remove and j == worst), near_singular_regularization=regularized,
                                reason='VIF >= 10' if remove and j == worst else 'retained_at_step'))
        if not remove:
            break
        retained.pop(worst)
        correlation = np.delete(np.delete(correlation, worst, axis=0), worst, axis=1)
        iteration += 1
    assert retained and np.isfinite(values).all() and values.max() < 10
    return retained, pd.DataFrame(history)


def rf(params, seed, n_features):
    return RandomForestClassifier(n_estimators=TREES, random_state=seed, n_jobs=THREADS,
        bootstrap=True, criterion='gini', max_depth=None, class_weight=None,
        min_samples_leaf=params['min_samples_leaf'], max_features=min(params['max_features'], n_features))


def tune(veg, folder, df, X, y, names):
    target = folder/'tuning/selection.json'
    if target.exists():
        saved = read(target)
        assert saved['test_used'] is False
        log(veg, 'verified completed tuning cache')
        return saved['chosen']['params']
    tune_dir = target.parent
    tune_dir.mkdir(parents=True, exist_ok=True)
    outer, test = train_test_split(np.arange(len(df)), test_size=.3, stratify=y, random_state=FIRST_SEED)
    inner0, val = train_test_split(np.sort(outer), test_size=.25, stratify=y[np.sort(outer)], random_state=FIRST_SEED+1100)
    inner0, val = np.sort(inner0), np.sort(val)
    inner, iqr = legacy.iqr(df, np.sort(inner0))
    selected, history = vif_select(X[inner], names)
    history.to_csv(tune_dir/'inner_VIF.csv', index=False)
    fs = legacy.sample(inner, y, 80000, FIRST_SEED+1101)
    vs = legacy.sample(np.sort(val), y, 30000, FIRST_SEED+1102)
    assert not np.intersect1d(inner0, val).size
    assert not np.intersect1d(outer, test).size
    np.savez_compressed(tune_dir/'split_indices.npz', outer_training=np.sort(outer), untouched_test=np.sort(test),
                        inner_training=inner, inner_validation=np.sort(val), sampled_fit=fs, sampled_validation=vs)
    dump(tune_dir/'design.json', dict(selected_features=[names[j] for j in selected],
        training_iqr=iqr, n_fit=len(inner), n_validation=len(val), n_sampled_fit=len(fs),
        n_sampled_validation=len(vs), n_untouched_test=len(test)))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler_path = tune_dir/'sampler.joblib'
    sampler = joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=FIRST_SEED, n_startup_trials=8)
    storage = optuna.storages.RDBStorage(url='sqlite:///'+(tune_dir/'tuning.sqlite3').as_posix())
    study = optuna.create_study(study_name='RF_QA0', direction='maximize', sampler=sampler,
                                storage=storage, load_if_exists=True)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    reference = dict(min_samples_leaf=5, max_features=min(4, len(selected)))
    if not study.trials:
        study.enqueue_trial(reference)

    def objective(trial):
        params = dict(min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 64, log=True),
                      max_features=trial.suggest_int('max_features', 2, len(selected)))
        duplicate = next((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                          and t.params == params), None)
        if duplicate is not None:
            trial.set_user_attr('auc', duplicate.user_attrs['auc'])
            trial.set_user_attr('duplicate_of', duplicate.number)
            return duplicate.value
        model = rf(params, FIRST_SEED, len(selected)).fit(X[fs][:, selected], y[fs])
        probability = model.predict_proba(X[vs][:, selected])[:, 1]
        trial.set_user_attr('auc', float(roc_auc_score(y[vs], probability)))
        return float(accuracy_score(y[vs], probability >= .5))

    def callback(study, trial):
        study.trials_dataframe().to_csv(tune_dir/'trials.csv', index=False)
        joblib.dump(study.sampler, sampler_path)
        log(veg, f'TPE {trial.number+1}/24, inner accuracy={trial.value:.2%}, {trial.params}')

    log(veg, f'TPE starting; {len(selected)} predictors; {len(fs):,}/{len(vs):,} tuning pixels')
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective, n_trials=max(0, 24-completed), callbacks=[callback])
    ranked = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                    key=lambda t: (-t.value, -t.user_attrs['auc'], t.number))
    choices, seen = [], set()
    for trial in ranked:
        key = tuple(sorted(trial.params.items()))
        if key not in seen:
            choices.append((f'trial_{trial.number}', trial.params))
            seen.add(key)
        if len(choices) == 3:
            break
    if tuple(sorted(reference.items())) not in seen:
        choices.append(('reference', reference))
    scores = []
    for tag, params in choices:
        cached = tune_dir/f'full_inner_{tag}.json'
        if cached.exists():
            score = read(cached)
        else:
            model = rf(params, FIRST_SEED, len(selected)).fit(X[inner][:, selected], y[inner])
            p = model.predict_proba(X[np.sort(val)][:, selected])[:, 1]
            score = dict(name=tag, params=params, accuracy=float(accuracy_score(y[val], p >= .5)),
                         auc=float(roc_auc_score(y[val], p)))
            dump(cached, score)
        scores.append(score)
        log(veg, f'Full inner check {tag}: {score["accuracy"]:.2%}')
    chosen = sorted(scores, key=lambda s: (-s['accuracy'], -s['auc'], s['name']))[0]
    storage.remove_session()
    storage.engine.dispose()
    dump(target, dict(chosen=chosen, candidates=scores, test_used=False))
    return chosen['params']


def tree_oob(tree, k, X, y, seed):
    indices = _generate_unsampled_indices(tree.random_state, len(y), len(y))
    values = np.array(X[indices], dtype='float32', order='C')
    labels = y[indices]
    base = np.mean(tree.predict(values, check_input=False) != labels)
    delta = np.zeros((3, X.shape[1]))
    for j in range(X.shape[1]):
        original = values[:, j].copy()
        for r in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1200, k, j, r]))
            values[:, j] = original[rng.permutation(len(original))]
            delta[r, j] = np.mean(tree.predict(values, check_input=False) != labels)-base
        values[:, j] = original
    return delta, float(base), len(indices)


def run_group(veg, output, table, names, run_indices=None):
    folder = output/veg
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(table)
    df = df.loc[df.vegetation.eq(veg)].reset_index(drop=True)
    assert df.pixel_id.is_unique and (df.tac_pvalue < .05).all()
    X = df[names].to_numpy('float32')
    y = df.y.to_numpy('int8')
    assert np.isfinite(X).all() and np.array_equal(np.unique(y), [0, 1])
    params = tune(veg, folder, df, X, y, names)
    log(veg, f'Selected parameters: {params}; eligible pixels={len(df):,}')
    for rep in (range(N_RUNS) if run_indices is None else run_indices):
        seed = FIRST_SEED+rep
        run = folder/'runs'/f'{rep:02d}'
        run.mkdir(parents=True, exist_ok=True)
        complete = run/'complete.json'
        if complete.exists():
            saved = read(complete)
            assert saved['seed'] == seed and digest(run/'model.joblib') == saved['model_sha256']
            continue
        started = time.monotonic()
        train0, test = train_test_split(np.arange(len(df)), test_size=.3, stratify=y, random_state=seed)
        train0, test = np.sort(train0), np.sort(test)
        train, iqr = legacy.iqr(df, train0)
        selected, history = vif_select(X[train], names)
        features = [names[j] for j in selected]
        history.to_csv(run/'VIF_history.csv', index=False)
        assert not np.intersect1d(train0, test).size
        np.testing.assert_array_equal(np.sort(np.r_[train0, test]), np.arange(len(df)))
        np.savez_compressed(run/'split_indices.npz', training_before_iqr=train0, training=train, test=test)
        design = dict(seed=seed, run=rep, selected_features=features,
                      removed_features=[n for n in names if n not in features], n_eligible=len(df),
                      n_training_before_iqr=len(train0), n_training=len(train), n_test=len(test),
                      training_IQR=iqr, test_filtered=False, max_VIF=float(history.loc[history.iteration == history.iteration.max(), 'vif'].max()),
                      training_decline_fraction=float(y[train].mean()), test_decline_fraction=float(y[test].mean()))
        dump(run/'design.json', design)
        Xt = np.ascontiguousarray(X[train][:, selected])
        yt = y[train]
        trained = run/'trained.json'
        if trained.exists():
            saved = read(trained)
            assert saved['seed'] == seed and saved['model_sha256'] == digest(run/'model.joblib')
            model = joblib.load(run/'model.joblib')
        else:
            log(veg, f'Run {rep+1}/30: fit {len(train):,} training pixels; {len(features)} features')
            model = rf(params, seed, len(selected)).fit(Xt, yt)
            assert np.array_equal(model.classes_, [0, 1])
            p = model.predict_proba(X[test][:, selected])[:, 1]
            baseline = np.full(len(test), int(yt.mean() >= .5), dtype='int8')
            result = legacy.metrics(y[test], p, baseline)
            dump(run/'metrics.json', dict(run=rep, seed=seed, **result))
            predictions = df.iloc[test][['pixel_id', 'longitude', 'latitude', 'y', 'tac_pvalue']].copy()
            predictions['probability_decline'] = p
            predictions.to_parquet(run/'predictions.parquet', index=False)
            joblib.dump(model, run/'model.joblib', compress=3)
            restored = joblib.load(run/'model.joblib')
            np.testing.assert_array_equal(restored.predict_proba(X[test[:100]][:, selected])[:, 1], p[:100])
            del restored
            dump(trained, dict(seed=seed, model_sha256=digest(run/'model.joblib')))
            log(veg, f'Run {rep+1}/30: accuracy={result["accuracy"]:.2%}, AUC={result["roc_auc"]:.4f}; OOB starting')
        if not (run/'importance.csv').exists():
            results = joblib.Parallel(n_jobs=THREADS, prefer='threads')(
                joblib.delayed(tree_oob)(tree, k, Xt, yt, seed) for k, tree in enumerate(model.estimators_))
            delta = np.stack([r[0] for r in results])
            mean = delta.mean(axis=0)
            sd = delta.std(axis=0, ddof=1)
            assert np.isfinite(delta).all() and (sd > 0).all()
            importance = pd.DataFrame(dict(variable=features, raw_delta_error=mean.mean(axis=0),
                raw_delta_error_pp=100*mean.mean(axis=0), standardized_importance=(mean/sd).mean(axis=0)))
            importance['rank_standardized'] = importance.standardized_importance.rank(ascending=False, method='min').astype(int)
            importance.sort_values('rank_standardized').to_csv(run/'importance.csv', index=False)
            np.savez_compressed(run/'per_tree_OOB.npz', delta=delta, baseline_error=[r[1] for r in results],
                                n_oob=[r[2] for r in results])
        dump(complete, dict(seed=seed, run=rep, model_sha256=digest(run/'model.joblib'),
                            importance_sha256=digest(run/'importance.csv'), elapsed_seconds=time.monotonic()-started,
                            train_test_disjoint=True, test_unfiltered=True, selected_features=features))
        log(veg, f'Run {rep+1}/30 COMPLETE, {(time.monotonic()-started)/60:.1f} min')
        del model, Xt
    if run_indices is None:
        summarize_group(veg, folder, names)


def summarize_group(veg, folder, names):
    summary = folder/'summary'
    summary.mkdir(exist_ok=True)
    metrics, importance, designs = [], [], []
    for rep in range(N_RUNS):
        run = folder/'runs'/f'{rep:02d}'
        complete = read(run/'complete.json')
        assert complete['seed'] == FIRST_SEED+rep
        assert complete['importance_sha256'] == digest(run/'importance.csv')
        metrics.append(read(run/'metrics.json'))
        designs.append(read(run/'design.json'))
        entry = pd.read_csv(run/'importance.csv')
        entry['run'] = rep
        importance.append(entry)
    performance = pd.DataFrame(metrics)
    performance.to_csv(summary/'performance_all30.csv', index=False)
    dump(summary/'untouched_seed0_test_metrics.json', metrics[0])
    perf = []
    for metric in ['accuracy', 'macro_f1', 'balanced_accuracy', 'roc_auc', 'majority_baseline_accuracy',
                   'recall_declining', 'recall_increasing']:
        values = performance[metric].to_numpy()
        perf.append(dict(metric=metric, median=float(np.median(values)), q025=float(np.quantile(values, .025)),
                         q975=float(np.quantile(values, .975)), minimum=float(values.min()), maximum=float(values.max())))
    pd.DataFrame(perf).to_csv(summary/'performance_summary.csv', index=False)
    full = pd.concat(importance, ignore_index=True)
    full.to_csv(summary/'importance_all30.csv', index=False)
    rows = []
    for name in names:
        group = full.loc[full.variable.eq(name)]
        row = dict(variable=name, retention_count=len(group), retention_fraction=len(group)/N_RUNS,
                   top6_count=int(group.rank_standardized.le(6).sum()),
                   median_rank=float(group.rank_standardized.median()) if len(group) else None)
        for metric in ['standardized_importance', 'raw_delta_error_pp']:
            row[metric] = float(group[metric].median()) if len(group) else None
            row[metric+'_q025'] = float(group[metric].quantile(.025)) if len(group) else None
            row[metric+'_q975'] = float(group[metric].quantile(.975)) if len(group) else None
        rows.append(row)
    ranking = pd.DataFrame(rows).sort_values('standardized_importance', ascending=False).reset_index(drop=True)
    ranking['rank_by_median'] = np.arange(1, len(ranking)+1)
    ranking.to_csv(summary/'importance_summary.csv', index=False)
    pd.DataFrame([{k: d[k] for k in ['run', 'seed', 'n_eligible', 'n_training_before_iqr', 'n_training', 'n_test', 'max_VIF']}
                  for d in designs]).to_csv(summary/'sample_counts_all30.csv', index=False)
    pd.DataFrame([dict(run=d['run'], variable=n, retained=n in d['selected_features'])
                  for d in designs for n in names]).to_csv(summary/'VIF_retention_all30.csv', index=False)
    log(veg, 'ALL 30 COMPLETE; top six: '+', '.join(ranking.loc[ranking.retention_count.gt(0), 'variable'].head(6)))


def figures(output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    palette = plt.get_cmap('PiYG_r')
    for ax, veg, color in zip(axes, ['Forest', 'Non-forest'], [palette(1/9), palette(8/9)]):
        ranking = pd.read_csv(output/veg/'summary/importance_summary.csv').dropna(subset=['standardized_importance']).head(10)
        y = np.arange(len(ranking))
        values = ranking.standardized_importance.to_numpy()
        lower = values-ranking.standardized_importance_q025.to_numpy()
        upper = ranking.standardized_importance_q975.to_numpy()-values
        ax.barh(y, values, color=color, height=.7)
        ax.errorbar(values, y, xerr=[lower, upper], fmt='none', color='#333333', capsize=3, lw=1)
        labels = [n.replace('P_SI', 'P.SI').replace('P_NIPV', 'P.NIPV') for n in ranking.variable]
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(veg, fontweight='bold')
        ax.set_xlabel('Standardized OOB permutation importance')
        ax.spines[['top', 'right']].set_visible(False)
        for i, row in enumerate(ranking.itertuples()):
            ax.text(.98, i, f'{row.standardized_importance:.3f} [{row.standardized_importance_q025:.3f}, {row.standardized_importance_q975:.3f}]',
                    ha='right', va='center', fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_xlim(0, max(ranking.standardized_importance_q975.max()*1.9, .1))
    fig.tight_layout()
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(folder/f'Fig_RF_QA0_OOB_top10_30runs.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--last-endpoint', type=int, choices=[2023, 2025], default=2023)
    args = parser.parse_args()
    manifest = read(ROOT/'data/source_manifest.json')
    assert manifest['status'] == 'completed' and manifest['forbidden_data_read'] is False
    selected = manifest['qa0_tables'][str(args.last_endpoint)]
    table = Path(selected['path'])
    assert digest(table) == selected['sha256']
    output = ROOT/f'RF_endpoints_2005_{args.last_endpoint}_QA0_MDNlt0p5_repeated30'
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(configuration=CONFIG, last_endpoint=args.last_endpoint, source_manifest_sha256=digest(ROOT/'data/source_manifest.json'),
                    table_sha256=selected['sha256'], script_sha256=digest(__file__), helper_sha256=digest(LEGACY_SCRIPT),
                    source_manifest=str(ROOT/'data/source_manifest.json'), output=str(output))
    path = output/'protocol.json'
    if path.exists():
        assert read(path) == protocol, 'Protocol changed; do not mix runs'
    else:
        dump(path, protocol)
    dump(output/'status.json', dict(status='running', configuration=CONFIG, last_endpoint=args.last_endpoint))
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_group, veg, output, table, manifest['candidates']) for veg in ['Forest', 'Non-forest']]
            for future in futures:
                future.result()
        figures(output)
        dump(output/'status.json', dict(status='completed', models=60, last_endpoint=args.last_endpoint,
                                       config=CONFIG, completion_checks='all 60 run markers and importance hashes verified'))
        preparation.log(f'RF ALL COMPLETED: {output}')
    except Exception as error:
        dump(output/'status.json', dict(status='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()
