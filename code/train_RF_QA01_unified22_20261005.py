"""Retune and refit 2 x 30 RF models using the unified22 first19 response."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import recompute_QA01_22periods_20261005 as data

rf, prep, np, pd = data.rf, data.prep, data.np, data.pd


def worker(job):
    veg, rep = job
    rf.THREADS = 8
    manifest = rf.read(data.DATA/'source_manifest.json')
    rf.run_group(veg, data.MODELS, Path(manifest['table']), manifest['candidates'], run_indices=[rep])
    return job


def verify(manifest):
    df = pd.read_parquet(manifest['table'])
    assert df.pixel_id.is_unique and (df.tac_pvalue < .05).all()
    assert (df.QA01_missing_fraction < .5).all()
    checks = []
    for veg in ['Forest', 'Non-forest']:
        group = df.loc[df.vegetation.eq(veg)].reset_index(drop=True)
        for rep in range(30):
            folder = data.MODELS/veg/'runs'/f'{rep:02d}'
            marker, design = rf.read(folder/'complete.json'), rf.read(folder/'design.json')
            assert marker['seed'] == rf.FIRST_SEED+rep
            assert rf.digest(folder/'model.joblib') == marker['model_sha256']
            assert rf.digest(folder/'importance.csv') == marker['importance_sha256']
            with np.load(folder/'split_indices.npz') as split:
                a, b, c = split['training_before_iqr'], split['training'], split['test']
                assert not np.intersect1d(a, c).size and np.isin(b, a).all()
                np.testing.assert_array_equal(np.sort(np.r_[a, c]), np.arange(len(group)))
            assert design['max_VIF'] < 10 and design['test_filtered'] is False
            prediction = pd.read_parquet(folder/'predictions.parquet')
            np.testing.assert_array_equal(prediction.pixel_id, group.iloc[c].pixel_id)
            np.testing.assert_array_equal(prediction.y, group.iloc[c].y)
            accuracy = float(((prediction.probability_decline >= .5).astype(int) == prediction.y).mean())
            assert abs(accuracy-rf.read(folder/'metrics.json')['accuracy']) < 1e-12
            with np.load(folder/'per_tree_OOB.npz') as saved:
                d = saved['delta']
                assert d.shape == (300, 3, len(design['selected_features'])) and np.isfinite(d).all()
                scaled = (d.mean(axis=0)/d.std(axis=0, ddof=1)).mean(axis=0)
                importance = pd.read_csv(folder/'importance.csv').set_index('variable').loc[design['selected_features']]
                np.testing.assert_allclose(importance.standardized_importance, scaled, rtol=1e-10, atol=1e-12)
            checks.append(dict(vegetation=veg, repeat=rep, n_training=len(b), n_test=len(c),
                               removed=', '.join(design['removed_features']), verified=True))
    pd.DataFrame(checks).to_csv(data.OUT/'RF_verification_all60.csv', index=False, encoding='utf-8-sig')
    prep.dump(data.OUT/'RF_verified.json', dict(status='verified', models=60,
             table_sha256=rf.digest(manifest['table']), same_unified_preprocessing_as_figures=True,
             RF_endpoints=[2005, 2023], candidates=manifest['candidates']))


def main():
    manifest = rf.read(data.DATA/'source_manifest.json')
    assert manifest['unified22'] and rf.digest(manifest['table']) == manifest['table_sha256']
    table = Path(manifest['table'])
    data.MODELS.mkdir(parents=True, exist_ok=True)
    config = dict(rf.CONFIG, version='QA01_unified22_first19_20261005',
                  preprocessing_period='2000-2025', TAC_endpoints=[2005, 2026], RF_endpoints=[2005, 2023],
                  threads_per_model=8, processes=4, tuning_threads_per_model=12,
                  missing_mask='QA0/1 2000-2023 missingness <0.5 plus original stable masks')
    protocol = dict(configuration=config, source_manifest_sha256=rf.digest(data.DATA/'source_manifest.json'),
                    table_sha256=manifest['table_sha256'], script_sha256=rf.digest(__file__),
                    training_helper_sha256=rf.digest(rf.__file__), original_models_preserved=True)
    if (data.MODELS/'protocol.json').exists():
        assert rf.read(data.MODELS/'protocol.json') == protocol
    else:
        prep.dump(data.MODELS/'protocol.json', protocol)
    started = time.monotonic()
    prep.dump(data.MODELS/'status.json', dict(status='running', stage='Bayesian retuning', models_planned=60))
    rf.THREADS = 12
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(rf.run_group, veg, data.MODELS, table, manifest['candidates'], run_indices=[])
                   for veg in ['Forest', 'Non-forest']]
        for f in futures:
            f.result()
    pending = [(veg, rep) for rep in range(30) for veg in ['Forest', 'Non-forest']
               if not (data.MODELS/veg/'runs'/f'{rep:02d}'/'complete.json').exists()]
    completed = 60-len(pending)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, job) for job in pending]
        for f in as_completed(futures):
            job = f.result()
            completed += 1
            prep.dump(data.MODELS/'status.json', dict(status='running', stage='RF and OOB importance',
                      completed_models=completed, planned_models=60, last_completed=list(job)))
            prep.log(f'Unified22 RF {completed}/60 completed')
    for veg in ['Forest', 'Non-forest']:
        rf.summarize_group(veg, data.MODELS/veg, manifest['candidates'])
    rf.figures(data.MODELS)
    for ext in ['png', 'pdf']:
        (data.MODELS/f'figures/Fig_RF_QA0_OOB_top10_30runs.{ext}').replace(
             data.MODELS/f'figures/Fig_RF_QA01_unified22_OOB_top10_30runs.{ext}')
    verify(manifest)
    prep.dump(data.MODELS/'status.json', dict(status='completed', models=60,
              verification='all60 hashes, splits, VIF, predicted labels and standardized OOB checked',
              elapsed_seconds=time.monotonic()-started))
    prep.log('Unified22 RF ALL60 COMPLETED AND VERIFIED')


if __name__ == '__main__':
    main()
