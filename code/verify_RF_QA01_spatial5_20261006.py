"""Verify saved spatial-fold metrics and representative serialized RF models."""
import joblib
import numpy as np
import pandas as pd
import run_RF_QA01_spatial5_20261006 as analysis


def main():
    protocol = analysis.helpers.read(analysis.OUT / 'protocol.json')
    source = pd.read_parquet(protocol['table']).set_index('pixel_id')
    assert analysis.digest(protocol['table']) == protocol['table_sha256']
    summary = pd.read_csv(analysis.OUT / 'performance_summary.csv')
    records, distances = [], []
    for veg in ['Forest', 'Non-forest']:
        for fold in range(1, 6):
            folder = analysis.OUT / veg / f'fold_{fold}'
            design = analysis.helpers.read(folder / 'design.json')
            saved = analysis.helpers.read(folder / 'metrics.json')
            p = pd.read_parquet(folder / 'predictions.parquet')
            calculated = analysis.helpers.legacy.metrics(
                p.y.to_numpy(), p.probability_decline.to_numpy(), p.baseline_prediction.to_numpy())
            for key, value in calculated.items():
                np.testing.assert_allclose(value, saved[key], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(p.y, source.loc[p.pixel_id, 'y'])
            assert design['max_VIF'] < 10 and design['test_filtered'] is False
            with np.load(folder / 'outer_split.npz') as split:
                group = source.loc[source.vegetation.eq(veg)].reset_index()
                assigned = pd.read_parquet(analysis.OUT / 'fold_assignments.parquet').set_index('pixel_id')
                train = group.iloc[split['training']].pixel_id
                test = group.iloc[split['test']].pixel_id
                assert not np.intersect1d(assigned.loc[train, 'block500'], assigned.loc[test, 'block500']).size
            restored_verified = False
            if fold == 1:
                trained = joblib.load(folder / 'model.joblib')
                assert len(trained.estimators_) == 300
                selected = p.iloc[:100]
                X = source.loc[selected.pixel_id, design['selected_features']].to_numpy('float32')
                np.testing.assert_allclose(trained.predict_proba(X)[:, 1], selected.probability_decline,
                                           rtol=0, atol=1e-12)
                restored_verified = True
                del trained
            records.append(dict(vegetation=veg, fold=fold, metrics_verified=True,
                                blocks_disjoint=True, restored_model_verified=restored_verified))
            distances.append(dict(vegetation=veg, fold=fold,
                **analysis.helpers.read(folder / 'nearest_training_distance.json')))
        p = pd.read_parquet(analysis.OUT / veg / 'OOF_predictions.parquet')
        expected = source.loc[source.vegetation.eq(veg)].sort_index()
        assert p.pixel_id.is_unique
        np.testing.assert_array_equal(p.pixel_id, expected.index)
        for row in summary.loc[summary.vegetation.eq(veg)].itertuples():
            scores = [analysis.helpers.read(analysis.OUT / veg / f'fold_{fold}/metrics.json')[row.metric]
                      for fold in range(1, 6)]
            np.testing.assert_allclose([np.mean(scores), np.std(scores, ddof=1)],
                                       [row.fold_mean, row.fold_SD], rtol=0, atol=1e-12)
    pd.DataFrame(distances).to_csv(analysis.OUT / 'nearest_training_distance_by_fold.csv', index=False)
    analysis.dump(analysis.OUT / 'verification.json', dict(status='passed',
        original_input_hash_unchanged=True, all10_fold_metrics_recomputed=True,
        exactly_one_OOF_prediction_per_pixel=True, representative_models_restored=2, folds=records))
    print('PASS: all 10 saved metrics, block separation, full OOF coverage and two restored models.', flush=True)


if __name__ == '__main__':
    main()
