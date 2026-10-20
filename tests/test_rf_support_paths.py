"""Small RF helper checks; no source rasters, model caches or training jobs."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'code/rf_support/run_rf.py'


class RFSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Isolate the three utilities; these checks do not require the tuning stack.
        tree = ast.parse(HELPER.read_text(encoding='utf-8-sig'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('iqr', 'sample', 'metrics')]
        namespace = dict(np=np, train_test_split=train_test_split, accuracy_score=accuracy_score,
                         balanced_accuracy_score=balanced_accuracy_score, confusion_matrix=confusion_matrix,
                         f1_score=f1_score, roc_auc_score=roc_auc_score)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(HELPER), 'exec'), namespace)
        cls.helper = SimpleNamespace(**{name: namespace[name] for name in ('iqr', 'sample', 'metrics')})

    def test_current_wrapper_resolves_shared_module(self):
        tree = ast.parse((ROOT / 'code/run_RF_local_QA0_repeated30_20261004.py').read_text(encoding='utf-8-sig'))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LEGACY_SCRIPT' for t in n.targets))
        value = eval(compile(ast.Expression(assignment.value), '<module lookup>', 'eval'),
                     {'Path': Path, '__file__': str(ROOT / 'code/run_RF_local_QA0_repeated30_20261004.py')})
        self.assertEqual(value, HELPER)
        self.assertTrue(value.is_file())

    def test_iqr_keeps_only_training_response_range(self):
        df = pd.DataFrame({'y_continuous': [0., 1., 2., 3., 100., -1000.]})
        selected, _ = self.helper.iqr(df, np.arange(5))
        np.testing.assert_array_equal(selected, [0, 1, 2, 3])

    def test_sample_is_reproducible_and_stratified(self):
        y = np.tile([0, 1], 50)
        a = self.helper.sample(np.arange(100), y, 20, 12)
        b = self.helper.sample(np.arange(100), y, 20, 12)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(int(y[a].sum()), 10)

    def test_metrics_match_sklearn(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        probability = np.array([.1, .7, .8, .4, .2, .9])
        result = self.helper.metrics(y, probability, np.zeros_like(y))
        predicted = (probability >= .5).astype(int)
        for name, expected in [('accuracy', accuracy_score(y, predicted)),
                               ('balanced_accuracy', balanced_accuracy_score(y, predicted)),
                               ('macro_f1', f1_score(y, predicted, average='macro')),
                               ('roc_auc', roc_auc_score(y, probability))]:
            self.assertAlmostEqual(result[name], expected)


if __name__ == '__main__':
    unittest.main()
