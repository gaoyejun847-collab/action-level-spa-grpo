"""Regression for real verl NumPy validation metrics."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from spa_next.training.verl_v1 import record_validation


class ValidationSerialization(unittest.TestCase):
    def test_numpy_metrics_preserve_checkpoint_and_best_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'global_step_0'
            source.mkdir()
            (source / 'data.pt').write_bytes(b'checkpoint fixture')
            trainer = SimpleNamespace(global_steps=np.int32(0), config=SimpleNamespace(
                algorithm=SimpleNamespace(adv_estimator='spa_v1'),
                trainer={'default_local_dir': tmp}))
            class Config(dict):
                __getattr__ = dict.__getitem__
            trainer.config.trainer = Config(default_local_dir=tmp)
            metrics = {'val-aux/webshop/success/mean@1': np.float64(.25),
                       'val-core/webshop/reward/mean@1': np.float32(.6),
                       'val-aux/num_turns/max': np.int32(20)}
            record_validation(trainer, metrics)
            saved = json.loads((root / 'best_validation.json').read_text())
            self.assertEqual(saved['metrics']['val-aux/num_turns/max'], 20)
            self.assertEqual(saved['step'], 0)
            self.assertEqual((Path(saved['checkpoint']) / 'data.pt').read_bytes(), b'checkpoint fixture')
            record_validation(trainer, metrics)  # Same result is idempotent.
