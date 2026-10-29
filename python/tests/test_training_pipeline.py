import contextlib
import io
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


class TrainingPipelineTests(unittest.TestCase):
    def test_training_evaluation_heatmaps_and_safe_snapshot(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory(prefix='fcdd-pipeline-') as directory:
            root = Path(directory)
            random = np.random.default_rng(7)
            for split in ('train', 'test'):
                for label in ('normal', 'anomalous'):
                    folder = root / 'data' / 'custom' / split / 'breast_img' / label
                    folder.mkdir(parents=True)
                    for index in range(2):
                        pixels = random.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                        Image.fromarray(pixels).save(folder / f'{index}.png')
            arguments = [
                'run_custom.py', '--datadir', str(root / 'data'),
                '--logdir', str(root / 'results'), '--supervise-mode', 'other',
                '--noise-mode', 'gaussian', '--net', 'FCDD_CNN224',
                '--workers', '0', '--it', '1', '--epochs', '1',
                '--batch-size', '2', '--preproc', 'none', '--cpu',
            ]
            with patch('sys.argv', arguments), patch('fcdd.datasets.CUSTOM_CLASSES', []):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    runpy.run_module('fcdd.runners.run_custom', run_name='__main__')
            output = root / 'results_custom_' / 'normal_0' / 'it_0'
            for name in ('snapshot.pt', 'history.json', 'roc.json', 'heatmaps_global.png'):
                self.assertTrue((output / name).is_file(), name)
            snapshot = torch.load(output / 'snapshot.pt', map_location='cpu', weights_only=True)
            self.assertEqual(snapshot['epoch'], 1)
            self.assertIn('net', snapshot)


if __name__ == '__main__':
    unittest.main()
