import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import struct
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zlib

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from fcdd.datasets import safe_io
from fcdd.datasets.image_folder import ADImageFolderDataset, ImageFolderDataset
from fcdd.datasets.image_folder_gtms import ImageFolderDatasetGTM
from fcdd.datasets.image_folder_refs import ADImageRefDataset, DatasetREF
from fcdd.datasets.imagenet import PathsMetaFileImageNet
from fcdd.datasets.online_supervisor import OnlineSupervisor, repeat_loader
from fcdd.datasets.outlier_exposure.imagenet import MyImageFolder, OEImageNet, OEImageNet22k
from fcdd.datasets.preprocessing import MultiCompose
from fcdd.models import weights
from fcdd.runners.bases import ClassesRunner, SeedsRunner, extract_viz_ids
from fcdd.runners.argparse_configs import DefaultConfig
from fcdd.runners.predictor import _resolve_path, load_config, load_model, load_model_ref
from fcdd.training.bases import BaseADTrainer, BaseTrainer
from fcdd.util import safety
from fcdd.util.io import extract_args, read_cfg
from fcdd.util.logging import Logger


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def image(self, relative='images/class0/sample.png', shape=(8, 8)):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.arange(shape[0] * shape[1] * 3, dtype=np.uint8).reshape(*shape, 3)
        Image.fromarray(pixels).save(path)
        return path

    def link(self, target, path, directory=False):
        try:
            path.symlink_to(target, target_is_directory=directory)
        except OSError as error:
            self.skipTest(f'Symlink creation is not permitted on this host: {error}')


class PathTests(TemporaryTest):
    def test_confined_paths_accept_nested_and_contained_absolute(self):
        expected = str(self.root / 'nested' / 'output')
        self.assertEqual(safety.confined_path(self.root, 'nested', 'output'), expected)
        self.assertEqual(safety.confined_path(self.root, expected), expected)

    def test_confined_paths_reject_traversal_absolute_and_drive_syntax(self):
        for path in ('../escape', '..\\escape', str(self.root.parent / 'escape'),
                     'C:escape', 'file:stream', 'a/../../escape', 'a\0b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safety.confined_path(self.root, path)

    def test_confined_paths_reject_file_and_directory_links(self):
        target = self.image()
        file_link = self.root / 'file-link.png'
        self.link(target, file_link)
        with self.assertRaises(ValueError):
            safety.confined_path(self.root, file_link)
        folder_link = self.root / 'folder-link'
        self.link(target.parent, folder_link, directory=True)
        with self.assertRaises(ValueError):
            safety.confined_path(self.root, folder_link / target.name)

    def test_dataset_anchor_preserves_legacy_data_paths(self):
        base = self.root / 'data' / 'results' / 'run'
        dataset = self.root / 'data' / 'MIP'
        base.mkdir(parents=True)
        dataset.mkdir()
        self.assertEqual(_resolve_path('../../data/MIP', str(base)), str(dataset))
        self.assertEqual(_resolve_path('..\\..\\data\\MIP', str(base)), str(dataset))

    def test_dataset_anchor_cannot_escape_or_follow_links(self):
        base = self.root / 'data' / 'results'
        base.mkdir(parents=True)
        outside = self.root / 'outside'
        outside.mkdir()
        for path in ('../../data/../outside', '../../data/MIP/../../outside',
                     '../outside', str(outside)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                _resolve_path(path, str(base))
        self.assertEqual(_resolve_path(str(outside), str(base), explicit=True), str(outside))
        self.link(outside, self.root / 'data' / 'linked', directory=True)
        with self.assertRaises(ValueError):
            _resolve_path('../../data/linked', str(base))

    def test_plain_relative_dataset_stays_under_snapshot(self):
        (self.root / 'images').mkdir()
        self.assertEqual(_resolve_path('images', str(self.root)), str(self.root / 'images'))

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'Named pipes are a POSIX filesystem feature')
    def test_special_files_are_rejected_without_blocking(self):
        path = self.root / 'fifo'
        os.mkfifo(path)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            with safety.bounded_reader(path, 1024):
                self.fail('FIFO was accepted as a regular file')


class ResourceTests(unittest.TestCase):
    def test_resource_boundaries(self):
        for key, low, high in (
            ('batch_size', 1, safety.MAX_BATCH_SIZE), ('workers', 0, safety.MAX_WORKERS),
            ('epochs', 0, safety.MAX_EPOCHS), ('acc_batches', 1, safety.MAX_BATCH_SIZE),
            ('oe_limit', 1, safety.MAX_SAMPLES), ('resdown', 1, 4096),
        ):
            for value in (low, high):
                safety.validate_resources({key: value})
            for value in (low - 1, high + 1, True, '2', 1.5, float('inf'), float('nan')):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    safety.validate_resources({key: value})
        with self.assertRaises(ValueError):
            safety.validate_resources({'batch_size': 128, 'acc_batches': 128})

    def test_quantile_rejects_invalid_values_before_model_allocation(self):
        for value in (-0.01, 1.01, 'nan', 'inf', '-inf', None, True, 'bad'):
            for load in (load_model, load_model_ref):
                with self.subTest(value=value, load=load), self.assertRaises(ValueError):
                    load({'quantile': value}, None)
        for value in (0, 1, '0.97'):
            self.assertEqual(safety.quantile_value(value), float(value))

    def test_quantile_endpoints_and_constant_maps_are_finite(self):
        for qu in (0, 0.001, 0.97, 1):
            for image in (torch.zeros(2, 1, 2, 2), torch.arange(8).reshape(2, 1, 2, 2).float()):
                for norm in (BaseADTrainer._BaseADTrainer__global_norm, BaseADTrainer._BaseADTrainer__local_norm):
                    result = norm(image.clone(), qu)
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_outlier_limits_are_checked_before_dataset_access(self):
        for dataset in (OEImageNet, OEImageNet22k):
            for shape in ((1, 3, 1_000_000, 1_000_000), (1024, 3, 4096, 4096), (1, 2, 8, 8)):
                with self.subTest(dataset=dataset, shape=shape), self.assertRaises(ValueError):
                    dataset(shape, root='not-present')
            with self.assertRaises(ValueError):
                dataset((1, 3, 8, 8), root='not-present', limit_var=float('inf'))
        with self.assertRaises(ValueError):
            OnlineSupervisor(None, 'noise', 'imagenet', oe_limit=safety.MAX_SAMPLES + 1)
        with self.assertRaises(ValueError):
            OnlineSupervisor(None, 'noise', 'not-a-noise-mode')

    def test_repeated_loader_does_not_cache_batches(self):
        class Loader:
            calls = 0

            def __iter__(self):
                self.calls += 1
                yield self.calls

        loader = Loader()
        stream = repeat_loader(loader)
        self.assertEqual([next(stream) for _ in range(3)], [1, 2, 3])


class DatasetTests(TemporaryTest):
    def refs(self, actual=None, reference=None, rows=1, **kwargs):
        actual = actual or self.image('custom/actual.png')
        reference = reference or self.image('custom/reference.png')
        path = self.root / 'custom' / 'train_ref.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('Actual', 'Reference', 'Label'))
            for _ in range(rows):
                writer.writerow((actual, reference, 0))
        return DatasetREF(str(path), 'unsupervised', (3, 8, 8), 0, 1,
                          root=str(self.root), **kwargs)

    def test_reference_dataset_normal_images_and_relative_paths(self):
        self.image('custom/actual.png')
        self.image('custom/reference.png')
        ds = self.refs('custom/actual.png', 'custom/reference.png',
                       transform=MultiCompose([transforms.ToTensor()]))
        actual, target, reference = ds[0]
        self.assertEqual(tuple(actual.shape), (3, 8, 8))
        self.assertEqual(target, 0)
        torch.testing.assert_close(actual, reference)

    def test_reference_paths_are_confined_in_both_columns(self):
        inside = self.image()
        for bad in ('../outside.png', str(self.root.parent / 'outside.png')):
            for actual, reference in ((bad, inside), (inside, bad)):
                with self.subTest(actual=actual, reference=reference), self.assertRaises(ValueError):
                    self.refs(actual, reference)

    def test_reference_image_links_are_rejected(self):
        image = self.image('custom/actual.png')
        linked = image.parent / 'linked.png'
        self.link(image, linked)
        with self.assertRaises(ValueError):
            self.refs(image, linked)

    def test_reference_csv_has_byte_and_row_limits(self):
        with patch('fcdd.datasets.image_folder_refs.MAX_CSV_BYTES', 16):
            with self.assertRaisesRegex(ValueError, 'regular file'):
                self.refs()
        with patch('fcdd.datasets.image_folder_refs.MAX_SAMPLES', 1):
            with self.assertRaisesRegex(ValueError, 'sample budget'):
                self.refs(rows=2)

    def test_reference_csv_rejects_invalid_schema_and_labels(self):
        path = self.root / 'refs.csv'
        for contents in ('Actual,Label\nx,0\n', 'Actual,Reference,Label\nx,y,2\n',
                         'Actual,Reference,Label\n,y,0\n'):
            path.write_text(contents, encoding='utf-8')
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                DatasetREF(str(path), 'unsupervised', (3, 8, 8), 0, 1)

    def test_oversized_image_headers_are_rejected_before_decode_for_both_columns(self):
        path = self.image('custom/bomb.png')
        data = bytearray(path.read_bytes())
        data[16:24] = struct.pack('>II', 5000, 5000)
        data[29:33] = struct.pack('>I', zlib.crc32(data[12:29]))
        path.write_bytes(data)
        normal = self.image('custom/normal.png')
        for actual, reference in ((path, normal), (normal, path)):
            ds = self.refs(actual, reference)
            with self.subTest(actual=actual, reference=reference), self.assertRaisesRegex(ValueError, 'pixels'):
                ds[0]

    def test_image_file_byte_limit(self):
        path = self.image()
        with patch.object(safe_io, 'MAX_IMAGE_BYTES', 8), self.assertRaises(ValueError):
            safe_io.load_image(path)

    def test_dataset_indexing_is_sorted_and_bounded(self):
        self.image('images/b/b.png')
        self.image('images/a/z.png')
        self.image('images/a/nested/a.png')
        ds = safe_io.BoundedImageFolder(str(self.root / 'images'))
        self.assertEqual(ds.classes, ['a', 'b'])
        self.assertEqual([Path(path).name for path, _ in ds.samples], ['z.png', 'a.png', 'b.png'])
        for constant, limit in (('MAX_SAMPLES', 2), ('MAX_ENTRIES', 2),
                                ('MAX_DIRECTORY_ENTRIES', 1), ('MAX_DEPTH', 0)):
            with self.subTest(constant=constant), patch.object(safe_io, constant, limit):
                with self.assertRaises(ValueError):
                    safe_io.BoundedImageFolder(str(self.root / 'images'))

    def test_dataset_rejects_symlink_cycles_and_outside_files(self):
        target = self.image()
        self.link(target.parent, target.parent / 'loop', directory=True)
        with self.assertRaises(ValueError):
            safe_io.BoundedImageFolder(str(self.root / 'images'))

    def test_dataset_rechecks_sample_paths_at_load(self):
        target = self.image()
        outside = self.image('outside.png')
        ds = safe_io.BoundedImageFolder(str(self.root / 'images'))
        target.unlink()
        self.link(outside, target)
        with self.assertRaises(ValueError):
            ds[0]

    def test_statistics_match_full_dataset_unbiased_statistics(self):
        data = torch.rand(7, 3, 8, 8, generator=torch.Generator().manual_seed(42))
        loader = DataLoader(TensorDataset(data, torch.zeros(7)), batch_size=2)
        mean, std = safe_io.channel_statistics(loader)
        values = data.permute(1, 0, 2, 3).flatten(1)
        torch.testing.assert_close(mean, values.mean(1))
        torch.testing.assert_close(std, values.std(1))

    def test_statistics_do_not_concatenate_dataset_or_keep_batches(self):
        def batches():
            for _ in range(40):
                yield torch.arange(48).reshape(1, 3, 4, 4).float(), None
        with patch.object(torch, 'cat', side_effect=AssertionError('dataset concatenated')):
            mean, std = safe_io.channel_statistics(batches())
        self.assertEqual(mean.numel(), 3)
        self.assertTrue((std > 0).all())

    def test_empty_and_constant_statistics_have_actionable_errors(self):
        with self.assertRaisesRegex(ValueError, 'nominal pixels'):
            safe_io.channel_statistics([])
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            safe_io.channel_statistics([(torch.zeros(1, 3, 4, 4), None)])

    def test_actual_folder_and_reference_statistics(self):
        path = self.image('custom/train/breast_img/normal/sample.png')
        self.image('custom/test/breast_img/normal/sample.png')
        folder = ADImageFolderDataset(str(self.root), 0, 'none', 0, 'unsupervised',
                                      'gaussian', 1, True)
        self.assertEqual(tuple(folder.train_set[0][0].shape), (3, 224, 224))
        self.refs(path, path)
        (self.root / 'custom' / 'test_ref.csv').write_bytes(
            (self.root / 'custom' / 'train_ref.csv').read_bytes())
        reference = ADImageRefDataset(str(self.root), 0, 'aug1', 0, 'unsupervised',
                                     'gaussian', 1, True)
        torch.testing.assert_close(reference.mean, folder.mean)
        torch.testing.assert_close(reference.std, folder.std)


class MetadataTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.sample = self.image()
        self.dataset_root = self.root / 'images'
        self.meta = self.dataset_root / 'meta.json'

    def test_metadata_creation_and_cache_loading(self):
        first = MyImageFolder(str(self.dataset_root))
        second = MyImageFolder(str(self.dataset_root))
        self.assertEqual(first.samples, second.samples)
        self.assertEqual(first[0][0].size, (8, 8))

    def test_metadata_schema_size_class_and_path_validation(self):
        for value in ({}, [[str(self.sample), True]], [[str(self.sample), 7]],
                      [[str(self.sample)]], [['../outside.png', 0]],
                      [[str(self.root / 'outside.png'), 0]],
                      [[str(self.sample.with_suffix('.txt')), 0]]):
            self.meta.write_text(json.dumps(value), encoding='utf-8')
            with self.subTest(value=value), self.assertRaises(ValueError):
                MyImageFolder(str(self.dataset_root))
        self.meta.write_text(json.dumps([[str(self.sample), 0]]), encoding='utf-8')
        with patch('fcdd.datasets.outlier_exposure.imagenet.MAX_METADATA_BYTES', 4):
            with self.assertRaises(ValueError):
                MyImageFolder(str(self.dataset_root))
        self.meta.write_text(json.dumps([[str(self.sample), 0]] * 2), encoding='utf-8')
        with patch('fcdd.datasets.outlier_exposure.imagenet.MAX_SAMPLES', 1):
            with self.assertRaises(ValueError):
                MyImageFolder(str(self.dataset_root))

    def test_metadata_symlink_never_reads_or_overwrites_target(self):
        target = self.root / 'external.json'
        target.write_text('untouched', encoding='utf-8')
        self.link(target, self.meta)
        with self.assertRaises(ValueError):
            MyImageFolder(str(self.dataset_root))
        self.assertEqual(target.read_text(encoding='utf-8'), 'untouched')

    def test_metadata_sample_symlink_rejected(self):
        link = self.dataset_root / 'class0' / 'link.png'
        self.link(self.sample, link)
        self.meta.write_text(json.dumps([[str(link), 0]]), encoding='utf-8')
        with self.assertRaises(ValueError):
            MyImageFolder(str(self.dataset_root))

    def test_imagenet_requires_explicit_preparation_without_extraction(self):
        archive = self.root / 'ILSVRC2012_img_train.tar'
        archive.write_bytes(b'not an archive')
        with self.assertRaisesRegex(ValueError, 'Automatic archive extraction is disabled'):
            PathsMetaFileImageNet(str(self.root))
        with self.assertRaisesRegex(ValueError, 'Automatic archive extraction is disabled'):
            OEImageNet((1, 3, 8, 8), root=str(self.root), split='train', limit_var=1)
        self.assertFalse((self.root / 'train').exists())

    def test_prepared_imagenet_loads_safe_metadata(self):
        self.image('imagenet/train/n0001/sample.png')
        root = self.root / 'imagenet'
        torch.save(({'n0001': ('example',)}, ['n0001']), root / 'meta.bin')
        ds = PathsMetaFileImageNet(str(root))
        self.assertEqual(ds.classes, [('example',)])
        oe = OEImageNet((1, 3, 8, 8), root=str(self.root), split='train', limit_var=1)
        self.assertEqual(tuple(oe[0].shape), (3, 8, 8))


class GroundTruthTests(TemporaryTest):
    def dataset(self, all_transform=None, nominal=0, tensor_transform=transforms.ToTensor()):
        self.image('train/example/normal/sample.png')
        return ImageFolderDatasetGTM(
            str(self.root / 'train'), 'malformed_normal', (3, 8, 8), False,
            nominal, 1 - nominal, normal_classes=[0], all_transform=all_transform,
            img_gtm_transform=MultiCompose([tensor_transform]),
        )

    def test_missing_supervisor_map_uses_default_mask(self):
        ds = self.dataset(all_transform=lambda image, gt, target, **kwargs: (image, None, target))
        with patch('fcdd.datasets.image_folder_gtms.random.random', return_value=0):
            image, target, gt = ds[0]
        self.assertEqual(tuple(gt.shape), (1, 8, 8))
        self.assertEqual(target, 0)
        self.assertEqual(gt.sum().item(), 0)

    def test_label_swap_works_for_unsigned_and_float_maps(self):
        for dtype in (torch.uint8, torch.float32):
            def image_transform(image):
                return to_tensor(image).to(dtype=dtype)
            def supervisor(image, gt, target, **kwargs):
                mask = torch.zeros(8, 8)
                mask[:, 4:] = 1
                return image, mask, target
            ds = self.dataset(supervisor, nominal=1)
            ds.img_gtm_transform = lambda images: tuple(image_transform(image) for image in images)
            with patch('fcdd.datasets.image_folder_gtms.random.random', return_value=0):
                _, _, gt = ds[0]
            self.assertTrue((gt[:, :, :4] == 1).all())
            self.assertTrue((gt[:, :, 4:] == 0).all())
            self.assertEqual(gt.dtype, dtype)

    def test_original_maps_are_loaded_lazily_and_output_is_bounded(self):
        self.image('train_maps/example/normal/sample.png', shape=(10, 12))
        with patch('fcdd.datasets.image_folder_gtms.load_image', side_effect=AssertionError('eager decoding')):
            ds = self.dataset()
        with patch('fcdd.datasets.image_folder_gtms.MAX_TENSOR_BYTES', 8), self.assertRaises(ValueError):
            ds.get_original_gtmaps_normal_class()
        result = ds.get_original_gtmaps_normal_class()
        self.assertEqual(tuple(result.shape), (1, 1, 10, 10))

    def test_noise_supervisor_map_is_not_discarded(self):
        def supervisor(image, gt, target, **kwargs):
            return torch.zeros(3, 8, 8, dtype=torch.uint8), torch.ones(8, 8), target
        ds = self.dataset(supervisor)
        ds.supervise_mode = 'noise'
        with patch('fcdd.datasets.image_folder_gtms.random.random', return_value=0):
            _, _, gt = ds[0]
        self.assertTrue((gt == 1).all())


class LoggerTests(TemporaryTest):
    def test_all_artifact_paths_reject_traversal_and_external_names(self):
        logger = Logger(str(self.root / 'logs'))
        for subdir in ('../escape', str(self.root / 'outside')):
            for action in (lambda: logger.save(subdir),
                           lambda: logger.single_save('scores', {}, subdir),
                           lambda: logger.single_save('tensor', torch.zeros(1), subdir),
                           lambda: logger.imsave('image', torch.zeros(1, 3, 8, 8), subdir)):
                with self.subTest(subdir=subdir, action=action), self.assertRaises(ValueError):
                    action()
        for name in ('../escape', '..\\escape', '/absolute', 'file:stream'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                logger.single_save(name, {})
        self.assertFalse((self.root / 'escape').exists())

    def test_logger_rejects_symlink_outputs(self):
        root = self.root / 'logs'
        root.mkdir()
        target = self.root / 'outside.json'
        target.write_text('original', encoding='utf-8')
        self.link(target, root / 'scores.json')
        with self.assertRaises(ValueError):
            Logger(str(root)).single_save('scores', {'ok': True})
        self.assertEqual(target.read_text(encoding='utf-8'), 'original')

    def test_log_buffer_and_disk_are_byte_bounded(self):
        logger = Logger(str(self.root))
        with patch('fcdd.util.logging.MAX_LOG_BYTES', 5):
            logger.logtxt('ab')
            with self.assertRaises(ValueError):
                logger.logtxt('cd')
        path = self.root / 'limited.log'
        safety.append_text(str(path), 'abc', maximum=5)
        safety.append_text(str(path), '\u00e9', maximum=5)
        with self.assertRaises(ValueError):
            safety.append_text(str(path), 'x', maximum=5)
        self.assertEqual(path.stat().st_size, 5)

    def test_logtxt_file_quota_applies_after_save(self):
        logger = Logger(str(self.root))
        logger.save()
        with open(logger.logtxtfile, 'wb') as stream:
            stream.truncate(safety.MAX_LOG_BYTES)
        with self.assertRaises(ValueError):
            logger.logtxt('extra')
        self.assertEqual(Path(logger.logtxtfile).stat().st_size, safety.MAX_LOG_BYTES)

    def test_history_and_nested_numpy_json_are_bounded_without_destroying_existing_output(self):
        logger = Logger(str(self.root))
        logger.single_save('scores', {'score': np.array([0.1, 0.2])})
        path = self.root / 'scores.json'
        original = path.read_bytes()
        for large in ({'nested': {'data': 'x' * (safety.MAX_JSON_BYTES + 1)}},
                      {'array': np.zeros(safety.MAX_JSON_BYTES // 32 + 1)}):
            with self.assertRaises(ValueError):
                logger.single_save('scores', large)
            self.assertEqual(path.read_bytes(), original)
        logger.history['oversized'] = 'x' * (safety.MAX_JSON_BYTES + 1)
        with self.assertRaises(ValueError):
            logger.save()
        self.assertFalse(list(self.root.glob('.fcdd-*')))

    def test_actual_encoded_json_bytes_are_measured(self):
        path = self.root / 'data.json'
        path.write_text('original', encoding='utf-8')
        with self.assertRaises(ValueError):
            safety.bounded_json(str(path), {'escaped': '\0' * 20}, maximum=100)
        self.assertEqual(path.read_text(encoding='utf-8'), 'original')
        self.assertFalse(list(self.root.glob('.fcdd-*')))

    def test_large_numpy_string_elements_are_checked_before_conversion(self):
        path = self.root / 'array.json'
        with self.assertRaises(ValueError):
            safety.bounded_json(str(path), {'text': np.zeros(1, dtype='U1000')}, maximum=1000)
        self.assertFalse(path.exists())

    def test_normal_image_and_plot_outputs_are_written(self):
        logger = Logger(str(self.root))
        logger.imsave('images', torch.rand(2, 3, 8, 8), subdir='nested')
        logger.imsave('annotated', torch.rand(2, 3, 32, 32), subdir='nested',
                      rowheaders=['input'], colcounter=['1', '2'], row_sep_at=(2, 16))
        logger.single_plot('curve', [0, 1], subdir='nested')
        self.assertTrue((self.root / 'nested' / 'images.png').is_file())
        with Image.open(self.root / 'nested' / 'annotated.png') as image:
            self.assertEqual(np.asarray(image).dtype, np.uint8)
        self.assertTrue((self.root / 'nested' / 'curve.pdf').is_file())

    def test_tensor_artifact_budget_and_normal_save(self):
        logger = Logger(str(self.root))
        logger.single_save('tensor', torch.arange(8))
        torch.testing.assert_close(torch.load(self.root / 'tensor.pth', weights_only=True), torch.arange(8))
        with patch('fcdd.util.logging.MAX_TENSOR_BYTES', 1):
            with self.assertRaises((ValueError, RuntimeError)):
                logger.single_save('large', torch.arange(8))
        self.assertFalse((self.root / 'large.pth').exists())

    def test_normal_logging_preserves_data_across_saves(self):
        logger = Logger(str(self.root))
        logger.logtxt('before')
        logger.log(0, 0, 1, torch.tensor(1.), force_print=True)
        logger.save('nested')
        logger.logtxt('after')
        logger.save('nested')
        text = (self.root / 'nested' / 'log.txt').read_text()
        self.assertIn('before', text)
        self.assertIn('after', text)
        self.assertIn('err', json.loads((self.root / 'nested' / 'history.json').read_text()))
        with self.assertRaises(ValueError):
            logger.log(safety.MAX_EPOCHS + 1, 0, 1, torch.tensor(1.))


class RunnerTests(TemporaryTest):
    def test_configuration_is_bounded_and_handles_real_json(self):
        value = {'datadir': r'C:\data\my,images', 'quantile': .97, 'blur_heatmaps': False}
        path = self.root / 'config.txt'
        path.write_text('Model()\n\n' + json.dumps(value) + '\nnotes', encoding='utf-8')
        self.assertEqual(load_config(str(self.root)), value)
        with patch('fcdd.util.io.MAX_CONFIG_BYTES', 8), self.assertRaises(ValueError):
            read_cfg(str(path))

    def test_config_cannot_override_trusted_output_root_or_resource_limits(self):
        args = Namespace(logdir=str(self.root / 'logs'))
        with self.assertRaises(ValueError):
            extract_args(args, {'logdir': '../outside'})
        with self.assertRaises(ValueError):
            extract_args(args, {'logdir': str(self.root / 'outside')})
        with self.assertRaises(ValueError):
            extract_args(args, {'workers': safety.MAX_WORKERS + 1})

    def test_configuration_restore_keeps_the_explicit_root(self):
        from argparse import ArgumentParser
        args = DefaultConfig()(ArgumentParser()).parse_args([])
        args.logdir = str(self.root / 'logs')
        config = vars(args).copy()
        config['normal_class'] = 0
        restored = extract_args(args, config)
        self.assertEqual(restored.logdir, config['logdir'])
        self.assertEqual(restored.batch_size, config['batch_size'])

    def test_visualization_log_paths_and_record_limits(self):
        path = self.root / 'normal_0' / 'it_0' / 'log.txt'
        path.parent.mkdir(parents=True)
        path.write_text(
            'Interpretation visualization paper image heatmaps label 0: [0, 2]\n'
            'Interpretation visualization paper image heatmaps label 1: [1, 3]\n', encoding='utf-8')
        self.assertEqual(extract_viz_ids(str(self.root), 0, 0), [[0, 2], [1, 3]])
        for cls, iteration in (('../outside', 0), (0, '../outside'), (-1, 0)):
            with self.assertRaises(ValueError):
                extract_viz_ids(str(self.root), cls, iteration)
        with patch('fcdd.runners.bases.MAX_LOG_BYTES', 8), self.assertRaises(ValueError):
            extract_viz_ids(str(self.root), 0, 0)
        path.write_text('x' * (64 * 1024 + 1), encoding='utf-8')
        with self.assertRaises(ValueError):
            extract_viz_ids(str(self.root), 0, 0)
        path.write_text('Interpretation visualization paper image label 0: [-1]\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            extract_viz_ids(str(self.root), 0, 0)

    def test_class_and_seed_restrictions_validate_before_training(self):
        runner = ClassesRunner.__new__(ClassesRunner)
        runner.start = 0
        runner.run_seeds = Mock(return_value={})
        for classes in (['../outside'], [True], [-1], [30], []):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                runner.run_classes(it=1, dataset='imagenet', logdir=str(self.root), cls_restrictions=classes)
        runner.run_seeds.assert_not_called()
        runner.run_classes(it=1, dataset='imagenet', logdir=str(self.root), cls_restrictions=[0])
        self.assertEqual(runner.run_seeds.call_args.kwargs['logdir'], str(self.root / 'normal_0'))
        seeds = SeedsRunner.__new__(SeedsRunner)
        seeds.start = 0
        seeds.run_one = Mock(return_value={})
        with self.assertRaises(ValueError):
            seeds.run_seeds(it=2, its_restrictions=['../outside'], logdir=str(self.root), viz_ids=None)
        seeds.run_one.assert_not_called()

    def test_visualization_indices_are_global_and_label_matched(self):
        trainer = SimpleNamespace(
            logger=Mock(), resdown=8, reduce_ascore=lambda x: x.flatten(1).mean(1),
            _create_heatmaps_picture=Mock(), _create_singlerow_heatmaps_picture=Mock(),
        )
        arguments = dict(labels=[0, 1, 0, 1], ascores=torch.ones(4, 1, 2, 2),
                         imgs=torch.ones(4, 3, 2, 2), show_per_cls=6)
        for indices in (([4], [1]), ([-1], [1]), ([True], [1]), ([1], [0]), ([0],), ('bad', [1])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                BaseADTrainer.heatmap_generation(trainer, **arguments, specific_idx=indices)
        trainer._create_singlerow_heatmaps_picture.assert_not_called()
        BaseADTrainer.heatmap_generation(trainer, **arguments, specific_idx=([2], [3]))
        self.assertEqual(trainer._create_singlerow_heatmaps_picture.call_count, 4)


class CheckpointTests(TemporaryTest):
    def trainer(self):
        return SimpleNamespace(device='cpu', net=torch.nn.Linear(2, 1), opt=None, sched=None)

    def test_safe_checkpoint_round_trip(self):
        trainer = self.trainer()
        original = {key: value.clone() for key, value in trainer.net.state_dict().items()}
        path = self.root / 'snapshot.pt'
        torch.save({'net': original, 'epoch': 4}, path)
        self.assertEqual(BaseTrainer.load(trainer, str(path)), 4)
        for key, value in original.items():
            torch.testing.assert_close(trainer.net.state_dict()[key], value)

    def test_unsupported_weights_only_never_falls_back(self):
        path = self.root / 'snapshot.pt'
        path.write_bytes(b'not a checkpoint')
        with patch('fcdd.training.bases.torch.load', side_effect=TypeError('unsupported')) as loader:
            with self.assertRaises(TypeError):
                BaseTrainer.load(self.trainer(), str(path))
        self.assertEqual(loader.call_count, 1)
        self.assertTrue(loader.call_args.kwargs['weights_only'])

    def test_untrusted_pickle_is_not_executed(self):
        marker = self.root / 'should-not-exist'

        class Payload:
            def __reduce__(self):
                return os.mkdir, (str(marker),)

        path = self.root / 'snapshot.pt'
        torch.save({'unexpected': Payload()}, path)
        with self.assertRaises(pickle.UnpicklingError):
            BaseTrainer.load(self.trainer(), str(path))
        self.assertFalse(marker.exists())

    def test_oversized_checkpoint_rejected_before_deserialization(self):
        path = self.root / 'snapshot.pt'
        path.write_bytes(b'12345')
        with patch('fcdd.training.bases.MAX_CHECKPOINT_BYTES', 4):
            with patch('fcdd.training.bases.torch.load') as load:
                with self.assertRaises(ValueError):
                    BaseTrainer.load(self.trainer(), str(path))
                load.assert_not_called()

    def test_cached_weights_are_verified_before_restricted_loading(self):
        path = self.root / 'vgg11_bn-6002323d.pth'
        state = {'weight': torch.ones(2)}
        torch.save(state, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(weights, 'VGG11_BN_SHA256', digest):
            loaded = weights.load_vgg11_bn_weights(str(self.root))
            torch.testing.assert_close(loaded['weight'], state['weight'])
            path.write_bytes(b'poisoned')
            with patch.object(weights.torch, 'load') as loader:
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    weights.load_vgg11_bn_weights(str(self.root))
                loader.assert_not_called()

    def test_download_hash_mismatch_leaves_no_cache(self):
        with patch.object(weights.urllib.request, 'urlopen', return_value=io.BytesIO(b'poisoned')):
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                weights.load_vgg11_bn_weights(str(self.root))
        self.assertFalse((self.root / 'vgg11_bn-6002323d.pth').exists())
        self.assertFalse(list(self.root.glob('.fcdd-*')))

    def test_valid_download_is_checked_cached_and_loaded(self):
        stream = io.BytesIO()
        torch.save({'weight': torch.ones(2)}, stream)
        data = stream.getvalue()
        with patch.object(weights, 'VGG11_BN_SHA256', hashlib.sha256(data).hexdigest()):
            with patch.object(weights.urllib.request, 'urlopen', return_value=io.BytesIO(data)) as download:
                result = weights.load_vgg11_bn_weights(str(self.root))
            download.assert_called_once_with(weights.VGG11_BN_URL, timeout=60)
        torch.testing.assert_close(result['weight'], torch.ones(2))


if __name__ == '__main__':
    unittest.main()
