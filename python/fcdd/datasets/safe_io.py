"""Bounded, confined image ingestion and streaming dataset statistics."""

import os
import warnings
from functools import partial

import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import has_file_allowed_extension

from fcdd.util.safety import (
    MAX_DEPTH, MAX_DIRECTORY_ENTRIES, MAX_ENTRIES, MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS, MAX_SAMPLES, bounded_reader, confined_path, is_link, trusted_root,
)


def directory_entries(path):
    entries = []
    with os.scandir(path) as iterator:
        for entry in iterator:
            if len(entries) >= MAX_DIRECTORY_ENTRIES:
                raise ValueError('Dataset directory contains too many entries')
            if is_link(entry.path):
                raise ValueError('Dataset links and junctions are not allowed')
            entries.append(entry)
    return sorted(entries, key=lambda entry: entry.name)


def find_classes(root):
    classes = [entry.name for entry in directory_entries(root) if entry.is_dir(follow_symlinks=False)]
    if not classes:
        raise FileNotFoundError(f'No class folders found in {root}')
    return classes, {name: i for i, name in enumerate(classes)}


def make_dataset(root, class_to_idx, extensions=None, is_valid_file=None, allow_empty=False):
    root = trusted_root(root)
    if (extensions is None) == (is_valid_file is None):
        raise ValueError('Specify exactly one of extensions or is_valid_file')
    if is_valid_file is None:
        is_valid_file = partial(has_file_allowed_extension, extensions=extensions)
    samples, available = [], set()
    visited = 0
    for name in sorted(class_to_idx):
        stack = [(confined_path(root, name), 0)]
        while stack:
            folder, depth = stack.pop()
            if depth > MAX_DEPTH:
                raise ValueError('Dataset directory nesting is too deep')
            directories = []
            for entry in directory_entries(folder):
                visited += 1
                if visited > MAX_ENTRIES:
                    raise ValueError('Dataset traversal exceeds the entry budget')
                path = confined_path(root, entry.path)
                if entry.is_dir(follow_symlinks=False):
                    directories.append((path, depth + 1))
                elif entry.is_file(follow_symlinks=False) and is_valid_file(path):
                    if len(samples) >= MAX_SAMPLES:
                        raise ValueError('Dataset exceeds the sample budget')
                    samples.append((path, class_to_idx[name]))
                    available.add(name)
            stack.extend(reversed(directories))
    if not allow_empty and set(class_to_idx) - available:
        raise FileNotFoundError('One or more dataset classes contain no supported images')
    return samples


def image_size(path, root=None):
    path = confined_path(root, path) if root is not None else path
    with bounded_reader(path, MAX_IMAGE_BYTES) as stream, warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(stream) as image:
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f'Image exceeds {MAX_IMAGE_PIXELS} pixels')
            return width, height


def load_image(path, root=None):
    path = confined_path(root, path) if root is not None else path
    with bounded_reader(path, MAX_IMAGE_BYTES) as stream, warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(stream) as image:
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f'Image exceeds {MAX_IMAGE_PIXELS} pixels')
            return image.convert('RGB')


class BoundedImageFolder(ImageFolder):
    find_classes = staticmethod(find_classes)
    make_dataset = staticmethod(make_dataset)

    def __init__(self, root, **kwargs):
        root = trusted_root(root)
        super().__init__(root, loader=partial(load_image, root=root), **kwargs)


def channel_statistics(loader):
    count, samples, mean, m2 = 0, 0, None, None
    for batch in loader:
        x = batch[0].to(dtype=torch.float64)
        samples += x.shape[0]
        if samples > MAX_SAMPLES:
            raise ValueError('Statistics exceed the dataset sample budget')
        values = x.permute(1, 0, 2, 3).flatten(1)
        n = values.shape[1]
        variance, batch_mean = torch.var_mean(values, dim=1, correction=0)
        if mean is None:
            mean, m2 = batch_mean, variance * n
        else:
            delta = batch_mean - mean
            m2 += variance * n + delta.square() * count * n / (count + n)
            mean += delta * n / (count + n)
        count += n
    if count < 2:
        raise ValueError('At least two nominal pixels are required to compute dataset statistics')
    std = (m2 / (count - 1)).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std == 0).any():
        raise ValueError('Nominal images must have finite values and nonzero per-channel variance')
    return mean.float(), std.float()
