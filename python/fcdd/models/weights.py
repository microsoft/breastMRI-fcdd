"""Integrity-checked loading of the published VGG11-BN checkpoint."""

import hashlib
import os
import urllib.request

import torch

from fcdd.util.safety import atomic_writer, bounded_reader, confined_path

VGG11_BN_URL = 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth'
VGG11_BN_SHA256 = '6002323d9413ae02f657473d3ecbbd7f86ee496b8a0dc8dc6140a639a0fcf13d'
VGG11_BN_MAX_BYTES = 600 * 1024 * 1024


def load_vgg11_bn_weights(model_dir=None):
    model_dir = model_dir or os.path.join(torch.hub.get_dir(), 'checkpoints')
    path = confined_path(model_dir, 'vgg11_bn-6002323d.pth')
    if not os.path.exists(path):
        with urllib.request.urlopen(VGG11_BN_URL, timeout=60) as response:
            with atomic_writer(path, VGG11_BN_MAX_BYTES) as writer:
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    writer.write(chunk)
                if digest.hexdigest() != VGG11_BN_SHA256:
                    raise ValueError('Downloaded VGG11-BN weights failed SHA-256 verification')
    # Check cache hits as well, and deserialize from the same verified file handle.
    with bounded_reader(path, VGG11_BN_MAX_BYTES) as reader:
        if hashlib.file_digest(reader, 'sha256').hexdigest() != VGG11_BN_SHA256:
            raise ValueError('Cached VGG11-BN weights failed SHA-256 verification; remove the cache file')
        reader.seek(0)
        return torch.load(reader, map_location='cpu', weights_only=True)
