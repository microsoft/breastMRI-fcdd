"""Input and artifact limits shared by dataset, runner, and logging entry points."""

import math
import ntpath
import os
import stat
import tempfile
from contextlib import contextmanager
from numbers import Integral

MAX_SAMPLES = 2_000_000
MAX_ENTRIES = 4_000_000
MAX_DIRECTORY_ENTRIES = 100_000
MAX_DEPTH = 32
MAX_METADATA_BYTES = 256 * 1024 * 1024
MAX_CSV_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_LOG_BYTES = 10 * 1024 * 1024
MAX_JSON_BYTES = 10_000_000
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_777_216
MAX_TENSOR_BYTES = 256 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
MAX_EPOCHS = 10_000
MAX_BATCH_SIZE = 1024
MAX_WORKERS = 32
DEFAULT_OE_LIMIT = 100_000


def bounded_int(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return int(value)


def quantile_value(value):
    if isinstance(value, bool):
        raise ValueError('quantile must be finite and in [0, 1]')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('quantile must be finite and in [0, 1]') from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('quantile must be finite and in [0, 1]')
    return result


def validate_resources(config):
    for name, low, high in (
        ('batch_size', 1, MAX_BATCH_SIZE), ('epochs', 0, MAX_EPOCHS),
        ('workers', 0, MAX_WORKERS), ('acc_batches', 1, MAX_BATCH_SIZE),
        ('resdown', 1, 4096), ('oe_limit', 1, MAX_SAMPLES),
    ):
        if name in config:
            bounded_int(config[name], name, low, high)
    if config.get('batch_size', 1) * config.get('acc_batches', 1) > MAX_BATCH_SIZE:
        raise ValueError(f'accumulated batch size must not exceed {MAX_BATCH_SIZE}')
    if 'quantile' in config:
        quantile_value(config['quantile'])


def canonical_root_path(path):
    """Normalize a caller-selected filesystem root; this does not establish trust."""
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
        raise ValueError('An explicit nonempty root directory is required')
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def is_link(path):
    return os.path.islink(path) or os.path.isjunction(path)


def confined_path(root, *parts):
    """Resolve a descendant, rejecting traversal, links, and foreign drive syntax."""
    root = canonical_root_path(root)
    path = root
    for part in parts:
        part = os.fspath(part)
        if not isinstance(part, str) or '\0' in part:
            raise ValueError('Invalid path component')
        if '..' in part.replace('\\', '/').split('/'):
            raise ValueError('Parent traversal is not allowed')
        if (ntpath.isabs(part) or ntpath.splitdrive(part)[0]) and not os.path.isabs(part):
            raise ValueError('Foreign absolute paths are not allowed')
        if ':' in part[len(os.path.splitdrive(part)[0]):]:
            raise ValueError('Alternate data streams are not allowed')
        path = os.path.abspath(os.path.join(path, part))
        try:
            inside = os.path.commonpath((root, path)) == root
        except ValueError:
            inside = False
        if not inside:
            raise ValueError('Path must remain inside its configured root')
    relative = os.path.relpath(path, root)
    current = root
    for component in relative.split(os.sep):
        current = os.path.join(current, component)
        if is_link(current):
            raise ValueError('Symbolic links and junctions are not allowed below the configured root')
    if os.path.commonpath((root, os.path.realpath(path))) != root:
        raise ValueError('Path resolves outside its configured root')
    return path


def filename(value):
    if not isinstance(value, str) or value in ('', '.', '..') or any(c in value for c in '/\\:\0'):
        raise ValueError('Artifact names must be single nonempty filename components')
    return value


@contextmanager
def bounded_reader(path, maximum, text=False):
    flags = (os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
             | getattr(os, 'O_NONBLOCK', 0))
    if is_link(path):
        raise ValueError('Refusing to read a symbolic link or junction')
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError(f'Input must be a regular file no larger than {maximum} bytes')
        if text:
            data = stream.read(maximum + 1)
            if len(data) > maximum:
                raise ValueError(f'Input exceeds {maximum} bytes')
            yield data.decode('utf-8-sig')
        else:
            yield stream


class LimitedWriter:
    def __init__(self, stream, maximum):
        self.stream, self.maximum = stream, maximum

    def write(self, data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        if self.stream.tell() + len(data) > self.maximum:
            raise ValueError(f'Artifact exceeds {self.maximum} bytes')
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def atomic_writer(path, maximum):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_link(path):
        raise ValueError('Refusing to overwrite a symbolic link')
    fd, temporary = tempfile.mkstemp(prefix='.fcdd-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w+b') as stream:
            yield LimitedWriter(stream, maximum)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bounded_json(path, value, maximum=MAX_JSON_BYTES):
    import numpy as np
    from fcdd.util import NumpyEncoder

    remaining = maximum
    active = set()

    def check(item, depth=0):
        nonlocal remaining
        if depth > 64:
            raise ValueError('JSON nesting exceeds 64 levels')
        remaining -= 1
        if isinstance(item, str):
            if len(item) > remaining:
                raise ValueError('Artifact string exceeds the byte budget')
            remaining -= len(item.encode('utf-8'))
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError('Object arrays cannot be serialized as artifacts')
            # Bound tolist() expansion before the encoder allocates Python objects.
            remaining -= max(item.nbytes, item.size * 32)
        elif isinstance(item, (dict, list, tuple)):
            if id(item) in active:
                raise ValueError('Circular artifact structure')
            if len(item) > remaining:
                raise ValueError('Artifact contains too many entries')
            active.add(id(item))
            for child in item:
                check(child, depth + 1)
                if isinstance(item, dict):
                    check(item[child], depth + 1)
            active.remove(id(item))
        if remaining < 0:
            raise ValueError(f'Artifact structure exceeds the {maximum}-byte budget')

    check(value)
    with atomic_writer(path, maximum) as writer:
        for chunk in NumpyEncoder().iterencode(value):
            writer.write(chunk)


def append_text(path, text, maximum=MAX_LOG_BYTES):
    if len(text) > maximum:
        raise ValueError(f'Log entry exceeds {maximum} bytes')
    data = text.encode('utf-8')
    if len(data) > maximum:
        raise ValueError(f'Log entry exceeds {maximum} bytes')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_link(path):
        raise ValueError('Refusing to append to a symbolic link')
    flags = (os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, 'O_BINARY', 0)
             | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(os.open(path, flags, 0o600), 'ab') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size + len(data) > maximum:
            raise ValueError(f'Log file exceeds {maximum} bytes')
        stream.write(data)


def validate_image_shape(size):
    if len(size) != 4:
        raise ValueError('Image batches must have NCHW shape')
    n, c, h, w = size
    bounded_int(n, 'image batch size', 1, MAX_BATCH_SIZE)
    bounded_int(c, 'image channels', 1, 3)
    bounded_int(h, 'image height', 1, 4096)
    bounded_int(w, 'image width', 1, 4096)
    if c not in (1, 3) or n * c * h * w * 4 > MAX_TENSOR_BYTES:
        raise ValueError('Requested image batch exceeds the tensor budget')


def prepared_imagenet(root, split):
    meta = confined_path(root, 'meta.bin')
    folder = confined_path(root, split)
    if not os.path.isfile(meta) or not os.path.isdir(folder):
        raise ValueError(
            'ImageNet must be prepared explicitly from trusted archives: provide meta.bin and '
            f'the extracted {split} directory. Automatic archive extraction is disabled.'
        )
    with bounded_reader(meta, MAX_METADATA_BYTES):
        pass
    return meta
