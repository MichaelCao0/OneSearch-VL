"""Optional filesystem publisher implementing the existing upload_cos contract.

An operator supplies ASSET_DIR and the matching externally reachable
ASSET_BASE_URL. This module does not start a server or configure cloud storage.
"""
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.parse import quote, urlparse


def upload_cos(local_path, filename=None, date=None, mode=None, userid=None, **kwargs):
    root, base = os.getenv('ASSET_DIR', ''), os.getenv('ASSET_BASE_URL', '')
    if not root or not base:
        raise RuntimeError('Set ASSET_DIR and ASSET_BASE_URL to publish tool-generated media.')
    parsed = urlparse(base)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('ASSET_BASE_URL must be an HTTP(S) URL.')
    source = Path(local_path)
    content = source.read_bytes()
    suffix = source.suffix.lower()
    if suffix not in {'.png', '.jpg', '.jpeg', '.webp', '.mp4'}:
        raise ValueError(f'Unsupported media extension: {suffix}')
    key = hashlib.sha256(content).hexdigest() + suffix
    target = Path(root).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.chmod(0o644)
        temporary.replace(target / key)
    finally:
        temporary.unlink(missing_ok=True)
    return key, base.rstrip('/') + '/' + quote(key)
