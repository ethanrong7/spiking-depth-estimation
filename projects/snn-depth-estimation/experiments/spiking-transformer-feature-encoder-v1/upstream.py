"""Fetch immutable author sources; no source rewriting is required on the server."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

EXP = Path(__file__).resolve().parent
SOURCES = json.loads((EXP / 'upstream-sources.json').read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_sources(cache, offline=False):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for name, record in SOURCES.items():
        files = [(name + '.py', record['url'], record['sha256'])]
        if record['license_path']:
            repo = record['repository'].replace('https://github.com/', '')
            url = f"https://raw.githubusercontent.com/{repo}/{record['commit']}/{record['license_path']}"
            files.append((name + '.LICENSE', url, record['license_sha256']))
        for filename, url, expected in files:
            path = cache / filename
            if not path.exists():
                if offline:
                    raise FileNotFoundError(f'Missing pinned source: {path}; prepare with network access first')
                value = urllib.request.urlopen(url, timeout=60).read()
                if hashlib.sha256(value).hexdigest() != expected:
                    raise RuntimeError(f'Upstream checksum mismatch: {url}')
                tmp = path.with_suffix(path.suffix + '.tmp')
                tmp.write_bytes(value)
                tmp.replace(path)
            if digest(path) != expected:
                raise RuntimeError(f'Pinned source was modified: {path}')
    return cache


def load_definitions(name, cache, neuron_class=None):
    """Evaluate exact author class/function bodies with explicit torch dependencies.

    Omit module import/training-demo side effects (CuPy, timm registration,
    CUDA extension imports). The only neuron replacement is the tested torch
    implementation. Attention equations are unchanged. QKFormer receives the documented
    final-projection shape correction so its neuron sees T,B rather than T*B.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F
    from functools import partial
    from neuron import MultiStepLIFNode, DropPath
    path = Path(cache) / (name + '.py')
    if digest(path) != SOURCES[name]['sha256']:
        raise RuntimeError(f'Pinned source mismatch: {path}')
    source = path.read_text()
    if name == 'qkformer':
        old = 'x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,W,H)'
        new = 'x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W))'
        if source.count(old) != 1:
            raise RuntimeError('Pinned QKFormer shape-correction anchor changed')
        source = source.replace(old, new)
    tree = ast.parse(source, filename=str(path))
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    namespace = dict(__name__='encoder_author_' + name.replace('-', '_'), torch=torch,
                     nn=nn, F=F, partial=partial, DropPath=DropPath,
                     MultiStepLIFNode=neuron_class or MultiStepLIFNode,
                     to_2tuple=lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x, x),
                     trunc_normal_=nn.init.trunc_normal_, register_model=lambda f: f,
                     _cfg=lambda: {}, WindowProcess=None, WindowProcessReverse=None)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace
