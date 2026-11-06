"""Hash-pinned historical recurrent models behind the study tensor contract.

Original module bytes are loaded directly from an immutable source capsule,
not copied into this tree. Absolute ``core`` imports are remapped in the AST
to a private package. Both original and transformed-source hashes are logged.
"""
import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import linecache
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import torch
from torch import nn


BASE_ID = 'cb43497fdc02bf612794dd48d516ee71e5658f10969e47ed6f4683591f3e8ec5'
ARCHIVE_SHA = 'dde27ebd4d8559c07f558a404d80cadb2f02f6f36bad962318de6dc86de5aa37'
MECHANISM_SHA = 'f8a4147f682a4794ee7f814defcb3684d14764958d61b6606e527de9865f9ae0'
CORE_PREFIX = 'projects/snn-depth-estimation/core/'
MECHANISM_MEMBER = 'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/campaigns/alif-mechanism-screen-v1/scripts/mechanism.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


class _PrivateImports(ast.NodeTransformer):
    def __init__(self, package):
        self.package = package

    def visit_ImportFrom(self, node):
        if node.module == 'core' or (node.module and node.module.startswith('core.')):
            node.module = self.package + node.module[4:]
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == 'core' or alias.name.startswith('core.'):
                if alias.asname is None:
                    raise ImportError('unaliased absolute core import requires explicit audit')
                alias.name = self.package + alias.name[4:]
        return node


class _CapsuleModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, capsule):
        self.capsule = Path(capsule).resolve()
        registry_bytes = (self.capsule / 'source-registry.json').read_bytes()
        self.registry_bytes_sha256 = sha(registry_bytes)
        self.registry = json.loads(registry_bytes)
        if self.registry['source_identity_sha256'] != BASE_ID:
            raise RuntimeError('wrong historical baseline source identity')
        archive = (self.capsule / 'source.tar.gz').read_bytes()
        if sha(archive) != ARCHIVE_SHA or self.registry['archive_sha256'] != ARCHIVE_SHA:
            raise RuntimeError('historical baseline archive hash mismatch')
        source_files = self.registry['source_files']
        if sha(json.dumps(source_files, sort_keys=True).encode()) != BASE_ID:
            raise RuntimeError('historical registry member map identity mismatch')
        self.package = '_study_baseline_' + BASE_ID[:16]
        self.sources, self.provenance = {}, {}
        with tarfile.open(self.capsule / 'source.tar.gz') as stream:
            for member, expected in source_files.items():
                if member.startswith(CORE_PREFIX) or member == MECHANISM_MEMBER:
                    raw = stream.extractfile(member).read()
                    if sha(raw) != expected:
                        raise RuntimeError('historical source member mismatch: ' + member)
                    self.sources[member] = raw

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.package or fullname.startswith(self.package + '.'):
            relative = fullname[len(self.package):].lstrip('.').replace('.', '/')
            package_member = CORE_PREFIX + (relative + '/' if relative else '') + '__init__.py'
            module_member = CORE_PREFIX + relative + '.py'
            is_package = package_member in self.sources
            member = package_member if is_package else module_member
            if member not in self.sources:
                raise ImportError('unregistered private baseline dependency: ' + fullname)
            return importlib.util.spec_from_loader(fullname, self, is_package=is_package)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = module.__name__[len(self.package):].lstrip('.').replace('.', '/')
        package_member = CORE_PREFIX + (relative + '/' if relative else '') + '__init__.py'
        member = package_member if package_member in self.sources else CORE_PREFIX + relative + '.py'
        raw = self.sources[member]
        tree = _PrivateImports(self.package).visit(ast.parse(raw, filename=member))
        ast.fix_missing_locations(tree)
        transformed = ast.unparse(tree).encode()
        self.provenance[member] = {'source_sha256': sha(raw), 'transformed_source_sha256': sha(transformed),
                                   'transformation': 'AST absolute core imports remapped to isolated package only'}
        module.__file__ = str(self.capsule / ('source.tar.gz::' + member))
        # TorchScript's original ALIF reset reads inspect/linecache source.
        # AST remapping preserves original line locations; reset code itself
        # is unchanged, so expose verified original bytes at this virtual path.
        linecache.cache[module.__file__] = (len(raw), None, raw.decode().splitlines(True), module.__file__)
        exec(compile(tree, module.__file__, 'exec'), module.__dict__)

    def load_mechanism(self):
        raw = self.sources[MECHANISM_MEMBER]
        if sha(raw) != MECHANISM_SHA:
            raise RuntimeError('wrong independent-current adapter')
        namespace = {'__name__': self.package + '._mechanism'}
        exec(compile(raw, MECHANISM_MEMBER, 'exec'), namespace)
        self.provenance[MECHANISM_MEMBER] = {'source_sha256': sha(raw), 'transformed_source_sha256': sha(raw), 'transformation': 'none'}
        return namespace['adapt']


_CAPSULE_LOADERS = {}


def load_pinned_core(capsule):
    path = str(Path(capsule).resolve())
    if path not in _CAPSULE_LOADERS:
        loader = _CapsuleModules(path)
        # One immutable namespace may be shared only after verifying that a
        # second requested capsule contains exactly the same pinned bytes.
        existing = next((x for x in _CAPSULE_LOADERS.values() if x.package == loader.package), None)
        if existing is not None:
            loader = existing
        else:
            sys.meta_path.insert(0, loader)
        _CAPSULE_LOADERS[path] = loader
    loader = _CAPSULE_LOADERS[path]
    module = importlib.import_module(loader.package + '.spikestereonet')
    utils = importlib.import_module(loader.package + '.utils.utils')
    return module.SpikeStereoNet, utils.InputPadder, loader


def flatten_sensor_channels(x):
    """Time-major ordering t0ON,t0OFF,t1ON,t1OFF; no polarity cancellation."""
    if x.ndim != 5 or x.shape[1] != 50 or x.shape[2] not in (1, 2):
        raise ValueError('legacy baseline requires B,50,C,H,W with C=1 or2')
    return x.flatten(1, 2)


def expand_polarity_stem(stem):
    """100 independently trainable input channels, equal ON/OFF initialization."""
    if not isinstance(stem, nn.Conv2d) or stem.in_channels != 50 or stem.groups != 1:
        raise ValueError('expected the registered50-channel first convolution')
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Conv2d(100, stem.out_channels, stem.kernel_size, stem.stride, stem.padding,
                             stem.dilation, stem.groups, stem.bias is not None, stem.padding_mode)
    expanded = expanded.to(device=stem.weight.device, dtype=stem.weight.dtype)
    with torch.no_grad():
        expanded.weight.copy_(stem.weight.repeat_interleave(2, dim=1) / (2 ** .5))
        if stem.bias is not None:
            expanded.bias.copy_(stem.bias)
    return expanded


def legacy_flow_to_disparity(flow):
    """Legacy coords1 indexes the right view: flow=x_right-x_left."""
    return -flow


class StudyBaseline(nn.Module):
    def __init__(self, capsule, architecture='original', input_channels=1, seed=1234,
                 train_refinements=16, eval_refinements=32):
        super().__init__()
        if architecture not in ('original', 'original_I') or input_channels not in (1, 2):
            raise ValueError('unknown baseline architecture or sensor channels')
        model_class, padder, loader = load_pinned_core(capsule)
        args = SimpleNamespace(hidden_dims=[128] * 3, context_norm='batch', shared_backbone=False,
                               n_snn_layers=3, n_downsample=2, corr_levels=4, corr_radius=4,
                               corr_implementation='reg', mixed_precision=False, feature_encoder='cnn',
                               context_encoder='cnn', matched_initialization_seed=seed, slow_fast_gru=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = model_class(args)
        self.base_parameter_count = sum(p.numel() for p in self.model.parameters())
        if architecture == 'original_I':
            self.model = loader.load_mechanism()(self.model, 'I', seed)
        before_stem = sum(p.numel() for p in self.model.parameters())
        if input_channels == 2:
            self.model.fnet.conv1 = expand_polarity_stem(self.model.fnet.conv1)
            self.model.cnet.conv1 = expand_polarity_stem(self.model.cnet.conv1)
        self.event_added_parameters = sum(p.numel() for p in self.model.parameters()) - before_stem
        self.architecture, self.input_channels, self.seed = architecture, input_channels, seed
        self.train_refinements, self.eval_refinements = train_refinements, eval_refinements
        self._padder, self._loader = padder, loader
        self.model.freeze_bn()

    def train(self, mode=True):
        super().train(mode)
        self.model.freeze_bn()
        return self

    def provenance(self):
        return {'source_identity_sha256': BASE_ID, 'archive_sha256': ARCHIVE_SHA,
                'registry_bytes_sha256': self._loader.registry_bytes_sha256,
                'module_sources': self._loader.provenance,
                'upstream_commit': '8c5e87750f25a035bca41bd37ae44b042a405f9a',
                'base_git_commit': '6ef353c5fc58878ead9f6fc5f58539b2a40abe28',
                'architecture': self.architecture, 'input_channels': self.input_channels,
                'sensor_semantics': '50_sensor_bins_as_CNN_channels_NOT_50_encoder_simulation_steps',
                'event_adaptation': self.input_channels == 2,
                'event_initializer': 'time_major100channels; each original weight copied ON/OFF divided sqrt2; separate trainable weights' if self.input_channels == 2 else None,
                'event_added_parameters': self.event_added_parameters,
                'current_initializer': 'isolated CPU RNG seed+19001' if self.architecture == 'original_I' else None,
                'input_normalization': 'legacy internal2*x-1; nonnegative event counts retain both polarities',
                'flow_conversion': 'd=-legacy_flow, where legacy_flow=x_right-x_left'}

    def forward(self, left, right):
        if left.shape != right.shape or left.ndim != 5 or left.shape[2] != self.input_channels:
            raise ValueError('left/right require matching B,50,C,H,W and registered modality')
        if sha((self._loader.capsule / 'source-registry.json').read_bytes()) != self._loader.registry_bytes_sha256:
            raise RuntimeError('baseline registry changed since verified module loading')
        from spikingjelly.activation_based import functional
        functional.reset_net(self.model)
        left, right = flatten_sensor_channels(left), flatten_sensor_channels(right)
        padder = self._padder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        refinements = self.train_refinements if self.training else self.eval_refinements
        if self.training:
            predictions, spikes, voltages = self.model(left, right, iters=refinements)
            predictions = [legacy_flow_to_disparity(padder.unpad(p)) for p in predictions]
        else:
            _, final, spikes = self.model(left, right, iters=refinements, test_mode=True)
            predictions, voltages = [legacy_flow_to_disparity(padder.unpad(final))], []
        return {'disparity': predictions[-1], 'refinement_disparities': predictions,
                'refinement_spikes': spikes, 'refinement_voltages': voltages,
                'diagnostics': {'hybrid': True, 'sensor_steps': 50, 'encoder_simulation_steps': 0,
                                'refinement_iterations': refinements, 'temporal_semantics': 'legacy_time_as_channel',
                                'event_adaptation': self.input_channels == 2,
                                'event_added_parameters': self.event_added_parameters,
                                'occlusion_supervision': 'unavailable_no_occlusion_head'},
                'baseline_provenance': self.provenance()}


def build_baseline(config):
    return StudyBaseline(**dict(config))
