"""Optional CUDA inference for the fixed local NMRF support operator.

The training/reference operator remains ordinary differentiable PyTorch.
Only the binary evaluation path packs Q/K and uses integer AND/popcount.
"""
import ctypes as c
from importlib.util import find_spec
import os
from pathlib import Path
import sys

import torch

CUDA_SOURCE = r'''
constexpr unsigned FULL = 0xffffffffu;
constexpr int HEADS = 4, WIDTH = 32, TOKENS = 144, CANDIDATES = 4, SLOTS = 20;
extern "C" __global__ void pack_qk(
    const float* q, const float* k, unsigned* qp, unsigned* kp, int rows) {
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / WIDTH;
    if (row >= rows) return;
    const int index = row * WIDTH + lane;
    const unsigned a = __ballot_sync(FULL, q[index] >= 1.f);
    const unsigned b = __ballot_sync(FULL, k[index] >= 1.f);
    if (lane == 0) { qp[row] = a; kp[row] = b; }
}

template<bool Binary, typename Feature>
__device__ void aggregate(
    const Feature* q, const Feature* k, const float* values,
    const float* prior, const float* relative, const long long* pool,
    const long long* rel, float* output, int rows) {
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / WIDTH;
    if (row >= rows) return;
    const int head = row % HEADS;
    const int token = row / HEADS;
    const int local_token = token % TOKENS;
    const int pixel = local_token / CANDIDATES;
    const int window_base = token - local_token;
    float accumulation = 0.f;
    float mass = 0.f;
    for (int slot = 0; slot < SLOTS; ++slot) {
        const float g = __shfl_sync(FULL, lane == 0 ? prior[token * SLOTS + slot] : 0.f, 0);
        if (g == 0.f) continue;
        const int key = (window_base + pool[pixel * SLOTS + slot]) * HEADS + head;
        float agreement = 0.f;
        if (Binary) {
            if (lane == 0) agreement = __popc(
                reinterpret_cast<const unsigned*>(q)[row] &
                reinterpret_cast<const unsigned*>(k)[key]);
        } else {
            agreement = reinterpret_cast<const float*>(q)[row * WIDTH + lane] *
                        reinterpret_cast<const float*>(k)[key * WIDTH + lane];
            for (int delta = 16; delta > 0; delta /= 2) {
                const float next = __shfl_down_sync(FULL, agreement, delta);
                if (lane < delta) agreement += next;
            }
        }
        const float weight = __shfl_sync(FULL, lane == 0 ? agreement * g : 0.f, 0);
        if (weight == 0.f) continue;
        const int r = rel[pixel * (SLOTS / CANDIDATES) + slot / CANDIDATES];
        const float value = values[key * WIDTH + lane] + relative[(r * HEADS + head) * WIDTH + lane];
        accumulation = fmaf(weight, value, accumulation);
        if (lane == 0) mass += weight;
    }
    const float denom = __shfl_sync(FULL, lane == 0 ? fmaxf(1.f, mass) : 1.f, 0);
    output[row * WIDTH + lane] = accumulation / denom;
}

extern "C" __global__ void binary_messages(
    const unsigned* q, const unsigned* k, const float* values, const float* prior,
    const float* relative, const long long* pool, const long long* rel, float* output, int rows) {
    aggregate<true>(q, k, values, prior, relative, pool, rel, output, rows);
}
extern "C" __global__ void continuous_messages(
    const float* q, const float* k, const float* values, const float* prior,
    const float* relative, const long long* pool, const long long* rel, float* output, int rows) {
    aggregate<false>(q, k, values, prior, relative, pool, rel, output, rows);
}
'''


def _bind(library, name, args):
    function = getattr(library, name)
    function.argtypes = args
    function.restype = c.c_int
    return function


def _check(result, name):
    if result != 0:
        raise RuntimeError(f'{name} failed with CUDA status {result}')


def compile_cubin(architecture='sm_86'):
    """Build native SASS using the CUDA 11.8 DLLs already bundled with Torch."""
    if sys.platform != 'win32':
        raise RuntimeError('The verified NVRTC loader is for the existing Windows CUDA environment')
    torch_lib = Path(find_spec('torch').origin).parent / 'lib'
    old_path = os.environ.get('PATH')
    os.environ['PATH'] = str(torch_lib) + os.pathsep + (old_path or '')
    try:
        with os.add_dll_directory(str(torch_lib)):
            builtins = c.WinDLL(str(torch_lib / 'nvrtc-builtins64_118.dll'))
            nvrtc = c.WinDLL(str(torch_lib / 'nvrtc64_112_0.dll'))
            create = _bind(nvrtc, 'nvrtcCreateProgram',
                           [c.POINTER(c.c_void_p), c.c_char_p, c.c_char_p, c.c_int,
                            c.POINTER(c.c_char_p), c.POINTER(c.c_char_p)])
            compile_program = _bind(nvrtc, 'nvrtcCompileProgram',
                                    [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)])
            log_size = _bind(nvrtc, 'nvrtcGetProgramLogSize', [c.c_void_p, c.POINTER(c.c_size_t)])
            log_data = _bind(nvrtc, 'nvrtcGetProgramLog', [c.c_void_p, c.c_void_p])
            cubin_size = _bind(nvrtc, 'nvrtcGetCUBINSize', [c.c_void_p, c.POINTER(c.c_size_t)])
            cubin_data = _bind(nvrtc, 'nvrtcGetCUBIN', [c.c_void_p, c.c_void_p])
            destroy = _bind(nvrtc, 'nvrtcDestroyProgram', [c.POINTER(c.c_void_p)])
            program = c.c_void_p()
            _check(create(c.byref(program), CUDA_SOURCE.encode(), b'binary_support.cu', 0, None, None),
                   'nvrtcCreateProgram')
            try:
                options = (c.c_char_p * 2)(b'--std=c++14', f'--gpu-architecture={architecture}'.encode())
                result = compile_program(program, len(options), options)
                if result:
                    size = c.c_size_t()
                    _check(log_size(program, c.byref(size)), 'nvrtcGetProgramLogSize')
                    log = c.create_string_buffer(size.value)
                    _check(log_data(program, log), 'nvrtcGetProgramLog')
                    raise RuntimeError(f'NVRTC compile failed ({result}): {log.value.decode(errors="replace")}')
                size = c.c_size_t()
                _check(cubin_size(program, c.byref(size)), 'nvrtcGetCUBINSize')
                output = c.create_string_buffer(size.value)
                _check(cubin_data(program, output), 'nvrtcGetCUBIN')
                return output.raw
            finally:
                _check(destroy(c.byref(program)), 'nvrtcDestroyProgram')
    finally:
        if old_path is None:
            os.environ.pop('PATH', None)
        else:
            os.environ['PATH'] = old_path


class FusedLocalSupportCUDA:
    """Own a CUBIN loaded in Torch's current CUDA context for the process lifetime."""

    def __init__(self, cubin, device):
        if sys.platform != 'win32':
            raise RuntimeError('The verified CUDA driver loader requires Windows')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Expected the CUDA device for the fixed local kernel')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        driver = c.WinDLL('nvcuda.dll')
        current = _bind(driver, 'cuCtxGetCurrent', [c.POINTER(c.c_void_p)])
        module_load = _bind(driver, 'cuModuleLoadData', [c.POINTER(c.c_void_p), c.c_void_p])
        function_load = _bind(driver, 'cuModuleGetFunction',
                              [c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p])
        self._launch = _bind(driver, 'cuLaunchKernel',
                            [c.c_void_p, *([c.c_uint] * 7), c.c_void_p,
                             c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)])
        self._module = c.c_void_p()
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device)
            context = c.c_void_p()
            _check(current(c.byref(context)), 'cuCtxGetCurrent')
            if not context:
                raise RuntimeError('Expected the current Torch CUDA context')
            image = c.create_string_buffer(cubin)
            _check(module_load(c.byref(self._module), image), 'cuModuleLoadData')
            self._functions = {}
            for name in ('pack_qk', 'binary_messages', 'continuous_messages'):
                function = c.c_void_p()
                _check(function_load(c.byref(function), self._module, name.encode()), 'cuModuleGetFunction')
                self._functions[name] = function

    def _run(self, name, tensors, rows):
        block = 128
        grid = (rows * 32 + block - 1) // block
        values = [c.c_void_p(tensor.data_ptr()) for tensor in tensors] + [c.c_int(rows)]
        parameters = (c.c_void_p * len(values))(*(c.addressof(value) for value in values))
        with torch.cuda.device(self.device):
            stream = c.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            _check(self._launch(self._functions[name], grid, 1, 1, block, 1, 1,
                                0, stream, parameters, None), name)

    def pack(self, q, k):
        if (q.shape != k.shape or q.shape[-1] != 32 or q.device != k.device
                or q.dtype != torch.float32 or k.dtype != torch.float32 or q.device != self.device):
            raise ValueError('Expected matching CUDA FP32 features with 32 channels per head')
        q, k = q.contiguous(), k.contiguous()
        packed_q = torch.empty(q.shape[:-1], device=q.device, dtype=torch.int32)
        packed_k = torch.empty_like(packed_q)
        self._run('pack_qk', (q, k, packed_q, packed_k), packed_q.numel())
        return packed_q, packed_k

    def aggregate(self, mode, q, k, value, prior, relative, pool, rel):
        expected = (q.shape[0], 36, 4, 4, 32)
        if (q.shape != expected or k.shape != expected or value.shape != expected
                or prior.shape != (q.shape[0], 36, 4, 20) or relative.shape != (121, 4, 32)
                or pool.shape != (36, 20) or rel.shape != (36, 5)
                or any(x.dtype != torch.float32 or x.device != value.device
                       for x in (q, k, value, prior, relative))
                or any(x.dtype != torch.long or x.device != value.device for x in (pool, rel))
                or value.device != self.device):
            raise ValueError('Unexpected fixed local-support CUDA tensor layout')
        if mode == 'binary':
            q, k = self.pack(q, k)
        elif mode == 'continuous':
            q, k = torch.sigmoid(q - 1), torch.sigmoid(k - 1)
        else:
            raise ValueError('Unknown support feature mode')
        output = torch.empty_like(value, memory_format=torch.contiguous_format)
        buffers = tuple(x.contiguous() for x in (q, k, value, prior, relative, pool, rel))
        self._run(mode + '_messages', (*buffers, output), value.numel() // 32)
        return output
