# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Optional Numba GPU implementation of distance-matrix double-centering."""

import os

import numpy as np


THREADS_PER_BLOCK = 256
_KERNEL_CACHE = {}


class NumbaGPUUnavailableError(RuntimeError):
    """Raised when the optional Numba GPU backend cannot be used."""


def _get_gpu_module(gpu_backend="auto"):
    """Return the requested Numba GPU backend module.

    Numba is imported lazily so GPU support remains an optional runtime
    dependency. ``gpu_backend="cuda"`` uses NVIDIA Numba CUDA, while
    ``gpu_backend="hip"`` uses ROCm Numba-HIP directly.
    """
    gpu_backend = os.environ.get("SKBIO_NUMBA_GPU_BACKEND", gpu_backend)
    if gpu_backend not in {"auto", "cuda", "hip"}:
        raise ValueError(
            f"gpu_backend must be 'auto', 'cuda', or 'hip', not {gpu_backend!r}."
        )

    errors = []
    candidates = ("cuda", "hip") if gpu_backend == "auto" else (gpu_backend,)
    for candidate in candidates:
        try:
            if candidate == "cuda":
                from numba import cuda as gpu
            else:
                from numba import hip as gpu
        except Exception as e:
            errors.append((candidate, e))
            continue

        try:
            available = gpu.is_available()
        except Exception as e:
            errors.append((candidate, e))
            continue

        if available:
            return gpu
        errors.append((candidate, NumbaGPUUnavailableError("backend unavailable")))

    detail = "; ".join(f"{name}: {type(err).__name__}: {err}" for name, err in errors)
    raise NumbaGPUUnavailableError(
        "center_backend='numba_gpu' requires an available Numba CUDA "
        f"or Numba-HIP backend. Tried {gpu_backend!r}. {detail}"
    )


def _get_cuda_module(gpu_backend="auto"):
    """Backward-compatible alias for the selected Numba GPU module."""
    return _get_gpu_module(gpu_backend)


def _validate_distance_matrix(distance_matrix):
    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0] != distance_matrix.shape[1]
    ):
        raise ValueError("Distance matrix must be a square 2D array.")
    if distance_matrix.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("Numba GPU centering requires float32 or float64 input.")


def _validate_device_distance_matrix(distance_matrix):
    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0] != distance_matrix.shape[1]
    ):
        raise ValueError("Distance matrix must be a square 2D array.")
    if np.dtype(distance_matrix.dtype) not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        raise TypeError("Numba GPU centering requires float32 or float64 input.")
    if not (
        hasattr(distance_matrix, "__cuda_array_interface__")
        or hasattr(distance_matrix, "__hip_array_interface__")
    ):
        raise TypeError(
            "Device input must expose __cuda_array_interface__ or "
            "__hip_array_interface__."
        )


def _compile_kernels(gpu):
    cache_key = id(gpu)
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    @gpu.jit
    def e_matrix_row_sums_kernel(mat, centered, row_sums):
        row = gpu.blockIdx.x
        tid = gpu.threadIdx.x
        n = mat.shape[0]
        shared = gpu.shared.array(THREADS_PER_BLOCK, dtype=np.float64)

        row_sum = 0.0
        for col in range(tid, n, gpu.blockDim.x):
            val = float(mat[row, col])
            e_val = -0.5 * val * val
            centered[row, col] = e_val
            row_sum += e_val

        shared[tid] = row_sum
        gpu.syncthreads()

        stride = gpu.blockDim.x // 2
        while stride > 0:
            if tid < stride:
                shared[tid] += shared[tid + stride]
            gpu.syncthreads()
            stride //= 2

        if tid == 0:
            row_sums[row] = shared[0]

    @gpu.jit
    def f_matrix_kernel(row_sums, global_mean, centered):
        col, row = gpu.grid(2)
        n = centered.shape[0]

        if row < n and col < n:
            row_mean = row_sums[row] / n
            col_mean = row_sums[col] / n
            centered[row, col] = centered[row, col] - row_mean - col_mean + global_mean

    kernels = e_matrix_row_sums_kernel, f_matrix_kernel
    _KERNEL_CACHE[cache_key] = kernels
    return kernels


def center_distance_matrix_numba_gpu(
    distance_matrix, inplace=False, gpu_backend="auto"
):
    """Center a NumPy distance matrix using an optional Numba GPU backend."""
    _validate_distance_matrix(distance_matrix)
    if not distance_matrix.flags.c_contiguous:
        distance_matrix = np.asarray(distance_matrix, order="C")

    gpu = _get_gpu_module(gpu_backend)
    e_kernel, f_kernel = _compile_kernels(gpu)

    n = distance_matrix.shape[0]
    d_mat = gpu.to_device(distance_matrix)
    d_centered = d_mat if inplace else gpu.device_array(d_mat.shape, dtype=d_mat.dtype)
    d_row_sums = gpu.device_array((n,), dtype=np.float64)

    e_kernel[n, THREADS_PER_BLOCK](d_mat, d_centered, d_row_sums)
    row_sums = d_row_sums.copy_to_host()
    global_mean = float(row_sums.sum() / (n * n))

    threads = (32, 8)
    blocks = ((n + threads[0] - 1) // threads[0], (n + threads[1] - 1) // threads[1])
    f_kernel[blocks, threads](d_row_sums, global_mean, d_centered)

    centered = d_centered.copy_to_host()
    if inplace:
        distance_matrix[...] = centered
        return distance_matrix
    return centered


def center_distance_matrix_numba_gpu_device(
    distance_matrix, inplace=False, gpu_backend="auto"
):
    """Center a GPU-resident distance matrix using Numba kernels.

    This helper accepts inputs that expose ``__cuda_array_interface__`` (for
    example CuPy arrays) and keeps the full centered matrix on the GPU. Only the
    row sums are copied back to the host to compute the scalar global mean.
    """
    _validate_device_distance_matrix(distance_matrix)

    gpu = _get_gpu_module(gpu_backend)
    e_kernel, f_kernel = _compile_kernels(gpu)

    n = distance_matrix.shape[0]
    d_mat = gpu.as_cuda_array(distance_matrix)
    d_centered = d_mat if inplace else gpu.device_array(d_mat.shape, dtype=d_mat.dtype)
    d_row_sums = gpu.device_array((n,), dtype=np.float64)

    e_kernel[n, THREADS_PER_BLOCK](d_mat, d_centered, d_row_sums)
    row_sums = d_row_sums.copy_to_host()
    global_mean = float(row_sums.sum() / (n * n))

    threads = (32, 8)
    blocks = ((n + threads[0] - 1) // threads[0], (n + threads[1] - 1) // threads[1])
    f_kernel[blocks, threads](d_row_sums, global_mean, d_centered)

    return d_centered
