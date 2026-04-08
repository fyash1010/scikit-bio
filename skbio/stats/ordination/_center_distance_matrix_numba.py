# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Numba implementation and benchmark for distance-matrix double-centering.

This module is intentionally standalone so it can be compared against the
existing Cython implementation in ``skbio.stats.ordination._cutils``.
"""

import time

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def e_matrix_means_nb(mat, centered, row_means):
    """Apply E-matrix transform and collect row/global means in one pass."""
    n = mat.shape[0]
    global_sum = np.float64(0.0)

    for row in prange(n):
        row_sum = np.float64(0.0)

        for col in range(n):
            val = np.float64(mat[row, col])
            el = np.float64(-0.5) * val * val
            centered[row, col] = el
            row_sum += el

        row_means[row] = row_sum / np.float64(n)
        global_sum += row_sum

    return (global_sum / np.float64(n)) / np.float64(n)


@njit(parallel=True, fastmath=True, cache=True)
def f_matrix_inplace_nb(row_means, global_mean, centered):
    """Double-center E-matrix in-place."""
    n = centered.shape[0]

    n_blocks = (n + 23) // 24
    for brow in prange(n_blocks):
        trow = brow * 24
        trow_max = min(trow + 24, n)

        for tcol in range(0, n, 24):
            tcol_max = min(tcol + 24, n)

            for row in range(trow, trow_max):
                gr_mean = global_mean - row_means[row]

                for col in range(tcol, tcol_max):
                    centered[row, col] += gr_mean - row_means[col]


def center_distance_matrix_nb(mat, centered):
    """Drop-in style replacement for ``center_distance_matrix_cy``.

    Parameters
    ----------
    mat : ndarray (n, n), float32 or float64, C-contiguous
        Input distance matrix.
    centered : ndarray (n, n), same dtype as ``mat``
        Pre-allocated output array. May alias ``mat`` for in-place use.
    """
    n = mat.shape[0]
    row_means = np.zeros(n, dtype=mat.dtype)
    global_mean = e_matrix_means_nb(mat, centered, row_means)
    f_matrix_inplace_nb(row_means, np.float64(global_mean), centered)


def warmup():
    """Trigger JIT compilation for float32 and float64 specializations."""
    for dtype in (np.float64, np.float32):
        d = np.random.rand(8, 8).astype(dtype)
        d = (d + d.T) / 2
        out = np.empty_like(d)
        center_distance_matrix_nb(d, out)


def _numpy_reference(mat, centered):
    """Pure-NumPy double-centering reference implementation."""
    e = -0.5 * mat**2
    row_means = e.mean(axis=1)
    global_mean = e.mean()
    centered[:] = e - row_means[:, None] - row_means[None, :] + global_mean


def _time(fn, *args, repeats=8):
    fn(*args)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return np.array(ts)


def benchmark(n=2000, dtype=np.float64, repeats=8):
    """Benchmark Numba centering against NumPy reference."""
    rng = np.random.default_rng(0)
    mat = rng.random((n, n)).astype(dtype)
    mat = ((mat + mat.T) / 2).copy()

    out_nb = np.empty_like(mat)
    out_np = np.empty_like(mat)

    t_nb = _time(center_distance_matrix_nb, mat, out_nb, repeats=repeats)
    t_np = _time(_numpy_reference, mat, out_np, repeats=repeats)

    if dtype == np.float32:
        np.testing.assert_allclose(out_nb, out_np, rtol=1e-4, atol=1e-6)
    else:
        np.testing.assert_allclose(out_nb, out_np, rtol=1e-4)

    return {
        "numba_mean_s": float(t_nb.mean()),
        "numba_std_s": float(t_nb.std()),
        "numpy_mean_s": float(t_np.mean()),
        "numpy_std_s": float(t_np.std()),
        "speedup": float(t_np.mean() / t_nb.mean()),
    }


if __name__ == "__main__":
    warmup()
    for size in (500, 1000, 2000, 4000):
        result = benchmark(n=size, dtype=np.float64, repeats=8)
        print(size, result)
    print("float32", benchmark(n=2000, dtype=np.float32, repeats=8))
