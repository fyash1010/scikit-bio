# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import copy
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from skbio.stats.ordination import corr, mean_and_std, e_matrix, f_matrix, \
    center_distance_matrix

from skbio.stats.ordination._principal_coordinate_analysis import (
    center_distance_matrix as center_distance_matrix_array_api,
)
from skbio.stats.ordination import _utils as ord_utils
from skbio.stats.ordination._utils import _e_matrix_inplace, _f_matrix_inplace


class TestUtils(TestCase):
    def setUp(self):
        self.x = np.array([[1, 2, 3], [4, 5, 6]])
        self.y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        self.matrix = np.arange(1, 7).reshape(2, 3)
        self.matrix2 = np.arange(1, 10).reshape(3, 3)

        self.small_mat = np.array([[7, 5, 5], [4, 4, 9], [7, 5, 3]])
        self.dist_mat = np.asarray([[0., 7., 5., 5.], [7., 0., 4., 9.],
                                    [5., 4., 0., 3.], [5., 9., 3., 0.]],
                                   dtype=np.float64)
        self.dist_mat_fp32 = np.asarray([[0., 7., 5., 5.], [7., 0., 4., 9.],
                                         [5., 4., 0., 3.], [5., 9., 3., 0.]],
                                        dtype=np.float32)

    def test_mean_and_std(self):
        obs = mean_and_std(self.x)
        npt.assert_almost_equal((3.5, 1.707825127), obs)

        obs = mean_and_std(self.x, with_std=False)
        self.assertEqual((3.5, None), obs)

        obs = mean_and_std(self.x, ddof=2)
        npt.assert_almost_equal((3.5, 2.091650066), obs)

    def test_mean_and_std_no_mean_no_std(self):
        with npt.assert_raises(ValueError):
            mean_and_std(self.x, with_mean=False, with_std=False)

    def test_corr(self):
        obs = corr(self.small_mat)
        npt.assert_almost_equal(np.array([[1, 1, -0.94491118],
                                          [1, 1, -0.94491118],
                                          [-0.94491118, -0.94491118, 1]]),
                                obs)

    def test_corr_shape_mismatch(self):
        with npt.assert_raises(ValueError):
            corr(self.x, self.y)

    def test_e_matrix(self):
        E = e_matrix(self.matrix)
        expected_E = np.array([[-0.5, -2., -4.5],
                               [-8., -12.5, -18.]])
        npt.assert_almost_equal(E, expected_E)

    def test_f_matrix(self):
        F = f_matrix(self.matrix2)
        expected_F = np.zeros((3, 3))
        # Note that `test_make_F_matrix` in cogent is wrong
        npt.assert_almost_equal(F, expected_F)

    def test_e_matrix_inplace(self):
        E = _e_matrix_inplace(self.matrix)
        expected_E = np.array([[-0.5, -2., -4.5],
                               [-8., -12.5, -18.]])
        npt.assert_almost_equal(E, expected_E)

    def test_f_matrix_inplace(self):
        F = _f_matrix_inplace(self.matrix2)
        expected_F = np.zeros((3, 3))
        npt.assert_almost_equal(F, expected_F)

    def test_center_distance_matrix_inplace(self):
        dm_expected = f_matrix(e_matrix(self.dist_mat))

        # make copy of matrix to test inplace centering
        matrix_copy = copy.deepcopy(self.dist_mat)
        with patch.dict("os.environ", {"SKBIO_PCOA_CENTER_BACKEND": "cpu"}):
            dm_centered = center_distance_matrix(matrix_copy, inplace=False)

        # ensure that matrix_copy was NOT modified inplace
        self.assertTrue(np.array_equal(matrix_copy, self.dist_mat))

        # and ensure that the result of centering was correct
        npt.assert_almost_equal(dm_expected, dm_centered)

        # next, sort same matrix inplace
        matrix_copy2 = copy.deepcopy(self.dist_mat)
        with patch.dict("os.environ", {"SKBIO_PCOA_CENTER_BACKEND": "cpu"}):
            dm_centered_inp = center_distance_matrix(matrix_copy2, inplace=True)

        # and ensure that the result of inplace centering was correct
        npt.assert_almost_equal(dm_expected, dm_centered_inp)

    def test_center_distance_matrix_single(self):
        dm_expected = f_matrix(e_matrix(self.dist_mat_fp32))

        # make copy of matrix to test inplace centering
        matrix_copy = copy.deepcopy(self.dist_mat_fp32)
        with patch.dict("os.environ", {"SKBIO_PCOA_CENTER_BACKEND": "cpu"}):
            dm_centered = center_distance_matrix(matrix_copy, inplace=False)

        # ensure that matrix_copy was NOT modified inplace
        self.assertTrue(np.array_equal(matrix_copy, self.dist_mat_fp32))

        # and ensure that the result of centering was correct
        npt.assert_almost_equal(dm_expected, dm_centered)

        # next, sort same matrix inplace
        matrix_copy2 = copy.deepcopy(self.dist_mat_fp32)
        with patch.dict("os.environ", {"SKBIO_PCOA_CENTER_BACKEND": "cpu"}):
            dm_centered_inp = center_distance_matrix(matrix_copy2, inplace=True)

        # and ensure that the result of inplace centering was correct
        npt.assert_almost_equal(dm_expected, dm_centered_inp)

    def test_center_distance_matrix_invalid_backend(self):
        with patch.dict(
            "os.environ",
            {"SKBIO_PCOA_CENTER_BACKEND": "invalid_backend"},
        ):
            with self.assertRaisesRegex(ValueError, "SKBIO_PCOA_CENTER_BACKEND"):
                center_distance_matrix(self.dist_mat)

    def test_center_distance_matrix_cpu(self):
        dm_expected = f_matrix(e_matrix(self.dist_mat))

        with patch.dict("os.environ", {"SKBIO_PCOA_CENTER_BACKEND": "cpu"}), patch(
            "skbio.stats.ordination._utils.center_distance_matrix_cy",
            wraps=ord_utils.center_distance_matrix_cy,
        ) as center_distance_matrix_cy:
            dm_centered = center_distance_matrix(self.dist_mat)

        center_distance_matrix_cy.assert_called_once()
        npt.assert_allclose(dm_expected, dm_centered, rtol=1e-7, atol=1e-7)

    def test_center_distance_matrix_numba(self):
        try:
            import numba  # noqa: F401
        except Exception:
            self.skipTest("Numba is not importable.")
        from skbio.stats.ordination import _center_distance_matrix_numba as cdm_numba

        dm_expected = f_matrix(e_matrix(self.dist_mat))
        matrix_copy = copy.deepcopy(self.dist_mat)
        with patch.dict("os.environ", {"SKBIO_PCOA_CENTER_BACKEND": "numba"}), patch(
            "skbio.stats.ordination._center_distance_matrix_numba."
            "center_distance_matrix_nb",
            wraps=cdm_numba.center_distance_matrix_nb,
        ) as center_distance_matrix_nb:
            dm_centered = center_distance_matrix(matrix_copy)

        center_distance_matrix_nb.assert_called_once()
        self.assertTrue(np.array_equal(matrix_copy, self.dist_mat))
        npt.assert_allclose(dm_expected, dm_centered, rtol=1e-7, atol=1e-7)

    def test_center_distance_matrix_numba_inplace(self):
        try:
            import numba  # noqa: F401
        except Exception:
            self.skipTest("Numba is not importable.")
        from skbio.stats.ordination import _center_distance_matrix_numba as cdm_numba

        dm_expected = f_matrix(e_matrix(self.dist_mat))
        matrix_copy = copy.deepcopy(self.dist_mat)
        with patch.dict("os.environ", {"SKBIO_PCOA_CENTER_BACKEND": "numba"}), patch(
            "skbio.stats.ordination._center_distance_matrix_numba."
            "center_distance_matrix_nb",
            wraps=cdm_numba.center_distance_matrix_nb,
        ) as center_distance_matrix_nb:
            dm_centered = center_distance_matrix(matrix_copy, inplace=True)
        center_distance_matrix_nb.assert_called_once()
        npt.assert_allclose(dm_expected, dm_centered, rtol=1e-7, atol=1e-7)
        npt.assert_allclose(dm_expected, matrix_copy, rtol=1e-7, atol=1e-7)

    def test_center_distance_matrix_numba_gpu(self):
        try:
            from skbio.stats.ordination import _center_distance_matrix_numba_gpu
            from skbio.stats.ordination._center_distance_matrix_numba_gpu import (
                NumbaGPUUnavailableError,
            )
        except Exception:
            self.skipTest("Numba GPU helper is not importable.")

        try:
            with patch.dict(
                "os.environ",
                {
                    "SKBIO_PCOA_CENTER_BACKEND": "numba_gpu",
                    "SKBIO_NUMBA_GPU_BACKEND": "cuda",
                },
            ), patch(
                "skbio.stats.ordination._center_distance_matrix_numba_gpu."
                "center_distance_matrix_numba_gpu",
                wraps=(
                    _center_distance_matrix_numba_gpu
                    .center_distance_matrix_numba_gpu
                ),
            ) as center_distance_matrix_numba_gpu:
                dm_centered = center_distance_matrix(self.dist_mat)
        except NumbaGPUUnavailableError:
            self.skipTest("Numba GPU backend is not available.")

        center_distance_matrix_numba_gpu.assert_called_once()
        dm_expected = f_matrix(e_matrix(self.dist_mat))
        npt.assert_allclose(dm_expected, dm_centered, rtol=1e-7, atol=1e-7)

    def test_center_distance_matrix_jax(self):
        try:
            import jax
            import jax.numpy as jnp
        except Exception:
            self.skipTest("JAX is not importable.")

        dm_expected = f_matrix(e_matrix(self.dist_mat_fp32))
        dist_mat = jnp.asarray(self.dist_mat_fp32)
        dm_centered = center_distance_matrix_array_api(dist_mat)
        self.assertIsInstance(dm_centered, jax.Array)
        dm_centered = np.asarray(jax.device_get(dm_centered))

        npt.assert_allclose(dm_expected, dm_centered, rtol=1e-5, atol=1e-5)

    def test_center_distance_matrix_cupy(self):
        try:
            import cupy as cp
        except Exception:
            self.skipTest("CuPy is not importable.")

        try:
            dist_mat = cp.asarray(self.dist_mat)
            dm_centered = center_distance_matrix_array_api(dist_mat)
            self.assertIsInstance(dm_centered, cp.ndarray)
            dm_centered = cp.asnumpy(dm_centered)
        except Exception as e:
            self.skipTest(f"CuPy backend is not available: {e}")

        dm_expected = f_matrix(e_matrix(self.dist_mat))
        npt.assert_allclose(dm_expected, dm_centered, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    main()
