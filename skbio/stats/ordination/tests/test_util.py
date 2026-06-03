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

from skbio.stats.ordination import _utils as ord_utils
from skbio.stats.ordination._cutils import center_distance_matrix_cy
from skbio.stats.ordination._utils import _e_matrix_inplace, _f_matrix_inplace
from skbio.util import numba_code


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

    @numba_code
    def test_center_distance_matrix_numba(self):
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

    @numba_code
    def test_center_distance_matrix_numba_inplace(self):
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

    @numba_code
    def test_center_distance_matrix_numba_matches_cython(self):
        from skbio.stats.ordination._center_distance_matrix_numba import (
            center_distance_matrix_nb,
        )

        rng = np.random.default_rng(0)
        for dtype, rtol, atol in (
            (np.float64, 1e-12, 1e-12),
            (np.float32, 1e-5, 1e-5),
        ):
            mat = rng.random((31, 31), dtype=dtype)
            mat = (mat + mat.T) / dtype(2)
            np.fill_diagonal(mat, dtype(0))
            mat = np.ascontiguousarray(mat)

            cy_centered = np.empty_like(mat)
            nb_centered = np.empty_like(mat)
            center_distance_matrix_cy(mat, cy_centered)
            center_distance_matrix_nb(mat, nb_centered)

            npt.assert_allclose(nb_centered, cy_centered, rtol=rtol, atol=atol)


if __name__ == '__main__':
    main()
