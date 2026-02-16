# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from numbers import Integral
from warnings import warn

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from skbio.stats.distance import DistanceMatrix
from skbio.table._tabular import _create_table, _create_table_1d
from ._ordination_results import OrdinationResults
from ._utils import scale
from skbio.binaries import (
    pcoa_fsvd_available as _skbb_pcoa_fsvd_available,
    pcoa_fsvd as _skbb_pcoa_fsvd,
)
from skbio.util._decorator import params_aliased

# Notes:
# - We intentionally use `jnp` (JAX) operations throughout below.

from scipy.linalg import eigh as scipy_eigh


def e_matrix(distance_matrix):  # ADDED JAX version
    return distance_matrix * distance_matrix / -2


def f_matrix(E_matrix):
    row_means = E_matrix.mean(axis=1, keepdims=True)
    col_means = E_matrix.mean(axis=0, keepdims=True)
    matrix_mean = E_matrix.mean()
    return E_matrix - row_means - col_means + matrix_mean


def center_distance_matrix(distance_matrix, inplace=False):
    """Center a distance matrix using JAX arrays.

    Note: JAX arrays are immutable, so the ``inplace`` argument is accepted
    for API compatibility but ignored — the function always returns a
    centered array (a JAX DeviceArray).
    """
    # compute centered matrix using JAX operations
    return f_matrix(e_matrix(distance_matrix))


def _host_partial_eigh(matrix_jax, subidx):
    """Compute partial eigendecomposition on host via SciPy LAPACK.

    This moves `matrix_jax` to the host (NumPy), ensures contiguous
    layout and a LAPACK-friendly dtype, runs SciPy's `eigh` with
    `subset_by_index` (fast), then moves results back to JAX.
    """
    mat_np = np.asarray(matrix_jax)
    if mat_np.dtype != np.float64:
        mat_np = mat_np.astype(np.float64, copy=False)
    mat_np = np.ascontiguousarray(mat_np)

    eigvals_np, eigvecs_np = scipy_eigh(mat_np, subset_by_index=subidx)

    # Move results back to device
    return jnp.asarray(eigvals_np), jnp.asarray(eigvecs_np)


@params_aliased(
    [
        ("dimensions", "number_of_dimensions", "0.7.0", False),
        ("distmat", "distance_matrix", "0.7.0", False),
    ]
)
def pcoa_JAX(  # ADDED _JAX
    distmat,
    method="eigh",
    dimensions=0,
    inplace=False,
    seed=None,
    warn_neg_eigval=0.01,
    output_format=None,
):
    r"""Perform Principal Coordinate Analysis (PCoA).

    PCoA is an ordination method similar to Principal Components Analysis (PCA), with
    the difference that it operates on distance matrices, calculated using meaningful
    and typically non-Euclidian methods.

    Parameters
    ----------
    distmat : DistanceMatrix
        The input distance matrix.
    method : str, optional
        Matrix decomposition method to use. Default is "eigh" (eigendecomposition),
        which computes exact eigenvectors and eigenvalues for all dimensions. The
        alternate is "fsvd" (fast singular value decomposition), a heuristic that can
        compute only a given number of dimensions.
    dimensions : int or float, optional
        Dimensions to reduce the distance matrix to. This number determines how many
        eigenvectors and eigenvalues will be returned. If an integer is provided, the
        exact number of dimensions will be retained. If a float between 0 and 1, it
        represents the fractional cumulative variance to be retained. Default is 0,
        which will retain the same number of dimensions as the distance matrix.
    inplace : bool, optional
        If True, the input distance matrix will be centered in-place to reduce memory
        consumption, at the cost of losing the original distances. Default is False.
    seed : int or np.random.Generator, optional
        A user-provided random seed or random generator instance for method "fsvd".
        See :func:`details <skbio.util.get_rng>`.

        .. versionadded:: 0.6.3

    warn_neg_eigval : bool or float, optional
        Raise a warning if any negative eigenvalue is obtained and its magnitude
        exceeds the specified fraction threshold compared to the largest positive
        eigenvalue, which suggests potential inaccuracy in the PCoA result. Default is
        0.01. Set True to warn regardless of the magnitude. Set False to disable
        warning completely.

        .. versionadded:: 0.6.3

    output_format : optional
        Standard table parameters. See :ref:`table_params` for details.

    Returns
    -------
    OrdinationResults
        Object that stores the PCoA results, including eigenvalues, the proportion
        explained by each of them, and transformed sample coordinates.

    See Also
    --------
    OrdinationResults

    Notes
    -----
    Principal Coordinate Analysis (PCoA) was first described in [1]_.

    This function uses a choice of two methods for matrix decomposition: The default
    method, ``eigh``, performs eigendecomposition, an exact method that computes all
    eigenvectors and eigenvalues. The alternative method, ``fsvd``, performs fast
    singular value decomposition (FSVD) [2]_, an efficient heuristic method that
    allows a custom number of dimensions to be specified to reduce calculation at the
    cost of losing accuracy. The degree of accuracy lost is dependent on dataset.

    Note that the default method ``eigh`` does not natively support a given number of
    dimensions to reduce a matrix to. Therefore, if this parameter is specified, all
    eigenvectors and eigenvalues will be simply be computed with no speed gain, and
    only the specified number of dimensions will be returned.

    Eigenvalues represent the magnitude of individual principal coordinates, and
    they are usually positive. However, negative eigenvalues can occur when the
    distances were calculated using a non-Euclidean metric that does not satisfy
    triangle inequality. If the negative eigenvalues are small in magnitude compared
    to the largest positive eigenvalue, it is usually safe to ignore them. However,
    large negative eigenvalues may indicate result inaccuracy, in which case a warning
    message will be displayed. The paramter ``warn_neg_eigval`` controls the threshold
    for the warning.

    PCoA on Euclidean distances is equivalent to Principal Component Analysis (PCA).
    However, in ecology, the Euclidean distance preserved by PCA is often not a good
    choice because it deals poorly with double zeros. For example, species have
    unimodal distributions along environmental gradients. If a species is absent from
    two sites simultaneously, it can't be known if an environmental variable is too
    high in one of them and too low in the other, or too low in both, etc. On the other
    hand, if a species is present in two sites, that means that the sites are similar.

    Note that the returned eigenvectors are not normalized to unit length.

    References
    ----------
    .. [1] Gower, J. C. (1966). Some distance properties of latent root and vector
       methods used in multivariate analysis. Biometrika, 53(3-4), 325-338.

    .. [2] Halko, N., Martinsson, P. G., Shkolnisky, Y., & Tygert, M. (2011). An
       algorithm for the principal component analysis of large data sets. SIAM
       Journal on Scientific computing, 33(5), 2580-2594.

    """
    # this converts to redundant form, regardless of input type
    distmat = DistanceMatrix(distmat)

    # If no dimension specified, by default will compute all eigenvectors
    # and eigenvalues
    if dimensions == 0:
        if method == "fsvd" and distmat.data.shape[0] > 10:
            warn(
                "FSVD: since no value for dimensions is specified, "
                "PCoA for all dimensions will be computed, which may "
                "result in long computation time if the original "
                "distance matrix is large.",
                RuntimeWarning,
            )
        elif method == "eigh" and distmat.data.shape[0] > 10:
            warn(
                "EIGH: since no value for dimensions is specified, "
                "PCoA for all dimensions will be computed, which may "
                "result in long computation time if the original "
                "distance matrix is large.",
                RuntimeWarning,
            )

        # distmat is guaranteed to be square
        dimensions = distmat.data.shape[0]
    elif dimensions < 0:
        raise ValueError(
            "Invalid operation: cannot reduce distance matrix "
            "to negative dimensions using PCoA. Did you intend "
            'to specify the default value "0", which sets '
            "the dimensions equal to the "
            "dimensionality of the given distance matrix?"
        )
    elif dimensions > distmat.data.shape[0]:
        raise ValueError("Invalid operation: cannot extend distance matrix size.")
    elif not isinstance(dimensions, Integral) and dimensions > 1:
        raise ValueError(
            "Invalid operation: A floating-point number greater than 1 cannot be "
            "supplied as the number of dimensions."
        )

    if warn_neg_eigval and not 0 <= warn_neg_eigval <= 1:
        raise ValueError(
            "warn_neg_eigval must be Boolean or a floating-point number between 0 "
            "and 1."
        )

    # new parameter for ndim = number of dimensions (accounting for
    # non-int values)
    ndim = dimensions

    # Perform eigendecomposition
    if method == "eigh":
        long_method_name = "Principal Coordinate Analysis"
        # Center distance matrix, a requirement for PCoA here
        matrix_data = center_distance_matrix(
            distmat.data, inplace=inplace
        )  # REMOVED - Original line
        matrix_data = jnp.asarray(matrix_data)  # ADDED
        if 0 < dimensions < 1:
            if matrix_data.shape[0] > 10:
                warn(
                    "EIGH: since value for dimensions is specified as float,"
                    " PCoA for all dimensions will be computed, which may"
                    " result in long computation time if the original"
                    " distance matrix is large."
                    " Consider specifying an integer value to optimize performance.",
                    RuntimeWarning,
                )
            ndim = matrix_data.shape[0]
        # If a subset of eigenpairs is requested (ndim < n), use SciPy's
        # partial solver via host transfer for performance. Otherwise use
        # JAX's full eigendecomposition to remain on-device.
        if ndim < matrix_data.shape[0]:
            subidx = [matrix_data.shape[0] - ndim, matrix_data.shape[0] - 1]
            eigvals, eigvecs = _host_partial_eigh(matrix_data, subidx)
        else:
            eigvals, eigvecs = jnp.linalg.eigh(matrix_data)
    elif method == "fsvd":
        long_method_name = "Approximate Principal Coordinate Analysis using FSVD"
        if 0 < dimensions < 1:
            if distmat.data.shape[0] > 10:
                warn(
                    "FSVD: since value for dimensions is specified as float,"
                    " PCoA for all dimensions will be computed, which may"
                    " result in long computation time if the original"
                    " distance matrix is large."
                    " Consider specifying an integer value to optimize performance.",
                    RuntimeWarning,
                )
            ndim = distmat.data.shape[0]
        # if _skbb_pcoa_fsvd_available( - Eventually want to support but for now ignore
        #     distmat.data, dimensions, inplace, seed
        # ):  # pragma: no cover
        #     # unlikely to throw here, but just in case
        #     try:
        #         eigvals, coordinates, proportion_explained = _skbb_pcoa_fsvd(
        #             distmat.data, dimensions, inplace, seed
        #         )
        #         eigvals, coordinates, proportion_explained = jnp.asarray(eigvals),
        # jnp.asarray(coordinates), jnp.asarray(proportion_explained)  # ADDED
        #         return _encapsulate_pcoa_result(
        #             long_method_name,
        #             eigvals,
        #             coordinates,
        #             proportion_explained,
        #             distmat.ids,
        #             output_format,
        #         )
        #     except Exception as e:
        #         warn(
        #             "Attempted to use binaries.pcoa_fsvd but failed, "
        #             "using regular logic instead.",
        #             RuntimeWarning,
        #         )
        # if we got here, we could not use skbb
        # Center distance matrix, a requirement for PCoA here
        matrix_data = center_distance_matrix(
            distmat.data, inplace=inplace
        )  # REMOVED - Original line
        matrix_data = jnp.asarray(matrix_data)  # ADDED

        # eigvals, eigvecs = _fsvd(matrix_data, ndim, seed=seed) # Removed
        eigvals, eigvecs = _fsvd_JAX(
            matrix_data, ndim, seed=seed
        )  # ADDED - JAX version
    else:
        raise ValueError(
            "PCoA eigendecomposition method {} not supported.".format(method)
        )
    # Ensure dimensions does not exceed available dimensions
    # dimensions = min(dimensions, eigvals.shape[0])

    # cogent makes eigenvalues positive by taking the
    # abs value, but that doesn't seem to be an approach accepted
    # by L&L to deal with negative eigenvalues. We raise a warning
    # in that case. First, we make values close to 0 equal to 0.
    # negative_close_to_zero = np.isclose(eigvals, 0) # REMOVED - Original line
    negative_close_to_zero = jnp.isclose(eigvals, 0)  # ADDED - JAX version
    # eigvals[negative_close_to_zero] = 0 # REMOVED - Original line
    eigvals = eigvals.at[negative_close_to_zero].set(0)  # ADDED - JAX version

    # eigvals might not be ordered, so we first sort them, then analogously
    # sort the eigenvectors by the ordering of the eigenvalues too
    idxs_descending = eigvals.argsort()[::-1]
    eigvals = eigvals[idxs_descending]
    eigvecs = eigvecs[:, idxs_descending]

    # large negative eigenvalues suggest result inaccuracy
    # see: https://github.com/scikit-bio/scikit-bio/issues/1410
    if warn_neg_eigval and eigvals[-1] < 0:
        if warn_neg_eigval is True or -eigvals[-1] > eigvals[0] * warn_neg_eigval:
            warn(
                "The result contains negative eigenvalues that are large in magnitude,"
                " which may suggest result inaccuracy. See Notes for details. The"
                " negative-most eigenvalue is {0} whereas the largest positive one is"
                " {1}.".format(eigvals[-1], eigvals[0]),
                RuntimeWarning,
            )

    # If we return only the coordinates that make sense (i.e., that have a
    # corresponding positive eigenvalue), then Jackknifed Beta Diversity
    # won't work as it expects all the OrdinationResults to have the same
    # number of coordinates. In order to solve this issue, we return the
    # coordinates that have a negative eigenvalue as 0
    num_positive = (eigvals >= 0).sum()
    eigvecs = eigvecs.at[:, num_positive:].set(0)  # ADDED - JAX version
    eigvals = eigvals.at[num_positive:].set(0)  # ADDED - JAX version

    if ndim != distmat.data.shape[0]:
        # Since the dimension parameter, hereafter referred to as 'd',
        # restricts the number of eigenvalues and eigenvectors that FSVD
        # computes, we need to use an alternative method to compute the sum
        # of all eigenvalues, used to compute the array of proportions
        # explained. Otherwise, the proportions calculated will only be
        # relative to d number of dimensions computed; whereas we want
        # it to be relative to the entire dimensionality of the
        # centered distance matrix.

        # An alternative method of calculating th sum of eigenvalues is by
        # computing the trace of the centered distance matrix.
        # See proof outlined here: https://goo.gl/VAYiXx
        # sum_eigenvalues = np.trace(matrix_data) # REMOVED - Original line
        sum_eigenvalues = jnp.trace(matrix_data)  # ADDED - JAX version
    else:
        # Calculate proportions the usual way
        # sum_eigenvalues = np.sum(eigvals) # REMOVED - Original line
        sum_eigenvalues = jnp.sum(eigvals)  # ADDED - JAX version

    proportion_explained = eigvals / sum_eigenvalues
    if 0 < dimensions < 1:
        # cumulative_variance = np.cumsum(proportion_explained) # REMOVED
        cumulative_variance = jnp.cumsum(proportion_explained)  # ADDED - JAX version
        ndim = (
            jnp.searchsorted(cumulative_variance, dimensions, side="left") + 1
        )  # ADDED - JAX version
        # gives the number of dimensions needed to reach specified variance
        # updates number of dimensions to reach the requirement of variance.
        dimensions = ndim

    # In case eigh is used, eigh computes all eigenvectors and -values.
    # So if dimensions was specified, we manually need to ensure
    # only the requested number of dimensions
    # (number of eigenvectors and eigenvalues, respectively) are returned.
    eigvecs = eigvecs[:, :dimensions]
    eigvals = eigvals[:dimensions]
    proportion_explained = proportion_explained[:dimensions]

    # Scale eigenvalues to have length = sqrt(eigenvalue). This
    # works because np.linalg.eigh returns normalized
    # eigenvectors. Each row contains the coordinates of the
    # objects in the space of principal coordinates. Note that at
    # least one eigenvalue is zero because only n-1 axes are
    # needed to represent n points in a euclidean space.
    # coordinates = eigvecs * np.sqrt(eigvals) # REMOVED - Original line
    coordinates = eigvecs * jnp.sqrt(eigvals)  # ADDED - JAX version

    return _encapsulate_pcoa_result(
        long_method_name,
        eigvals,
        coordinates,
        proportion_explained,
        distmat.ids,
        output_format,
    )


def _encapsulate_pcoa_result(
    long_method_name, eigvals, coordinates, proportion_explained, ids, output_format
):
    r"""Format PCoA results

    Helper function for converting raw buffers of the pcoa function
    into proper OrdinationResult object.

    Parameters
    ----------
    long_method_name: str
        The verbose name of the method used.
    eigvals: ndarray
        Eigenvalues
    coordinates: ndarray
        Sample coordinates
    proportion_explained: ndarray
        Proportions explained
    ids: array
        Distance matrix ids
    output_format : optional
        Standard table parameters. See :ref:`table_params` for details.

    Returns
    -------
    OrdinationResults
        Object that stores the PCoA results, including eigenvalues, the proportion
        explained by each of them, and transformed sample coordinates.

    See Also
    --------
    OrdinationResults
    """

    dimensions = eigvals.shape[0]
    axis_labels = ["PC%d" % i for i in range(1, dimensions + 1)]

    eigvals = _create_table_1d(
        eigvals, index=axis_labels, backend=output_format
    )  # ADDED - Assuming _create_table_1d is not JAX compatible
    if isinstance(
        eigvals, pd.Series
    ):  # ADDED - Assuming _create_table_1d is not JAX compatible
        eigvals = jnp.asarray(
            eigvals.values
        )  # ADDED - Assuming _create_table_1d is not JAX compatible
    elif (
        "polars" in str(type(eigvals)).lower()
    ):  # ADDED - Assuming _create_table_1d is not JAX compatible
        eigvals = jnp.asarray(
            eigvals.to_numpy()
        )  # ADDED - Assuming _create_table_1d is not JAX compatible
    else:  # ADDED - Assuming _create_table_1d is not JAX compatible
        eigvals = jnp.asarray(
            eigvals
        )  # ADDED - Assuming _create_table_1d is not JAX compatible

    samples = _create_table(
        coordinates,
        index=ids,
        columns=axis_labels,
        backend=output_format,
    )  # ADDED - Assuming _create_table is not JAX compatible
    if isinstance(
        samples, pd.DataFrame
    ):  # ADDED - Assuming _create_table is not JAX compatible
        samples = jnp.asarray(
            samples.values
        )  # ADDED - Assuming _create_table is not JAX compatible
    elif (
        "polars" in str(type(samples)).lower()
    ):  # ADDED - Assuming _create_table is not JAX compatible
        samples = jnp.asarray(
            samples.to_numpy()
        )  # ADDED - Assuming _create_table is not JAX compatible
    else:  # ADDED - Assuming _create_table is not JAX compatible
        samples = jnp.asarray(
            samples
        )  # ADDED - Assuming _create_table is not JAX compatible

    proportion_explained = _create_table_1d(
        proportion_explained, index=axis_labels, backend=output_format
    )  # ADDED - Assuming _create_table_1d is not JAX compatible
    if isinstance(
        proportion_explained, pd.Series
    ):  # ADDED - Assuming _create_table_1d is not JAX compatible
        proportion_explained = jnp.asarray(
            proportion_explained.values
        )  # ADDED - Assuming _create_table_1d is not JAX compatible
    elif (
        "polars" in str(type(proportion_explained)).lower()
    ):  # ADDED - Assuming _create_table_1d is not JAX compatible
        proportion_explained = jnp.asarray(
            proportion_explained.to_numpy()
        )  # ADDED - Assuming _create_table_1d is not JAX compatible
    else:  # ADDED - Assuming _create_table_1d is not JAX compatible
        proportion_explained = jnp.asarray(
            proportion_explained
        )  # ADDED - Assuming _create_table_1d is not JAX compatible

    return OrdinationResults(
        short_method_name="PCoA",
        long_method_name=long_method_name,
        eigvals=eigvals,  # ADDED
        # samples=_create_table(
        #     coordinates,
        #     index=ids,
        #     columns=axis_labels,
        #     backend=output_format,
        # ), # REMOVED - Original lines
        samples=samples,  # ADDED
        # proportion_explained=_create_table_1d(
        #     proportion_explained, index=axis_labels, backend=output_format
        # ), # REMOVED - Original line
        proportion_explained=proportion_explained,  # ADDED
    )


@params_aliased([("dimensions", "number_of_dimensions", "0.7.0", False)])
def _fsvd_JAX(
    centered_distance_matrix, dimensions=10, seed=None
):  # ADDED - JAX version
    """Perform singular value decomposition.

    More specifically in this case eigendecomposition, using fast heuristic algorithm
    nicknamed "FSVD" (FastSVD), adapted and optimized from the algorithm described
    by Halko et al (2011).

    Parameters
    ----------
    centered_distance_matrix : np.array
       Numpy matrix representing the distance matrix for which the
       eigenvectors and eigenvalues shall be computed
    dimensions : int
       Number of dimensions to keep. Must be lower than or equal to the
       rank of the given distance_matrix.
    seed : int or np.random.Generator, optional
        A user-provided random seed or random generator instance.

    Returns
    -------
    np.array
       Array of eigenvectors, each with dimensions length.
    np.array
       Array of eigenvalues, a total number of dimensions.

    Notes
    -----
    The algorithm is based on [1]_.

    Ported from MATLAB implementation described in [2]_.

    References
    ----------
    .. [1] Halko, N., Martinsson, P. G., Shkolnisky, Y., & Tygert, M. (2011). An
       algorithm for the principal component analysis of large data sets. SIAM
       Journal on Scientific computing, 33(5), 2580-2594.

    .. [2] https://stats.stackexchange.com/a/11934/211065

    """
    m, n = centered_distance_matrix.shape

    # Number of levels of the Krylov method to use.
    # For most applications, num_levels=1 or num_levels=2 is sufficient.
    num_levels = 1

    # Changes the power of the spectral norm, thus minimizing the error).
    use_power_method = False

    # Note: a (conjugate) transpose is removed for performance, since we
    # only expect square matrices.
    if m != n:
        raise ValueError("FSVD expects square distance matrix")

    if dimensions > m or dimensions > n:
        raise ValueError(
            "FSVD: dimensions cannot be larger than"
            " the dimensionality of the given distance matrix."
        )

    if dimensions < 0:
        raise ValueError(
            "Invalid operation: cannot reduce distance matrix "
            "to negative dimensions using PCoA. Did you intend "
            'to specify the default value "0", which sets '
            "the dimensions equal to the "
            "dimensionality of the given distance matrix?"
        )

    k = dimensions + 2

    # Form a real nxl matrix G whose entries are independent, identically
    # distributed Gaussian random variables of zero mean and unit variance
    # rng = get_rng(seed) # REMOVED - Original line
    # Create an integer seed for JAX PRNG. If ``seed`` is None,
    # generate a non-deterministic integer from NumPy's RNG to avoid
    # passing None to jax.random.PRNGKey.
    if seed is None:
        seed_int = np.random.randint(0, 2**31 - 1)
    else:
        try:
            seed_int = int(seed)
        except Exception:
            seed_int = np.random.randint(0, 2**31 - 1)

    rng = jax.random.PRNGKey(seed_int)
    G = jax.random.normal(rng, shape=(n, k))

    # `use_power_method` is constantly False, so `if` won't start.
    if use_power_method:  # pragma: no cover
        # use only the given exponent
        H = jnp.dot(centered_distance_matrix, G)

        for x in range(2, num_levels + 2):
            # enhance decay of singular values
            H = jnp.dot(centered_distance_matrix, jnp.dot(centered_distance_matrix, H))

    else:
        # compute the m x l matrices H^{(0)}, ..., H^{(i)}
        # Note that this is done implicitly in each iteration below.
        H = jnp.dot(centered_distance_matrix, G)
        # to enhance performance
        H = jnp.hstack(
            (H, jnp.dot(centered_distance_matrix, jnp.dot(centered_distance_matrix, H)))
        )

        # `num_levels` is constantly 1, so `for` loop won't start
        for x in range(3, num_levels + 2):  # pragma: no cover
            tmp = jnp.dot(
                centered_distance_matrix, jnp.dot(centered_distance_matrix, H)
            )

            H = jnp.hstack(
                (
                    H,
                    jnp.dot(
                        centered_distance_matrix, jnp.dot(centered_distance_matrix, tmp)
                    ),
                )
            )

    # Using the pivoted QR-decomposition, form a real m * ((i+1)l) matrix Q
    # whose columns are orthonormal, s.t. there exists a real
    # ((i+1)l) * ((i+1)l) matrix R for which H = QR
    Q, R = jnp.linalg.qr(H)

    # Compute the n * ((i+1)l) product matrix T = A^T Q
    T = jnp.dot(centered_distance_matrix, Q)

    # Form an SVD of T
    # jnp.linalg.svd returns (U, S, Vh). Use V = Vh.T to match original
    U_svd, St, Vh = jnp.linalg.svd(T, full_matrices=False)
    W = Vh.T

    # Compute the m * ((i+1)l) product matrix
    Ut = jnp.dot(Q, W)

    U_fsvd = Ut[:, :dimensions]

    S = St[:dimensions]

    # drop imaginary component, if we got one
    # Note:
    #   In cogent, after computing eigenvalues/vectors, the imaginary part
    #   is dropped, if any. We know for a fact that the eigenvalues are
    #   real, so that's not necessary, but eigenvectors can in principle
    #   be complex (see for example
    #   http://math.stackexchange.com/a/47807/109129 for details)
    eigenvalues = S.real
    eigenvectors = U_fsvd.real

    return eigenvalues, eigenvectors


def pcoa_biplot(ordination, y):
    """Compute the projection of descriptors into a PCoA matrix.

    Parameters
    ----------
    ordination : OrdinationResults
        The computed principal coordinates analysis of dimensions (n, c) where
        the matrix ``y`` will be projected onto.
    y : DataFrame
        Samples by features table of dimensions (n, m). These can be
        environmental features or abundance counts. This table should be
        normalized in cases of dimensionally heterogenous physical variables.

    Returns
    -------
    OrdinationResults
        The modified input object that includes projected features onto the
        ordination space in the ``features`` attribute.

    Notes
    -----
    This implementation is as described in Chapter 9 of [1]_.

    References
    ----------
    .. [1] Legendre P. and Legendre L. 1998. Numerical Ecology. Elsevier, Amsterdam.

    """
    # acknowledge that most saved ordinations lack a name, however if they have
    # a name, it should be PCoA
    if ordination.short_method_name != "" and ordination.short_method_name != "PCoA":
        raise ValueError(
            "This biplot computation can only be performed in a PCoA matrix."
        )

    if set(y.index) != set(ordination.samples.index):
        raise ValueError(
            "The eigenvectors and the descriptors must describe the same samples."
        )

    eigvals = ordination.eigvals.values
    coordinates = ordination.samples
    N = coordinates.shape[0]

    # align the descriptors and eigenvectors in a sample-wise fashion
    y = y.reindex(coordinates.index)

    # S_pc from equation 9.44
    # Represents the covariance matrix between the features matrix and the
    # column-centered eigenvectors of the pcoa.
    spc = (1 / (N - 1)) * y.values.T.dot(scale(coordinates, ddof=1))

    # U_proj from equation 9.55, is the matrix of descriptors to be projected.
    #
    # Only get the power of non-zero values, otherwise this will raise a
    # divide by zero warning. There shouldn't be negative eigenvalues(?)
    Uproj = jnp.sqrt(N - 1) * spc.dot(
        jnp.diag(jnp.power(eigvals, -0.5, where=eigvals > 0))
    )

    ordination.features = pd.DataFrame(
        data=Uproj, index=y.columns.copy(), columns=coordinates.columns.copy()
    )
    ordination.features.fillna(0.0, inplace=True)

    return ordination
