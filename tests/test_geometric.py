import pytest
from gpu_pairwise import pairwise_distance
from scipy.spatial.distance import cdist, squareform

import numpy as np
from numpy.testing import assert_allclose


SCIPY_METRICS = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
    'braycurtis',
    'canberra', 
    'chebyshev',
]


# def test_simple_call():
#     distances = pairwise_distance([[1, 0], [1, 1]])
#     assert (distances == [[0, 0.5], [0.5, 0]]).all()


# def test_non_contiguous_call():
#     source = np.transpose(np.array([[0] * 9, [1] * 9, [0] * 9] * 9, dtype=np.bool_))
#     print(source.astype(int))
#     print(_pack64(source))


# def test_squareform():
#     distances = pairwise_distance([
#         [0, 0],
#         [1, 0],
#         [0, 0],
#         [1, 1],
#         [1, 0],
#         [0, 0],
#     ], metric='equal', out_dtype='bool', squareform=True)

#     assert (list(distances) == [
#         False, True, False, False, True, 
#         False, False, True, False, 
#         False, False, True, 
#         False, False,
#         False
#     ])


# def test_squareform_jaccard():
#     data = [
#         [0, 0, 1],
#         [0, 1, 1],
#         [1, 1, 1],
#         [0, 0, 0],
#     ]
#     distances_norm = pairwise_distance(data, metric='jaccard')
#     distances_square = pairwise_distance(data, metric='jaccard', squareform=True)

#     assert list(squareform(distances_norm)) == list(distances_square)


# # def test_equal():
# #     distances = pairwise_distance([
# #         [0, 0],
# #         [1, 0],
# #         [0, 0],
# #         [1, 1],
# #         [1, 0],
# #         [0, 0],
# #     ], metric='equal')
# #     assert (distances == [
# #         [1, 0, 1, 0, 0, 1],
# #         [0, 1, 0, 0, 1, 0],
# #         [1, 0, 1, 0, 0, 1],
# #         [0, 0, 0, 1, 0, 0],
# #         [0, 1, 0, 0, 1, 0],
# #         [1, 0, 1, 0, 0, 1],
# #     ]).all()


@pytest.mark.parametrize("matrix", [
    [[1, 0], [0, 1]],
    [[0, 0], [0, 0]],
    [[0,], [10,]],
    [[2, 0.2], [1, 1]],
    [[1.1, 100], [100, 1.1]],

    np.random.randint(2, size=(100, 2)),
    np.random.rand(50, 50) * 100,
    np.random.rand(2, 100),
])
@pytest.mark.parametrize("metric", SCIPY_METRICS)
def test_orig(matrix, metric):
    ours = pairwise_distance(matrix, metric=metric)
    matrix = np.asarray(matrix)
    theirs = cdist(matrix, matrix, metric=metric)
    np.fill_diagonal(theirs, 0)
    assert_allclose(ours, theirs)


# @pytest.mark.parametrize("matrix", [
#     [[1, 0], [0, 1]],
#     [[0, 0], [0, 0]],
#     [[0,], [1,]],
#     [[1, 0], [1, 1]],
#     [[1, 1], [1, 1]],
#     #[[], []],
#     np.random.randint(2, size=(9, 9)),
#     np.random.randint(2, size=(100, 2)),
#     np.random.randint(2, size=(50, 50)),
#     np.random.randint(2, size=(2, 100)),
# ])
# @pytest.mark.parametrize("metric", SCIPY_METRICS)
# def test_squareform_variations(matrix, metric):
#     ours_normal = pairwise_distance(matrix, metric=metric)
#     ours_square = pairwise_distance(matrix, metric=metric, squareform=True)
#     theirs_square = squareform(ours_normal, checks=False)
#     assert_allclose(ours_square, theirs_square)


# @pytest.mark.parametrize("matrix, weights", [
#     ([[1, 0], [0, 1]], [0.5, 1]),
#     ([[0, 0], [0, 0]], [1, 1]),
#     ([[0,], [1,]], [1]),
#     ([[1, 0], [1, 1]], [1, 0.2]),
#     ([[1, 1], [1, 1]], [0.5, 0.5]),
#     (np.random.randint(2, size=(100, 2)), np.random.rand(2)),
#     (np.random.randint(2, size=(50, 50)), np.random.rand(50)),
#     (np.random.randint(2, size=(2, 100)), np.random.rand(100)),
# ])
# @pytest.mark.parametrize("metric", SCIPY_METRICS)
# def test_weight(matrix, weights, metric):
#     if metric == 'sokalsneath' and any(np.asarray(matrix).sum(axis=1) == 0):
#         return
#     if metric == 'sokalmichener':
#         pytest.skip("scipy bug #13693")
#     ours = pairwise_distance(matrix, metric=metric, weights=weights)
#     matrix = np.asarray(matrix)
#     weights = np.asarray(weights)
#     theirs = cdist(matrix, matrix, metric=metric, w=weights)
#     assert_allclose(ours, theirs)


# @pytest.mark.parametrize("matrix, weights", [
#     ([[1, 0], [0, 1]], [0.5, 1]),
#     ([[0, 0], [0, 0]], [1, 1]),
#     ([[0,], [1,]], [1]),
#     ([[1, 0], [1, 1]], [1, 0.2]),
#     ([[1, 1], [1, 1]], [0.5, 0.5]),
#     (np.random.randint(2, size=(9, 9)), np.random.rand(9)),
#     (np.random.randint(2, size=(100, 2)), np.random.rand(2)),
#     (np.random.randint(2, size=(50, 50)), np.random.rand(50)),
#     (np.random.randint(2, size=(2, 100)), np.random.rand(100)),
# ])
# @pytest.mark.parametrize("metric", SCIPY_METRICS)
# @pytest.mark.parametrize("dtype, scale", [
#     ('float32', 1), 
#     ('float64', 1), 
#     ('uint8', 255), 
#     ('uint16', 65535)
# ])
# def test_weight_dtypes(matrix, weights, metric, dtype, scale):
#     if metric == 'sokalsneath' and any(np.asarray(matrix).sum(axis=1) == 0):
#         return
#     if metric == 'sokalmichener':
#         pytest.skip("scipy bug #13693")
#     ours = pairwise_distance(matrix, metric=metric, weights=weights, out_dtype=dtype)
#     matrix = np.asarray(matrix)
#     weights = np.asarray(weights)
#     theirs = cdist(matrix, matrix, metric=metric, w=weights)
#     if scale != 1:
#        theirs = np.nan_to_num(theirs, nan=0)
#     assert_allclose(ours, theirs * scale, atol=1, )


# @pytest.mark.parametrize("matrix", [
#     [[1, 0], [0, 1]],
#     [[0, 0], [0, 0]],
#     [[0,], [1,]],
#     [[1, 0], [1, 1]],
#     [[1, 1], [1, 1]],
#     #[[], []],
#     np.random.randint(2, size=(9, 9)),
#     np.random.randint(2, size=(100, 2)),
#     np.random.randint(2, size=(50, 50)),
#     np.random.randint(2, size=(2, 100)),
# ])
# @pytest.mark.parametrize("metric", SCIPY_METRICS)
# @pytest.mark.parametrize("dtype, scale", [
#     ('float32', 1), 
#     ('float64', 1), 
#     ('uint8', 255), 
#     ('uint16', 65535)
# ])
# def test_noweight_dtypes(matrix, metric, dtype, scale):
#     ours = pairwise_distance(matrix, metric=metric, out_dtype=dtype)
#     matrix = np.asarray(matrix)
#     theirs = cdist(matrix, matrix, metric=metric)
#     if scale != 1:
#        theirs = np.nan_to_num(theirs, nan=0)
#     assert_allclose(ours, theirs * scale, atol=1.)

