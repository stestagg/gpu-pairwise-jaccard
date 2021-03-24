import pytest
from gpu_pairwise.boolean import pairwise_distance
from scipy.spatial.distance import cdist
import numpy as np
from numpy.testing import assert_allclose

METRICS = [
    'jaccard', 
    'matching', 
    'hamming',
    'dice',
    'kulsinski', 
    'rogerstanimoto', 
    'russellrao', 
    'sokalmichener',
    'sokalsneath',
]

def test_simple_call():
    distances = pairwise_distance([[1, 0], [1, 1]])
    assert (distances == [[0, 0.5], [0.5, 0]]).all()


@pytest.mark.parametrize("matrix", [
    [[1, 0], [0, 1]],
    [[0, 0], [0, 0]],
    [[0,], [1,]],
    [[1, 0], [1, 1]],
    [[1, 1], [1, 1]],
    #[[], []],
    np.random.randint(2, size=(100, 2)),
    np.random.randint(2, size=(50, 50)),
    np.random.randint(2, size=(2, 100)),
])
@pytest.mark.parametrize("metric", METRICS)
def test_noweight(matrix, metric):
    ours = pairwise_distance(matrix, metric=metric)
    matrix = np.asarray(matrix)
    theirs = cdist(matrix, matrix, metric=metric)
    assert_allclose(ours, theirs)


@pytest.mark.parametrize("matrix, weights", [
    ([[1, 0], [0, 1]], [0.5, 1]),
    ([[0, 0], [0, 0]], [1, 1]),
    ([[0,], [1,]], [1]),
    ([[1, 0], [1, 1]], [1, 0.2]),
    ([[1, 1], [1, 1]], [0.5, 0.5]),
    (np.random.randint(2, size=(100, 2)), np.random.rand(2)),
    (np.random.randint(2, size=(50, 50)), np.random.rand(50)),
    (np.random.randint(2, size=(2, 100)), np.random.rand(100)),
])
@pytest.mark.parametrize("metric", METRICS)
def test_weight(matrix, weights, metric):
    if metric == 'sokalsneath' and any(np.asarray(matrix).sum(axis=1) == 0):
        return
    if metric == 'sokalmichener':
        pytest.skip("scipy bug #13693")
    ours = pairwise_distance(matrix, metric=metric, weights=weights)
    matrix = np.asarray(matrix)
    weights = np.asarray(weights)
    theirs = cdist(matrix, matrix, metric=metric, w=weights)
    assert_allclose(ours, theirs)


@pytest.mark.parametrize("matrix, weights", [
    ([[1, 0], [0, 1]], [0.5, 1]),
    ([[0, 0], [0, 0]], [1, 1]),
    ([[0,], [1,]], [1]),
    ([[1, 0], [1, 1]], [1, 0.2]),
    ([[1, 1], [1, 1]], [0.5, 0.5]),
    (np.random.randint(2, size=(100, 2)), np.random.rand(2)),
    (np.random.randint(2, size=(50, 50)), np.random.rand(50)),
    (np.random.randint(2, size=(2, 100)), np.random.rand(100)),
])
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype, scale", [
    ('float32', 1), 
    ('float64', 1), 
    ('uint8', 255), 
    ('uint16', 65535)
])
def test_weight_dtypes(matrix, weights, metric, dtype, scale):
    if metric == 'sokalsneath' and any(np.asarray(matrix).sum(axis=1) == 0):
        return
    if metric == 'sokalmichener':
        pytest.skip("scipy bug #13693")
    ours = pairwise_distance(matrix, metric=metric, weights=weights, out_dtype=dtype)
    matrix = np.asarray(matrix)
    weights = np.asarray(weights)
    theirs = cdist(matrix, matrix, metric=metric, w=weights)
    if scale != 1:
       theirs = np.nan_to_num(theirs, nan=0)
    assert_allclose(ours, theirs * scale, atol=1, )


@pytest.mark.parametrize("matrix", [
    [[1, 0], [0, 1]],
    [[0, 0], [0, 0]],
    [[0,], [1,]],
    [[1, 0], [1, 1]],
    [[1, 1], [1, 1]],
    #[[], []],
    np.random.randint(2, size=(100, 2)),
    np.random.randint(2, size=(50, 50)),
    np.random.randint(2, size=(2, 100)),
])
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype, scale", [
    ('float32', 1), 
    ('float64', 1), 
    ('uint8', 255), 
    ('uint16', 65535)
])
def test_noweight_dtypes(matrix, metric, dtype, scale):
    ours = pairwise_distance(matrix, metric=metric, out_dtype=dtype)
    matrix = np.asarray(matrix)
    theirs = cdist(matrix, matrix, metric=metric)
    if scale != 1:
       theirs = np.nan_to_num(theirs, nan=0)
    assert_allclose(ours, theirs * scale, atol=1.)