import numpy as np
from numba import cuda
import math

from . import util
from .metric import Metric



TEMPLATE = """
@cuda.jit('void({{ type_sig(*fn_args.values()) }})')
def {{ fn_name }}({{ ", ".join(fn_args.keys()) }}):
    rows, cols = values.shape
    row_from, row_to = cuda.grid(2)

    {% for var in metric.vars %}
    {{ var }} = 0.
    {%- endfor %}
    
    if row_from < rows and row_to < rows:
        if row_from == row_to:
            out[row_from, row_to] = {{ metric.DIAGONAL_VAL }}
        else:
            for col in range(cols):
                from_val = values[row_from, col]
                to_val = values[row_to, col]
                {% for clause in metric.calc_cell %}
                {{ clause }}
                {% endfor %}
            
        
            {{ metric.calc_dist }}
            out[row_from, row_to] = distance
"""


class ContinuousMetric(Metric):
    NAMES = ()
    VARS = NotImplemented
    CALC_CELL = NotImplemented
    CALC_DIST = NotImplemented
    DIAGONAL_VAL = 0.

    # @dataclass
    # class Params(Metric.Params):
    #     w: Sequence[float] = None
    #     nan: str = None

    @property
    def template(self):
        return TEMPLATE

    @property
    def vars(self):
        return self.VARS

    @property
    def calc_cell(self):
        return self.CALC_CELL

    @property
    def calc_dist(self):
        return self.CALC_DIST


class Euclidean(ContinuousMetric):
    NAMES = ('euclidean', )
    VARS = ('total', )
    CALC_CELL = "total += (to_val - from_val) ** 2",
    CALC_DIST = '''
            distance = math.sqrt(total)
    '''


class SqEuclidean(ContinuousMetric):
    NAMES = ('sqeuclidean', )
    VARS = ('total', )
    CALC_CELL = "total += (to_val - from_val) ** 2",
    CALC_DIST = '''
            distance = total
    '''


class CityBlock(ContinuousMetric):
    NAMES = ('cityblock', 'manhattan')
    VARS = ('total', )
    CALC_CELL = "total += abs(to_val - from_val)",
    CALC_DIST = '''
            distance = total
    '''

class BrayCurtis(ContinuousMetric):
    NAMES = ('braycurtis', )
    VARS = ('total_sub', 'total_add')
    CALC_CELL = (
        "total_sub += abs(to_val - from_val)",
        "total_add += abs(to_val + from_val)",
    )
    CALC_DIST = '''
            if total_add == 0:
                distance = math.nan
            else:
                distance = total_sub / total_add
    '''


class Canberra(ContinuousMetric):
    NAMES = ('canberra', )
    VARS = ('total', )
    CALC_CELL = (
        'num = abs(to_val - from_val)',
        'denom = abs(to_val) + abs(from_val)',        
        'if denom > 0:',
        '   total += num / denom',
    )
    CALC_DIST = '''
            distance = total
    '''

class Chebyshev(ContinuousMetric):
    NAMES = ('chebyshev', )
    VARS = ('max_val', )
    CALC_CELL = (
        'diff = abs(to_val - from_val)',
        'max_val = max(max_val, diff)'
    )
    CALC_DIST = '''
            distance = max_val
    '''


# class Minkowski(Metric):
#     NAMES = ('minkowski', )
#     VARS = ('total', )
#     PARAMS = ('P', )
#     CALC_CELL = "            total += (to_val - from_val) ** 2"
#     CALC_DIST = '''
#         distance = math.sqrt(total)
#     '''



def pairwise_distance(values, metric='euclidean', squareform=False, block_size=(16, 16), out_dtype='float32', params=None):
    metric_cls = Metric.get_metric(metric)
    params = params or {}
    if squareform:
        if isinstance(block_size, (list, tuple)):
            block_size = block_size[0]
        return metric_cls.pairwise_distance_squareform(values, block_size=block_size, out_dtype=out_dtype, params=params)
    return metric_cls.pairwise_distance(values, block_size=block_size, out_dtype=out_dtype, params=params)