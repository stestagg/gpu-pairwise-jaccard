from dataclasses import dataclass
import math
from typing import Sequence
from numba import cuda

import numpy as np

from .metric import Metric
from . import  util


def _pack64(values):
    """
    These metrics work by counting 1s and 0s in long sets of presence flags.

    In the simple, unweighted case, it's possible to pack these flags into fixed-sized
    integers, and take advantage of bitwise operators to calculate the metrics.
    
    This function takes an array that looks like this:
    [[True, False, True...], ...]
    and turns it into something like this:
    [[9758984729384, 98127809801, ...], ...]

    All packing is done along the first axis only.

    It does that by:
    1. Taking each 8 values in the boolean input array, and packing them into bits in a byte, i.e.:
    ```
    [False, False, False, False, False, False, True, True] 
    -> [0, 0, 0, 0, 0, 0, 1, 1] 
    -> 0b00000011 
    -> 3
    ```
    Each 8 value stride along the first axis is packed like this, with '0' values used to pad the 
    last group if it doesn't contain 8 values.

    Now, we have an array that looks like this:
    [2, 1, 253, 64, 19, 127, ...]

    Next this array is '0' padded until the length is a multiple of 8 values long

    Finally, the padded array is reinterpreted as a sequence of unsighed 64-bit numbers.

    There is a potential danger with zero-padding these values, as they introduce data
    that isn't part of the actual dataset. Luckily the operators used in these algoritms 
    do not ever actually count 0 values (on both sides):

    NTT: count of a = True and b = True
    NTF: count of a = True and b = False
    NFT: count of a = False and b = True
    NNEQ: count of NTF or NFT
    NNZ: count of NTF or NFT or NTT

    If an algorithm needed to count NFF (count of a = False and b = False) directly
    then a more comple solution should be used.
    """

    values = np.ascontiguousarray(values)
    packed = np.packbits(values, -1)
    last_dim = packed.shape[-1]
    new_dim = int(np.ceil(last_dim / 8) * 8)
    pad_len = new_dim - last_dim
    if pad_len:
        pads = [(0, 0)] * (len(packed.shape) - 1) + [(0, pad_len)]
        padded = np.pad(packed, pads, mode='constant', constant_values=0)
    else:
        padded = packed
    view = padded.view(np.uint64)
    return view


STATS_VARS = {
    'NTT': 'from_val & to_val',
    'NTF': 'from_val & (~to_val)',
    'NFT': '(~from_val) & to_val',
    'NNEQ': 'from_val ^ to_val',
    'NNZ': 'from_val | to_val',
}


TEMPLATE = """
@cuda.jit('void({{ type_sig(*fn_args.values()) }})')
def {{ fn_name }}({{ ", ".join(fn_args.keys()) }}):
    rows, cols = values.shape
    row_from, row_to = cuda.grid(2)
    {% with vars = metric.var_clauses -%}
    {% for var in vars %}
    {{ var }} = 0
    {%- endfor %}
    
    if row_from < rows and row_to < rows:
        if row_from == row_to:
            out[row_from, row_to] = {{ metric.DIAGONAL_VAL }}
        else:
            for col in range(cols):
                from_val = values[row_from, col]
                to_val = values[row_to, col]
                {% if 'weights' in fn_args %}
                weight = weights[col]
                {% for var, logic in vars.items() %}
                {{ var }} += ({{ logic }}) * weight
                {%- endfor %}            

                {% else %}
                {% for var, logic in vars.items() %}
                {{ var }} += ({{ logic }})
                {%- endfor %}
                {% endif %}
            
        {%- endwith %}
            {{ metric.distance_clause }}
            out[row_from, row_to] = (
                distance
                {% if metric.out_scale_factor != 1 %} * {{ metric.out_scale_factor }}{% endif %}
            )
"""


class BooleanMetric(Metric):
    NAMES = ()
    VARS = NotImplemented
    CALC_DIST = NotImplemented
    DEFAULT_NAN_VAL = 0.
    NEEDS_N = False
    DIAGONAL_VAL = 0

    @dataclass
    class Params(Metric.Params):
        w: Sequence[float] = None
        nan: str = None

    @property
    def use_packed(self):
        return self.params.w is None

    @property
    def template(self):
        return TEMPLATE

    @property
    def out_scale_factor(self):
        return self.out_dtype.unit_scale

    @property
    def var_clauses(self):
        if self.use_packed:
            return {v: f'cuda.popc({STATS_VARS[v]})' for v in self.VARS}    
        return {v: STATS_VARS[v] for v in self.VARS}

    @property
    def nan_val_str(self):
        nan_val = self.params.nan
        if nan_val is None:
            if self.out_dtype.has_nan:
                nan_val = 'math.nan'
            else:
                nan_val = self.DEFAULT_NAN_VAL
        return str(nan_val)

    @property
    def distance_clause(self):
        return self.CALC_DIST.replace('!!NANVAL!!', self.nan_val_str)

    def get_fn_args(self, values, stream):
        if self.use_packed:
            packed_values = _pack64(values)
            device_array = cuda.to_device(packed_values, stream=stream)
        else:
            values_array = np.asarray(values, dtype=np.bool_)
            device_array = cuda.to_device(values_array, stream=stream)
        fn_args = super().get_fn_args(device_array, stream)
        if self.NEEDS_N:
            fn_args['N'] = np.shape(values)[0]
        if self.params.w is not None:
            weight_array = np.asfarray(self.params.w)
            fn_args['weights'] = cuda.to_device(weight_array, stream=stream)
        return fn_args


class Jaccard(BooleanMetric):
    NAMES = ('jaccard', )
    VARS = ('NNEQ', 'NNZ')
    CALC_DIST = '''
            if NNZ == 0:
                distance = 0
            else:
                distance = NNEQ / NNZ
    '''

class Equal(BooleanMetric):
    NAMES = ('equal', )
    VARS = ('NNEQ', )
    CALC_DIST = '''
            distance = int(NNEQ == 0)
    '''
    DIAGONAL_VAL = 1


class WeightTotalMetric(BooleanMetric):

    def get_fn_args(self, values, stream):
        fn_args = super().get_fn_args(values, stream)
        if 'weights' in fn_args:
            sum_weight = np.sum(self.params.w)
        else:
            sum_weight = np.shape(values)[1]
        
        fn_args['weight_total'] = sum_weight
        return fn_args


class Hamming(WeightTotalMetric):
    NAMES = ('matching', 'hamming')
    VARS = ('NNEQ', )
    NEEDS_N = True
    CALC_DIST = '''distance = cuda.selp(N == 0, 0., NNEQ / weight_total)'''


class Dice(BooleanMetric):
    NAMES = ('dice',)
    VARS = ('NNEQ', 'NTT', 'NNZ')
    CALC_DIST = '''
            divisor = (NTT + NNZ)
            if divisor == 0:
                distance = !!NANVAL!!
            else:
                distance = NNEQ / divisor
    '''


class Kulsinski(WeightTotalMetric):
    NAMES = ('kulsinski',)
    VARS = ('NNEQ', 'NTT')
    CALC_DIST = '''
            divisor = (NNEQ + weight_total)
            if divisor == 0:
                distance = !!NANVAL!!
            else:
                distance = (NNEQ + weight_total - NTT) / divisor
    '''


class RogersTanimoto(WeightTotalMetric):
    NAMES = ('rogerstanimoto',)
    VARS = ('NNEQ',)
    CALC_DIST = '''
            divisor = (NNEQ + weight_total)
            if divisor == 0:
                distance = !!NANVAL!!
            else:
                distance = 2 * NNEQ / divisor
    '''


class RussellRao(WeightTotalMetric):
    NAMES = ('russellrao',)
    VARS = ('NTT', )
    NEEDS_N = True
    CALC_DIST = '''
            if N == 0:
                distance = !!NANVAL!!
            else:
                distance = (weight_total - NTT) / weight_total
    '''


class SokalMichener(WeightTotalMetric):
    NAMES = ('sokalmichener',)
    VARS = ('NNEQ', )
    CALC_DIST = '''
            divisor = (weight_total + NNEQ)
            if divisor == 0:
                distance = !!NANVAL!!
            else:
                distance = (2 * NNEQ) / divisor
    '''


class SokalSneath(BooleanMetric):
    NAMES = ('sokalsneath',)
    VARS = ('NNEQ', 'NTT')
    CALC_DIST = '''
            divisor = (NNEQ + 0.5 * NTT)
            if divisor == 0:
                distance = !!NANVAL!!
            else:
                distance = NNEQ / divisor
    '''


