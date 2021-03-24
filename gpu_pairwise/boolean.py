import numpy as np
from numba import cuda

if not cuda.is_available():
    raise ImportError('gpu_pairwise requires a CUDA compatible device, and correct CUDA installation')


def _pack64(values):
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
    'NFF': '~(from_val & to_val)',
    'NNEQ': 'from_val ^ to_val',
    'NNZ': 'from_val | to_val',
}


PACKED_TEMPLATE = """
@cuda.jit('void(uint64[:, :], !!OUTTYPE!![:, :], uint64)')
def !!NAME!!(values, out, N):
    rows, cols = values.shape
    row_from, row_to = cuda.grid(2)
    
!!VARS!!
    
    if row_from < rows and row_to < rows:
        for col in range(cols):
            from_val = values[row_from, col]
            to_val = values[row_to, col]
!!CALC_FOR_CELL!!
        
!!CALC_DIST!!
        out[row_from, row_to] = distance
"""


TEMPLATE = """
@cuda.jit('void(uint8[:, :], float32[:], !!OUTTYPE!![:, :])')
def !!NAME!!(values, weights, out):
    rows, N = values.shape
    row_from, row_to = cuda.grid(2)
    
!!VARS!!

    if row_from < rows and row_to < rows:
        for col in range(N):
            from_val = values[row_from, col]
            to_val = values[row_to, col]
            weight = weights[col]
!!CALC_FOR_CELL!!
        
!!CALC_DIST!!
        out[row_from, row_to] = distance
"""

class Metric:
    NAMES = ()
    VARS = NotImplemented
    CALC_DIST = NotImplemented

    OUT_DTYPES = {
        'float32': (1, np.float32),
        'float64': (1, np.float64),
        'uint8': (255, np.uint8),
        'uint16': (65535, np.uint16),
    }

    @classmethod
    def get_cuda_fn(cls, weights, out_dtype):
        key = f'__fn_{"weight" if weights else "packed"}_{out_dtype}'
        if key not in cls.__dict__:
            template = TEMPLATE if weights else PACKED_TEMPLATE
            setattr(cls, key, cls._make_fn(weights, template, out_dtype))
        return cls.__dict__[key]

    @classmethod
    def _make_cell_calc(cls, have_weights, var, logic):
        if have_weights:
            return f'            {var} += ({logic}) * weight'
        else:
            return f'            {var} += cuda.popc({logic})'

    @classmethod
    def _calc_dist(cls, have_weights, out_scale_factor):
        if out_scale_factor != 1:
            return cls.CALC_DIST + f'\n        distance *= {out_scale_factor}'
        return cls.CALC_DIST

    @classmethod
    def _make_fn(cls, have_weights, template, out_dtype):
        out_scale_factor, _ = cls.OUT_DTYPES[out_dtype]
        fn_name = f'kernel_{cls.NAMES[0]}_{"weight" if have_weights else "packed"}_{out_dtype}'

        vars_defs = [f'    {v} = 0' for v in cls.VARS]
        calc_defs = [cls._make_cell_calc(have_weights, v, STATS_VARS[v]) for v in cls.VARS]
        fn_def = template.replace('!!NAME!!', fn_name)
        fn_def = fn_def.replace('!!VARS!!', "\n".join(vars_defs))
        fn_def = fn_def.replace('!!CALC_FOR_CELL!!', "\n".join(calc_defs))
        fn_def = fn_def.replace('!!CALC_DIST!!', cls._calc_dist(have_weights, out_scale_factor))
        fn_def = fn_def.replace('!!OUTTYPE!!', out_dtype)
        fn_def = fn_def.replace('!!NANVAL!!', "np.nan" if out_scale_factor == 1 else "0")

        temp_local = {}
        exec(compile(fn_def, __file__, 'single'), globals(), temp_local)
        return temp_local[fn_name]

    @classmethod
    def pairwise_distance(cls, values, weights=None, block_size=(16, 16), out_dtype='float32'):
        _, out_dtype_ob = cls.OUT_DTYPES[out_dtype]
        values = np.asarray(values, dtype=np.bool_)
        rows, cols = values.shape

        block_x, block_y = block_size
        grid_dim = (rows // block_x + 1, rows //block_y + 1)
        stream = cuda.stream()
        distances = cuda.device_array((rows, rows), dtype=out_dtype_ob)

        if weights is None:
            fn = cls.get_cuda_fn(False, out_dtype)
            packed = _pack64(values)
            packed_mat = cuda.to_device(packed, stream=stream)
            fn[grid_dim, block_size](packed_mat, distances, cols)
        else:
            fn = cls.get_cuda_fn(True, out_dtype)
            values_np = values.astype(np.uint8)
            values_mat = cuda.to_device(values_np, stream=stream)
            weight_np = np.broadcast_to(np.asarray(weights, dtype=np.float32), (cols, ))
            weight_mat = cuda.to_device(weight_np, stream=stream)
            fn[grid_dim, block_size](values_mat, weight_mat, distances)

        cuda.synchronize()
        distances_np = distances.copy_to_host(stream=stream)
        return distances_np

    @classmethod
    def _walk_subclasses(cls):
        yield cls
        for sub in cls.__subclasses__():
            yield from sub._walk_subclasses()

    @classmethod
    def get_metric(cls, name):
        try:
            match, = [m for m in Metric._walk_subclasses() if name in m.NAMES]
        except ValueError:
            raise KeyError(f'Unknown metric: {name}')
        return match


class Jaccard(Metric):
    NAMES = ('jaccard', )
    VARS = ('NNEQ', 'NNZ')
    CALC_DIST = '''
        if NNZ == 0:
            distance = 0
        else:
            distance = NNEQ / NNZ
    '''

class WeightTotalMetric(Metric):

    @classmethod
    def _calc_dist(cls, have_weights, out_scale_factor):
        if have_weights:
            prefix = """
        weight_total = 0
        for col in range(N):
            weight_total += weights[col]
            """
        else:
            prefix = """
        weight_total = N
            """
        return prefix + super(WeightTotalMetric, cls)._calc_dist(have_weights, out_scale_factor)


class Hamming(WeightTotalMetric):
    NAMES = ('matching', 'hamming')
    VARS = ('NNEQ', )
    CALC_DIST = '''
        if N == 0:
            distance = 0
        else:
            distance = NNEQ / weight_total
    '''


class Dice(Metric):
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
    VARS = ('NNZ', )
    VARS = ('NTT', )
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


class SokalSneath(Metric):
    NAMES = ('sokalsneath',)
    VARS = ('NNEQ', 'NTT')
    CALC_DIST = '''
        divisor = (NNEQ + 0.5 * NTT)
        if divisor == 0:
            distance = !!NANVAL!!
        else:
            distance = NNEQ / divisor
    '''


def pairwise_distance(values, metric='jaccard', weights=None, block_size=(16, 16), out_dtype='float32'):
    metric_cls = Metric.get_metric(metric)
    return metric_cls.pairwise_distance(values, weights=weights, block_size=block_size, out_dtype=out_dtype)