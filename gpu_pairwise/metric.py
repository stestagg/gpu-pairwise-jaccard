from dataclasses import dataclass, asdict
from typing import Sequence
from functools import partial

import numpy as np
from numba import cuda

from .dtype import DtypeInfo
from .compiler import make_fn


class Metric:
    _METRIC_REGISTRY = {}
    NAMES = ()    
    TEMPLATE = NotImplemented

    @dataclass
    class Params:
        block_size: Sequence[int] = (16, 16)
        out_dtype: str = 'double'

    def __init__(self, **kwargs):
        self.params = self.Params(**kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        registry = Metric._METRIC_REGISTRY
        for name in cls.NAMES:
            if name in registry and registry[name] is not cls:
                raise ValueError(f'Multiple metrics for name: {name}')
            registry[name] = cls

    @classmethod
    def get_metric(cls, name):
        return Metric._METRIC_REGISTRY[name]

    def specialize(self, more_params):
        base_params = asdict(self.params)
        base_params.update(more_params)
        return type(self)(**base_params)
    
    @property
    def template(self):
        assert self.TEMPLATE is not NotImplemented
        return self.TEMPLATE

    def get_cuda_fn(self, args):
        fn_name = f'cuda_{type(self).__name__.lower()}_transform'
        return make_fn(
            fn_name, 
            self.template, 
            metric=self, 
            fn_args=args
        )

    @property
    def out_dtype(self):
        return DtypeInfo.by_name(self.params.out_dtype)

    def get_fn_args(self, values, stream):
        if isinstance(values, cuda.devicearray.DeviceNDArray):
            device_values = values
        else:
            float_array = np.asfarray(values)
            device_values = cuda.to_device(float_array, stream=stream)

        rows, _ = device_values.shape
        out_type = self.out_dtype
        out_size = (rows, rows)
        distances = cuda.device_array(out_size, dtype=out_type.dtype)

        return {
            'out': distances,
            'values': device_values,
        }

    def transform(self, values, **params):
        if params:
            return self.specialize(params).transform(values)

        stream = cuda.stream()
        fn_args = self.get_fn_args(values, stream)
        output = fn_args['out']
        fn = self.get_cuda_fn(fn_args)
        
        block_size = self.params.block_size
        rows, _ = fn_args['values'].shape

        block_x, block_y = block_size
        grid_dim = (rows // block_x + 1, rows //block_y + 1)
        fn[grid_dim, block_size](*fn_args.values())

        cuda.synchronize()
        distances_np = output.copy_to_host(stream=stream)
        return distances_np