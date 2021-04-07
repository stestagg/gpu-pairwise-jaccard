import numpy as np
from dataclasses import dataclass


@dataclass
class DtypeInfo:
    names: tuple
    cuda_name: str
    dtype: np.generic
    unit_scale: float = 1
    has_nan: bool = False

    _REGISTRY = {}

    @classmethod
    def by_name(cls, name):
        return cls._REGISTRY[name]

    @classmethod
    def _register(cls, names, *args, **kwargs):
        if isinstance(names, str):
            names = (names, )
        inst = DtypeInfo(names, *args, **kwargs)
        for name in names:
            cls._REGISTRY[name] = inst

    def n_dim(self, n):
        dims = ", ".join(":")
        return f'{self.cuda_name}[{dims}]'


DtypeInfo._register('float32', 'float32', np.float32, has_nan=True)
DtypeInfo._register(('float', 'float64', 'double'), 'float64', np.float64, has_nan=True)
DtypeInfo._register(('bool', ), 'bool_', np.bool_)
DtypeInfo._register(('uint8', ), 'uint8', np.uint8, unit_scale=255)
DtypeInfo._register(('uint16', ), 'uint16', np.uint16, unit_scale=65535)
DtypeInfo._register(('uint32', ), 'uint32', np.uint32, unit_scale=4294967295)
DtypeInfo._register(('uint64', ), 'uint64', np.uint64, unit_scale=18446744073709551615)