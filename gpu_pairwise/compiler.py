from jinja2 import Template, StrictUndefined
from numba import cuda
import numpy as np
import math


_FN_CACHE = {}
_JINJA_CACHE = {}

PLACEHOLDER_NAME = "!!!FN_NAME!!!"


def _type_sig(ob):
    if isinstance(ob, cuda.devicearray.DeviceNDArray):
        base_type = str(ob.dtype)
        if base_type == 'bool':
            base_type = 'bool_'
        array_spec = ", ".join([":"] * ob.ndim)
        return f'{base_type}[{array_spec}]'
    elif isinstance(ob, int):
        return 'int64'
    elif isinstance(ob, float):
        return 'float64'
    elif isinstance(ob, np.number):
        return str(np.dtype(ob))
    else:
        raise ValueError('Unsupported argument type')


def type_sig(*obs):
    return ", ".join(_type_sig(o) for o in obs)


def _get_template(template):
    if template not in _JINJA_CACHE:        
        _JINJA_CACHE[template] = Template(template, undefined=StrictUndefined)
    return _JINJA_CACHE[template]


def _compile_fn(nominal_name, code):
    temp_local = {}
    body = code.replace(PLACEHOLDER_NAME, nominal_name)    
    exec(compile(body, "<numba fn>", 'exec'), globals(), temp_local)
    #import ipdb; ipdb.set_trace()
    return temp_local[nominal_name]


def make_fn(nominal_name, template, **vars):
    compiled_template = _get_template(template)
    prototype = compiled_template.render(
        fn_name=PLACEHOLDER_NAME, 
        type_sig=type_sig,
        **vars
    )
    if prototype not in _FN_CACHE:
        _FN_CACHE[prototype] = _compile_fn(nominal_name, prototype)
    return _FN_CACHE[prototype]
