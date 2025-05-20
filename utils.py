from torch.optim.lr_scheduler import LambdaLR
from torch.nn import Linear
from copy import copy
from models.utils import MuReadout

import math


from time import perf_counter
from contextlib import contextmanager

@contextmanager
def catchtime():
    t1 = t2 = perf_counter() 
    yield lambda: t2 - t1
    t2 = perf_counter() 


class InfDim:
    '''A dimension with a base dimension, used for calculating μP scaling.

    An `InfDim` object is made up of 2 numbers: a dimension and a base
    dimension. If the base dimension is None, then this object represents a
    "finite", or "non-width" dimension. Otherwise, it represents an "infinite",
    or "width" dimension.
    '''

    def __init__(self, base_dim, dim):
        self.base_dim = base_dim
        self.dim = dim

    def isinf(self):
        return self.base_dim is not None

    def width_mult(self):
        '''Width multiplier used for calculating μP scaling.

        If finite, return 1.
        If infinite, return dim / base_dim.
        '''
        if self.isinf():
            return self.dim / self.base_dim
        return 1

    def __repr__(self):
        return f'InfDim({self.base_dim}, {self.dim})'

    def __str__(self):
        if self.isinf():
            return repr(self)
        return f'FinDim({self.dim})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfDim):
            return False
        return self.base_dim == other.base_dim and \
                self.dim == other.dim

class InfShape(tuple):
    '''A tuple of `InfDim`s.

    This is intended to be attached to each parameter tensor `p` as `p.infshape`.
    '''

    def __init__(self, *args, **kwargs):
        tuple.__init__(*args, **kwargs)
        for dim in self:
            if not isinstance(dim, InfDim):
                raise ValueError('Elements of InfShape needs to be of class InfDim')
        # set main to be the last dimension that is infinite
        # for inf x inf this is fanin
        # for inf x fin or fin x inf it's the unique inf dim
        # user can set this manually if necessary
        self.main_idx = self.main = None
        for i, dim in list(enumerate(self))[::-1]:
            if dim.isinf():
                self.main_idx = i
                self.main = dim
                break

    def ninf(self):
        return sum(1 for dim in self if dim.isinf())

    def width_mult(self):
        if self.main is not None:
            return self.main.width_mult()
        return 1

    def shape(self):
        return [d.dim for d in self]

    def __repr__(self):
        r = tuple.__repr__(self)[1:-1]
        return f'InfShape([{r}])'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfShape):
            return False
        return all(d == dd for d, dd in zip(self, other))

def get_shapes(model):
    return {name: param.shape for name, param in model.named_parameters()}

def _extract_shapes(x):
    x_shapes = get_shapes(x)
    return x_shapes

def zip_infshape(base_dims, dims, fin_if_same=True):
    infshape = []
    for bd, d in zip(base_dims, dims):
        if isinstance(bd, InfDim):
            # retain bd's base_dim but overwrite dim
            infdim = copy(bd)
            infdim.dim = d
            infshape.append(infdim)
        elif isinstance(bd, int):
            if bd == d and fin_if_same:
                infshape.append(InfDim(None, d))
            else:
                infshape.append(InfDim(bd, d))
        else:
            raise ValueError(f'unhandled base_dim type: {type(bd)}')
    return InfShape(infshape)

def _zip_infshape_dict(base_shapes, shapes):
    basenames = set(base_shapes.keys())
    names = set(shapes.keys())
    assert basenames == names, (
        f'`base_shapes` has extra names {basenames - names}. '
        f'`shapes` has extra names {names - basenames}.'
    )
    infshapes = {}
    for name, bsh in base_shapes.items():
        infshapes[name] = zip_infshape(bsh, shapes[name])
    return infshapes

def apply_infshapes(model, infshapes):
    for name, p in model.named_parameters():
        if 'cls_projection' in name:
            pass
            # print(name, infshapes[name], infshapes[name].width_mult())
        p.infshape = infshapes[name]

def compute_and_set_base_shapes(model, base, do_assert = True):
    base_shapes = _extract_shapes(base)
    shapes = get_shapes(model)
    infshapes = _zip_infshape_dict(base_shapes, shapes)

    apply_infshapes(model, infshapes)
    if do_assert:
        assert_hidden_size_inf(model)

    for _, module in model.named_modules():
        if isinstance(module, MuReadout):
            module._rescale_parameters()

    return model

def assert_hidden_size_inf(model):
    '''
    This tests for any `nn.Linear` whose output dimension is finite but input
    dimension is infinite and is not of type `MuReadout`. Such `nn.Linear`
    modules should not exist in a correctly parametrized models.
    '''
    for name, module in model.named_modules():
        if isinstance(module, Linear) and not isinstance(module, MuReadout):
            if not module.weight.infshape[0].isinf() and module.weight.infshape[1].isinf():
                assert False, (
                    f'{name} has infinite fan-in and finite fan-out dimensions but is not type `MuReadout`. '
                    'To resolve this, either change the module to `MuReadout` or change the fan-out to an infinite dimension.'
                )

def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles = 0.5,
                                    num_warmup_steps=0,
                                    last_epoch=-1):

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


