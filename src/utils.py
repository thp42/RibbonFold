import os, sys, pickle, time, random, gzip, io, collections, numbers, math, functools
import functorch
import torch

from .common_modules import large_value, small_value



def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
    """Masked mean."""
    if drop_mask_channel:
        mask = mask[..., 0]
    
    value = value.float()
    
    mask_shape = mask.shape
    value_shape = value.shape
    
    assert len(mask_shape) == len(value_shape)
    
    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    assert isinstance(axis, collections.abc.Iterable), ('axis needs to be either an iterable, integer or "None"')
    
    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size
    
    eps = small_value(eps)
    return (torch.sum(mask * value, axis=axis) / (torch.sum(mask, axis=axis) * broadcast_factor + eps))

def batch_take(params, dim, indices):
    """
    Like jax.numpy.take
    """    
    old_shape = list(indices.shape)
    inp_shape = list(params.shape)
    if dim < 0:
        dim = params.ndim + dim
    to_shape = inp_shape[:dim] + old_shape + inp_shape[dim+1:]
    indices = indices.flatten()
    
    out = torch.index_select(params, dim, indices)
    return out.reshape(to_shape)

def batched_gather(params, indices, dim=0, batch_dims=0):
    """
    Like alphafold.model.utils.batched_gather
    """
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params)
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices)
    indices = indices.long().to(params.device)

    ### Don't raise error when the indices exceed the dimension
    ### Compatible with jax: jnp.take(p, i, axis=axis)
    if indices.max() >= params.shape[dim]:
        padding = [0] * params.ndim * 2
        padding[-2*dim-1] = indices.max() - params.shape[dim] + 1
        params = torch.nn.functional.pad(params, padding)

    take_fn = lambda p, i: batch_take(p, indices=i, dim=dim)
    for _ in range(batch_dims):
        take_fn = functorch.vmap(take_fn)
    return take_fn(params, indices)


def batched_gather_new(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

from .pytree import tree_map, tree_multimap

def main_print(*args, sep=' ', end='\n', file=sys.stdout, flush=False):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, sep=sep, end=end, file=file, flush=flush)

def get_format_time():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def read_sequence_from_fasta(fastafile):
    sequences = []
    seq = None
    for line in open(fastafile):
        if line.startswith('>'):
            if seq is not None:
                sequences.append(seq)
            seq = ''
        else:
            seq += line.strip()
    sequences.append(seq)
    sequences = list(set(sequences))
    return sequences[0]

def ddp_gather_pyobject(py_object, all_gather=False):
    """
    Gather python object (int, float or other objects) to all processes or only rank 0
    Look (https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_object) for more information.

    Parameters
    --------------
    py_object: python object
    all_gather: bool.
        True means all rank got same full py_object list
        False means only rank 0 get full py_object list

    Return
    --------------
    py_object_list: None or list of py_object

    Notice: Torch cuda tensor is convert to cpu
    """
    def _convert_torch_cuda_to_cpu(x):
        return x.detach().cpu() if isinstance(x, torch.Tensor) else x
    py_object = torch.utils._pytree.tree_map(_convert_torch_cuda_to_cpu, py_object)
    if torch.distributed.is_initialized():
        py_object_list = [None] * torch.distributed.get_world_size()
        # Distribute object from all processes to another processes
        if all_gather:
            torch.distributed.all_gather_object(py_object_list, py_object)
        else:
            py_object_list = py_object_list if torch.distributed.get_rank() == 0 else None
            torch.distributed.gather_object(py_object, py_object_list, dst=0)
        return py_object_list
    else:
        return [py_object]

def atom_line(idx: int, atom_name: str, restype: str, chain: str, res_index: int,
              x: float, y: float, z: float, occ: float = 0.0, temp: float = 0.0):
    assert len(atom_name) <= 4
    assert len(restype) == 3
    assert len(chain) == 1
    assert len(str(res_index)) <= 4
    atom_mark = atom_name[0]
    return f"ATOM  {idx:5d} {atom_name:4s} {restype} {chain}{res_index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}           {atom_mark:1s}"

