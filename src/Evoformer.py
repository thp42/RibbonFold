from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from .checkpoint import checkpoint, checkpoint_sequential
import math
import numpy as np
import numbers, collections
from . import quat_affine, utils, residue_constants, all_atom
from .common_modules import *
import time
import datetime

USE_SUBBATCH = True
SUBBATCH_SIZE = 1024        # Increased from 128 for A100 (4x faster attention)
GLOBAL_SUBBATCH_SIZE = 16   # Increased from 1 for A100 (4x faster global ops)
RUN_LOW_MEMORY = False
DISABLE_TQDM = True

from .profiler import def_profiler as profiler

###############################
### Util functions
###############################

def create_extra_msa_feature(batch):
    """Expand extra_msa into 1hot and concat with other extra msa features.

    We do this as late as possible as the one_hot extra msa can be very large.

    Arguments:
    batch: a dictionary with the following keys:
     * 'extra_msa': [N_extra_seq, N_res] MSA that wasn't selected as a cluster
       centre. Note, that this is not one-hot encoded.
     * 'extra_has_deletion': [N_extra_seq, N_res] Whether there is a deletion to
       the left of each position in the extra MSA.
     * 'extra_deletion_value': [N_extra_seq, N_res] The number of deletions to
       the left of each position in the extra MSA.

    Returns:
    Concatenated tensor of extra MSA features.
    """
    # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
    msa_1hot = torch.nn.functional.one_hot(batch['extra_msa'], 23).to( batch['extra_deletion_value'] ) # [N_extra_seq, N_res, 23]
    msa_feat = [msa_1hot,
              torch.unsqueeze(batch['extra_has_deletion'], axis=-1), # [N_extra_seq, N_res, 1]
              torch.unsqueeze(batch['extra_deletion_value'], axis=-1)] # [N_extra_seq, N_res, 1]
    return torch.cat(msa_feat, dim=-1) # [N_extra_seq, N_res, 25]

def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return loss

def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    """Compute distogram from amino acid positions.
    
    Arguments:
    positions: [N_res, 3] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
        everything larger than `max_bin`.
    
    Returns:
    Distogram with the specified number of bins.
    """
    
    lower_breaks = torch.linspace(min_bin, max_bin, num_bins).to(positions)
    lower_breaks = torch.square(lower_breaks).to(positions)
    upper_breaks = torch.cat([lower_breaks[1:],torch.tensor([ large_value() ]).to(positions)], dim=-1)
    dist2 = torch.sum(torch.square(torch.unsqueeze(positions, dim=-2) - torch.unsqueeze(positions, dim=-3)), dim=-1, keepdims=True)
    dgram = ((dist2 > lower_breaks).to(positions) * (dist2 < upper_breaks).to(positions))
    return dgram


def dgram_from_pairs(pairs, num_bins, min_bin, max_bin):
    """Compute distogram from amino acid positions.

    Arguments:
    pairs: [N_res, N_res] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
        everything larger than `max_bin`.

    Returns:
    Distogram with the specified number of bins.
    """

    lower_breaks = torch.linspace(min_bin, max_bin, num_bins).to(pairs)
    lower_breaks = torch.square(lower_breaks).to(pairs)
    upper_breaks = torch.cat([lower_breaks[1:], torch.tensor([large_value()]).to(pairs)], dim=-1)
    dist2 = pairs.unsqueeze(-1)
    dgram = ((dist2 > lower_breaks).to(pairs) * (dist2 < upper_breaks).to(pairs))
    return dgram




def generate_random_unit_vector(device):
    vec = torch.randn(3).to(device)  
    return vec / vec.norm() 


def fill_random_vectors(distance_matrix, device):
    N_res = distance_matrix.size(0)
    result_matrix = torch.zeros((N_res, N_res, 3), device=device)  # 初始化结果矩阵
    non_zero_indices = torch.nonzero(distance_matrix, as_tuple=False)
    vec3 = generate_random_unit_vector(device)
    lower_triangle_indices = non_zero_indices[non_zero_indices[:, 0] < non_zero_indices[:, 1]]
    result_matrix[lower_triangle_indices[:, 0], lower_triangle_indices[:, 1]] = vec3
    upper_triangle_indices = non_zero_indices[non_zero_indices[:, 0] > non_zero_indices[:, 1]]
    result_matrix[upper_triangle_indices[:, 0], upper_triangle_indices[:, 1]] = -vec3

    return result_matrix



def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks=None):
    """
    Create pseudo beta features.
    
    aatype: [N_res]
    all_atom_positions: [N_res, 37, 3]
    """
    
    restype_order = residue_constants.restype_order
    atom_order = residue_constants.atom_order
    
    assert aatype.ndim == 1
    assert all_atom_positions.ndim == 3
    assert aatype.shape[0] == all_atom_positions.shape[0]
    
    is_gly = (aatype[...,None] == torch.tensor([restype_order['G']]).to(aatype.device))
    ca_idx = torch.tensor([atom_order['CA']]).to(aatype.device)
    cb_idx = torch.tensor([atom_order['CB']]).to(aatype.device)
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * is_gly.ndim + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :]) # [N_res, 1, 3]
    pseudo_beta = pseudo_beta.squeeze(1) # [N_res, 3]

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.to(aatype)
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta

# def sigmoid(tensor):
#     output = torch.sigmoid(tensor)
#     if torch.is_autocast_enabled():
#         print("Old dtype:", output.dtype)
#         output = output.to( torch.get_autocast_gpu_dtype() )
#         print("New dtype:", output.dtype)
#     return output

def amp_safe_add(tensor1, tensor2):
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_gpu_dtype()
        out = tensor1.to(dtype) + tensor2.to(dtype)
    else:
        out = tensor1 + tensor2
    return out



###############################
### Sub-batch
###############################

def tensor_slice(array, i, slice_size, axis):
    result = torch.index_select(array, dim=axis, index=torch.arange(i, i+slice_size).to(array.device))
    return result

def sharded_apply(func, output_shape, subbatch_size=128, in_axes=0, out_axes=0):
    def mapped_fn(*args):
        flat_sizes = [ arg.shape[in_axes] for arg in args ] # 有维度的为shape大小
        in_size = max(flat_sizes) # 最大的维度
        assert all(i == in_size for i in flat_sizes)
        
        last_subbatch_size = in_size % subbatch_size 
        last_subbatch_size = subbatch_size if last_subbatch_size == 0 else last_subbatch_size # 最后一片的大小
        
        def apply_func_to_slice(slice_start, slice_size):
            input_slice = [ tensor_slice(arg, slice_start, slice_size, in_axes) for arg in args ]
            return func(*input_slice)
        
        outputs = torch.zeros(output_shape, dtype=args[0].dtype, device=args[0].device)
        
        def compute_subbatch(outputs, slice_start, slice_size):
            slice_out = apply_func_to_slice(slice_start, slice_size).to(outputs)
            outputs.index_copy_(out_axes, torch.arange(slice_start, slice_start+slice_size).to(outputs.device), slice_out)
        
        for i in range(0, in_size - subbatch_size + 1, subbatch_size):
            compute_subbatch(outputs, i, subbatch_size) 
        
        if last_subbatch_size != subbatch_size:
            remainder_start = in_size - last_subbatch_size
            compute_subbatch(outputs, remainder_start, last_subbatch_size)
        
        return outputs
    return mapped_fn

def inference_subbatch(module, 
                       output_shape, 
                       subbatch_size=128, 
                       batched_args=[], 
                       nonbatched_args=[], 
                       low_memory=True, 
                       input_subbatch_dim=0, 
                       output_subbatch_dim=None):
    """
    Run through subbatches (like batch apply but with split and concat).
    """
    assert len(batched_args) > 0
    
    if not low_memory or batched_args[0].shape[input_subbatch_dim] <= subbatch_size:
        args = list(batched_args) + list(nonbatched_args)
        return module(*args)
    
    if output_subbatch_dim is None:
        output_subbatch_dim = input_subbatch_dim
    
    def run_module(*batched_args):
        args = list(batched_args) + list(nonbatched_args)
        return module(*args)
    
    sharded_module = sharded_apply(run_module,
                                 output_shape=output_shape,
                                 subbatch_size=subbatch_size,
                                 in_axes=input_subbatch_dim,
                                 out_axes=output_subbatch_dim)
    
    return sharded_module(*batched_args)


###############################
### Basic layers
###############################

COUNT = 0

class Attention(nn.Module):
    """
    Multihead attention
    """
    
    def __init__(self, query_dim, memo_dim, output_dim, num_head, 
        key_dim=None, value_dim=None, gating=True, zero_init=True):
        super().__init__()
        
        if key_dim is None:
            key_dim = query_dim
        if value_dim is None:
            value_dim = memo_dim
        
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        
        self.key_dim = key_dim // num_head
        self.value_dim = value_dim // num_head
        self.query_dim = query_dim
        self.memo_dim = memo_dim
        self.output_dim = output_dim
        self.num_head = num_head
        self.gating = gating
        self.zero_init = zero_init
        #self.inner_checkpoint = inner_checkpoint
        
        self.query_w = nn.Parameter(torch.randn(query_dim, self.num_head, self.key_dim))
        self.key_w = nn.Parameter(torch.randn(memo_dim, self.num_head, self.key_dim))
        self.value_w = nn.Parameter(torch.randn(memo_dim, self.num_head, self.value_dim))
        
        VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')(self.query_w)
        VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')(self.key_w)
        VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')(self.value_w)
        
        if gating:
            self.gating_w = nn.Parameter(torch.zeros(query_dim, self.num_head, self.value_dim))
            self.gating_b = nn.Parameter(torch.ones(self.num_head, self.value_dim))
        
        self.output_w = nn.Parameter(torch.randn(self.num_head, self.value_dim, self.output_dim))
        self.output_b = nn.Parameter(torch.zeros(self.output_dim))
        if zero_init:
            torch.nn.init.zeros_(self.output_w)
        else:
            VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')(self.output_w)
    
    def forward(self, q_data, m_data, bias, nonbatched_bias=None):
        """
        q_data: queries,   [B, N_q, E_q]
        m_data: memories,  [B, N_k, E_k]
        bias: bias,        [B, 1, N_q, N_k]
        nonbatched_bias: Shared bias, [N_queries, N_keys].
        """
        assert q_data.ndim == m_data.ndim == 3
        assert bias.ndim == 4
        assert q_data.shape[2] == self.query_dim
        assert m_data.shape[2] == self.memo_dim
        
        q = torch.einsum('bqa,ahc->bqhc', q_data, self.query_w) * self.key_dim**(-0.5) # [B, N_q, H, K_dim]
        k = torch.einsum('bka,ahc->bkhc', m_data, self.key_w) # [B, N_k, H, K_dim]
        v = torch.einsum('bka,ahc->bkhc', m_data, self.value_w) # [B, N_k, H, V_dim]
        
        logits = torch.einsum('bqhc,bkhc->bhqk', q, k) + bias     # [B, H,   N_q, N_k]
        if nonbatched_bias is not None:
            logits = logits + nonbatched_bias[None,...] #torch.expand_dims(nonbatched_bias, axis=0)
        
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
        #dtype = None
        #logits = logits.float()
        weights = F.softmax(logits, dim=-1, _stacklevel=3, dtype=dtype) # [B, H,   N_q, N_k]
        #weights = weights.to(logits)
        #print(logits.max(), logits.min())
        
        weighted_avg = torch.einsum('bhqk,bkhc->bqhc', weights, v) # [B, N_q, H, N_k]
        
        if self.gating:
            # [B, N_q, E_q] @ [E_q, H, V_dim] => [B, N_q, H, V_dim] [E_q, V_dim]
            gate_values = torch.einsum('bqc, chv->bqhv', q_data, self.gating_w) + self.gating_b
            gate_values = torch.sigmoid(gate_values)
            weighted_avg = weighted_avg * gate_values
        
        output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b # [B, q_dim, o_dim]
        return output

class GlobalAttention(nn.Module):
    def __init__(self, query_dim, memo_dim, output_dim, num_head, 
                key_dim=None, value_dim=None, gating=True, zero_init=True):
        super().__init__()
        
        if key_dim is None:
            key_dim = query_dim
        if value_dim is None:
            value_dim = memo_dim
        
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        
        self.key_dim = key_dim // num_head
        self.value_dim = value_dim // num_head
        self.query_dim = query_dim
        self.memo_dim = memo_dim
        self.output_dim = output_dim
        self.num_head = num_head
        self.gating = gating
        self.zero_init = zero_init
        
        self.query_w = nn.Parameter(torch.randn(query_dim, self.num_head, self.key_dim))
        self.key_w = nn.Parameter(torch.randn(memo_dim, self.key_dim)) # Here is different with Attention
        self.value_w = nn.Parameter(torch.randn(memo_dim, self.value_dim))
        
        VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')(self.query_w)
        VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')(self.key_w)
        VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')(self.value_w)
        
        if gating:
            self.gating_w = nn.Parameter(torch.zeros(query_dim, self.num_head, self.value_dim))
            self.gating_b = nn.Parameter(torch.ones(self.num_head, self.value_dim))
        
        self.output_w = nn.Parameter(torch.randn(self.num_head, self.value_dim, self.output_dim))
        self.output_b = nn.Parameter(torch.zeros(self.output_dim))
        if zero_init:
            torch.nn.init.zeros_(self.output_w)
        else:
            VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')(self.output_w)
    
    def forward(self, q_data, m_data, q_mask, bias):
        """
        q_data: queries, [batch_size, N_queries, q_channels]
        m_data: memories,[batch_size, N_keys, m_channels]
        q_mask: mask for q_data with zeros, [batch_size, N_queries, q_channels]
        
        Return:
        [batch_size, N_queries, output_dim]
        """
        """
        Parameters
        ------------
        q_data: [batch_size, N_queries, q_channels]
        m_data: [batch_size, N_keys,    m_channels]
        q_mask: [batch_size, N_queries, 1]
        
        Return
        ------------
        [batch_size, N_queries, output_dim]
        """
        assert q_data.ndim == m_data.ndim == q_mask.ndim == 3
        assert q_data.shape[2] == self.query_dim
        assert m_data.shape[2] == self.memo_dim
        
        v = torch.einsum('bka,ac->bkc', m_data, self.value_w) # [B, N_k, V_dim]
        q_avg = utils.mask_mean(q_mask, q_data, axis=1) # [batch_size, q_channels]
        
        q = torch.einsum('ba,ahc->bhc', q_avg, self.query_w) * self.key_dim**(-0.5) # [B, H, K_dim]
        k = torch.einsum('bka,ac->bkc', m_data, self.key_w) # [B, N_k, K_dim]
        bias = (large_value() * (q_mask[:, None, :, 0] - 1.)) # [batch_size, 1, N_k]
        
        logits = torch.einsum('bhc,bkc->bhk', q, k) + bias # [B, H, N_k]
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
        # dtype = torch.float32
        weights = torch.softmax(logits, -1, dtype=dtype) # [B, H, N_k]
        weighted_avg = torch.einsum('bhk,bkc->bhc', weights, v) # [B, H, V_dim]
        
        if self.gating:
            gate_values = torch.einsum('bqc, chv->bqhv', q_data, self.gating_w) # [B, N_q, H, V_dim]
            gate_values = torch.sigmoid(gate_values + self.gating_b) # [B, N_q, H, V_dim]
            weighted_avg = weighted_avg[:, None] * gate_values # [B, N_q, H, V_dim]
            output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b # [B, N_q, o_dim]
        else:
            output = torch.einsum('bhc,hco->bo', weighted_avg, self.output_w) + self.output_b # [B, o_dim]
            output = output[:, None] # [B, 1, o_dim]
        
        return output

class MSARowAttention(nn.Module):
    """
    MSA per-row attention biased by the pair representation.
    """
    def __init__(self,  input_dim=256, 
                        output_dim=256, 
                        num_head=8, 
                        gating=True, 
                        zero_init=True):
        super().__init__()
        self.input_dim = input_dim
        self.attention = Attention(  query_dim=input_dim, memo_dim=input_dim, output_dim=output_dim, 
                                    num_head=num_head, gating=gating, zero_init=zero_init)
        self.query_norm = LayerNorm(input_dim)
    
    def forward(self, msa_act, msa_mask):
        """
        msa_act:  [N_seq, N_res, c_m]
        msa_mask: [N_seq, N_res], 0 is the masked res
        """
        assert msa_act.ndim == 3
        assert msa_mask.ndim == 2
        
        bias = (large_value() * (msa_mask - 1.))[:, None, None, :] # [N_seq, 1, 1, N_seq]
        assert bias.ndim == 4
        bias = bias.to(msa_act.device)
        
        msa_act = self.query_norm(msa_act)

        if USE_SUBBATCH:
            msa_act = inference_subbatch(
                        module=self.attention, 
                        output_shape=msa_act.shape, 
                        subbatch_size=SUBBATCH_SIZE, 
                        batched_args=[msa_act, msa_act, bias], 
                        nonbatched_args=[], 
                        low_memory=not self.training)
        else:
            msa_act = self.attention(msa_act, msa_act, bias)
        
        return msa_act

def get_memory():
    return round(torch.cuda.memory_allocated() / (1024**3),3)

def estimate_tensor_memory(tensor):
    size = tensor.numel()
    if tensor.dtype == torch.float32:
        size *= 4
    elif tensor.dtype == torch.float16:
        size *= 2
    elif tensor.dtype == torch.bfloat16:
        size *= 2
    elif tensor.dtype == torch.float64:
        size *= 8
    else:
        raise RuntimeError("Unrecognized tensor")
    return round(size / (1024**3), 3)

def show_current_state(name, tensor, last_memory):
    memory_inc = get_memory() - last_memory
    est_memory = estimate_tensor_memory(tensor)
    info = f"{name:20s} -- Tensor: {list(tensor.shape)}({tensor.dtype}); Estimated memory: {est_memory:.3f}G; Actual memory: {memory_inc:.3f}G"
    print(info)
    return get_memory()
    
class MSARowAttentionWithPairBias(nn.Module):
    """
    MSA per-row attention biased by the pair representation.
    """
    def __init__(self,  msa_input_dim=256, 
                        pair_input_dim=128,
                        output_dim=256, 
                        num_head=8, 
                        gating=True, 
                        zero_init=True):
        super().__init__()
        self.msa_input_dim = msa_input_dim
        self.pair_input_dim = pair_input_dim
        self.attention = Attention(  query_dim=msa_input_dim, memo_dim=msa_input_dim,
                                    output_dim=output_dim, num_head=num_head, 
                                    gating=gating, zero_init=zero_init)
        self.query_norm = LayerNorm(msa_input_dim)
        self.feat_2d_norm = LayerNorm(pair_input_dim)
        
        self.feat_2d_weights = nn.Parameter(torch.randn(self.pair_input_dim, num_head))
        init_factor = 1. / np.sqrt( self.pair_input_dim )
        torch.nn.init.normal_(self.feat_2d_weights, std=init_factor)
    
    def forward(self, msa_act, msa_mask, pair_act):
        """
        msa_act:  [N_seq, N_res, c_m]
        msa_mask: [N_seq, N_res], 0 is the masked res
        pair_act: [N_res, N_res, c_z]
        """
        assert msa_act.ndim == pair_act.ndim == 3
        assert msa_mask.ndim == 2
        assert msa_act.shape[2] == self.msa_input_dim
        assert pair_act.shape[2] == self.pair_input_dim
        
        #last_mem = get_memory()
        
        bias = (large_value() * (msa_mask - 1.))[:, None, None, :] # [N_seq, 1, 1, N_res]
        #print("bias:", bias.dtype)
        assert bias.ndim == 4
        bias = bias.to(msa_act.device)
        #print("bias:", bias.dtype)
        
        #last_mem = show_current_state("bias", bias, last_mem)
        
        msa_act = self.query_norm(msa_act)     # [N_seq, N_res, c_m]
        
        #last_mem = show_current_state("msa_act", msa_act, last_mem)
        #print("msa_act:", msa_act.dtype)
        pair_act = self.feat_2d_norm(pair_act)  # [N_res, N_res, c_z]
        #print("pair_act:", pair_act.dtype)
        #last_mem = show_current_state("pair_act", pair_act, last_mem)
        
        nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights) # [H, N_res, N_res]
        #last_mem = show_current_state("nonbatched_bias", nonbatched_bias, last_mem)
        
        if USE_SUBBATCH:
            msa_act = inference_subbatch(
                        module=self.attention, 
                        output_shape=msa_act.shape, 
                        subbatch_size=GLOBAL_SUBBATCH_SIZE, 
                        batched_args=[msa_act, msa_act, bias], 
                        nonbatched_args=[nonbatched_bias], 
                        low_memory=not self.training)
        else:
            msa_act = self.attention(msa_act, msa_act, bias, nonbatched_bias) # [N_seq, N_res, c_m]
        
        #last_mem = show_current_state("msa_act", msa_act, last_mem)
        #print("msa_act:", msa_act.dtype)
        return msa_act

    
class MSAColumnAttention(nn.Module):
    """
    MSA per-column attention.
    """
    
    def __init__(self,  input_dim=256, 
                        output_dim=256, 
                        num_head=8, 
                        gating=True, 
                        zero_init=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.attention = Attention(  query_dim=input_dim, memo_dim=input_dim, output_dim=output_dim, 
                                    num_head=num_head, gating=gating, zero_init=zero_init)
        self.query_norm = LayerNorm(input_dim)
    
    def forward(self, msa_act, msa_mask):
        """Builds MSAColumnAttention module.
        
        Arguments:
          msa_act:  [N_seq, N_res, c_m]
          msa_mask: [N_seq, N_res], 0 is the masked res
        """
        assert msa_act.ndim == 3
        assert msa_mask.ndim == 2
        assert msa_act.shape[2] == self.input_dim
        
        msa_act = msa_act.permute([1,0,2]) # [N_res, N_seq, c_m]
        msa_mask = msa_mask.permute([1,0]) # [N_res, N_seq]
        
        bias = (large_value() * (msa_mask - 1.))[:, None, None, :]
        assert bias.ndim == 4
        bias = bias.to(msa_act.device)
        
        msa_act = self.query_norm(msa_act)

        if USE_SUBBATCH:
            msa_act = inference_subbatch(
                        module=self.attention, 
                        output_shape=msa_act.shape, 
                        subbatch_size=GLOBAL_SUBBATCH_SIZE, 
                        batched_args=[msa_act, msa_act, bias], 
                        nonbatched_args=[], 
                        low_memory=not self.training)
        else:
            msa_act = self.attention(msa_act, msa_act, bias)
        
        msa_act = msa_act.permute([1,0,2])
        
        return msa_act
    
class Transition(nn.Module):
    """Transition layer
    """
    def __init__(self, input_dim, num_intermediate_factor=4, zero_init=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_intermediate_factor = num_intermediate_factor
        self.zero_init = zero_init
        
        self.input_layer_norm = LayerNorm(input_dim)
        
        self.transition1 = Linear(input_dim, input_dim * num_intermediate_factor, initializer='relu')
        self.transition2 = Linear(input_dim * num_intermediate_factor, input_dim)
        
        if zero_init:
            torch.nn.init.zeros_(self.transition2.weights)
        
        self.transition_module = nn.Sequential(
            self.transition1,
            nn.ReLU(),
            self.transition2
        )
    
    def forward(self, act, mask):
        """Builds Transition module.
        
        Arguments:
          act:  [batch_size, N_res, N_channel]
          mask: [batch_size, N_res]
        """
        assert act.ndim == 3
        assert mask.ndim == 2
        assert act.shape[2] == self.input_dim
        
        mask = torch.unsqueeze(mask, axis=-1)
        
        act = self.input_layer_norm(act)
        
        if USE_SUBBATCH:
            act = inference_subbatch(
                module=self.transition_module,
                output_shape=act.shape,
                subbatch_size=4,
                batched_args=[act],
                nonbatched_args=[],
                low_memory=not self.training)
        else:
            act = self.transition_module(act)

        return act

class OuterProductMean(nn.Module):
    """
    Computes mean outer product.
    """
    def __init__(self, input_dim, output_dim, num_outer_channel=32, zero_init=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_outer_channel  = num_outer_channel
        self.zero_init = zero_init
        
        self.layer_norm_input = LayerNorm(input_dim)
        self.left_projection = Linear(input_dim, num_outer_channel)
        self.right_projection = Linear(input_dim, num_outer_channel)
        
        VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')( self.left_projection.weights )
        VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')( self.right_projection.weights )
        
        self.output_w = nn.Parameter(torch.randn(num_outer_channel, num_outer_channel, output_dim)) # [num_outer_channel, num_outer_channel, output_dim]
        self.output_b = nn.Parameter(torch.zeros(output_dim))
        
        if self.zero_init:
            torch.nn.init.zeros_(self.output_w)
        else:
            VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')( self.output_w )
    
    def forward(self, act, mask):
        """
        Builds OuterProductMean module.
        
        Arguments:
          act:  [N_seq, N_res, c_m].
          mask: [N_seq, N_res], 0 is the masked res
        """
        assert act.ndim == 3
        assert mask.ndim == 2
        assert act.shape[2] == self.input_dim
        N_seq, N_res, c_m = act.shape
        
        mask = mask[..., None]
        
        act = self.layer_norm_input(act)
        
        left_act = mask * self.left_projection(act)   # [N_seq, N_res, num_outer_channel]
        right_act = mask * self.right_projection(act) # [N_seq, N_res, num_outer_channel]
        
        def compute_chunk(left_act):
            # This is equivalent to
            #
            # act = jnp.einsum('abc,ade->dceb', left_act, right_act)
            # act = jnp.einsum('dceb,cef->bdf', act, output_w) + output_b
            #
            # but faster.
            left_act = left_act.permute([0,2,1]) # [N_seq, num_outer_channel, N_res]
            act = torch.einsum('acb,ade->dceb', left_act, right_act) # [N_res, num_outer_channel, num_outer_channel, N_res]
            act = torch.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b # [N_res, N_res, output_dim]
            act = act.to(left_act)
            return act.permute([1, 0, 2]) # [N_res, N_res, output_dim]
        
        if USE_SUBBATCH:
            act = inference_subbatch(
                module=compute_chunk,
                output_shape=[N_res, N_res, self.output_dim],
                subbatch_size=SUBBATCH_SIZE,
                batched_args=[left_act],
                nonbatched_args=[],
                low_memory=True,
                input_subbatch_dim=1,
                output_subbatch_dim=0)
        else:
            act = compute_chunk(left_act)
        
        epsilon = 1e-3
        norm = torch.einsum('abc,adc->bdc', mask, mask)
        act /= epsilon + norm # [N_res, N_res, output_dim]
        return act

class MSAColumnGlobalAttention(nn.Module):
    """MSA per-column global attention.
    
    Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention"
    """
    def __init__(self, input_dim, output_dim, num_head, gating=True, zero_init=True):
        super().__init__()
        self.input_dim = input_dim
        self.query_norm = LayerNorm(input_dim)
        self.attention = GlobalAttention(query_dim=input_dim, memo_dim=input_dim, output_dim=output_dim, 
                                        num_head=num_head, gating=gating, zero_init=zero_init)
    
    def forward(self, msa_act, msa_mask):
        """
        msa_act: [N_seq, N_res, c_m]
        msa_mask: [N_seq, N_res]
        """
        assert msa_act.ndim == 3
        assert msa_mask.ndim == 2
        assert msa_act.shape[2] == self.input_dim
        
        msa_act = msa_act.permute([1,0,2]) # [N_res, N_seq, c_m]
        msa_mask = msa_mask.permute([1,0]) # [N_res, N_seq]
        
        bias = (large_value() * (msa_mask - 1.))[:, None, None, :] # [N_res, 1, 1, N_seq]
        assert bias.ndim == 4
        
        msa_act = self.query_norm(msa_act) # [N_res, N_seq, c_m]
        msa_mask = msa_mask[..., None] # [N_res, N_seq, 1]
        
        if USE_SUBBATCH:
            msa_act = inference_subbatch(
                        module=self.attention, 
                        output_shape=msa_act.shape, 
                        subbatch_size=GLOBAL_SUBBATCH_SIZE, 
                        batched_args=[msa_act, msa_act, msa_mask, bias], 
                        nonbatched_args=[], 
                        low_memory=not self.training)
        else:
            msa_act = self.attention(msa_act, msa_act, msa_mask, bias) # [N_res, N_seq, c_m]
        msa_act = msa_act.permute([1,0,2]) # [N_seq, N_res, c_m]
        
        return msa_act

class TriangleAttention(nn.Module):
    """Triangle Attention.
    
    Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
    Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
    """
    def __init__(self, input_dim=128, output_dim=128, 
                        key_dim=None, value_dim=None,
                        num_head=4, gating=True, 
                        zero_init=True, orientation='per_row'):
        """
        orientation: str
            per_row -- triangle_attention_starting_node
            per_column -- triangle_attention_ending_node
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_head = num_head
        assert orientation in ['per_row', 'per_column']
        self.orientation = orientation
        self.query_norm = LayerNorm(input_dim)
        
        self.feat_2d_weights = nn.Parameter(torch.randn(input_dim, num_head))
        init_factor = 1. / np.sqrt( self.input_dim )
        torch.nn.init.normal_(self.feat_2d_weights, std=init_factor)
        
        if key_dim is None:
            key_dim = input_dim
        if value_dim is None:
            value_dim = input_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.attention = Attention(  query_dim=input_dim, memo_dim=input_dim, 
                                    key_dim=key_dim, value_dim=value_dim,
                                    output_dim=output_dim, num_head=num_head, 
                                    gating=gating, zero_init=zero_init)
    
    def forward(self, pair_act, pair_mask):
        """
        pair_act:  [N_res, N_res, c_z]
        pair_mask: [N_res, N_res]
        """
        assert pair_act.ndim == 3
        assert pair_mask.ndim == 2
        assert pair_act.shape[2] == self.input_dim
        
        if self.orientation == 'per_column':
            pair_act  = pair_act.permute([1,0,2]) # [N_res, N_res, c_z]
            pair_mask = pair_mask.permute([1,0])  # [N_res, N_res]
        
        bias = (large_value() * (pair_mask - 1.))[:, None, None, :] # [N_res, 1, 1, N_res]
        assert bias.ndim == 4
        
        pair_act = self.query_norm(pair_act) # [N_res, N_res, c_z]
        nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights) # [H, N_res, N_res]
        
        if USE_SUBBATCH:
            pair_act = inference_subbatch(
                        module=self.attention, 
                        output_shape=pair_act.shape, 
                        subbatch_size=GLOBAL_SUBBATCH_SIZE, 
                        batched_args=[pair_act, pair_act, bias], 
                        nonbatched_args=[nonbatched_bias], 
                        low_memory=not self.training)
        else:
            pair_act = self.attention(pair_act, pair_act, bias, nonbatched_bias)
        
        if self.orientation == 'per_column':
            pair_act = torch.permute(pair_act, [1,0,2])
        
        return pair_act

class TriangleMultiplication(nn.Module):
    """Triangle multiplication layer ("outgoing" or "incoming").
    
    Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
    Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
    """
    def __init__(self, input_dim, output_dim, equation='ikc,jkc->ijc', num_intermediate_channel=128, zero_init=True):
        """
        equation: str
            ikc,jkc->ijc -- triangle_multiplication_outgoing
            kjc,kic->ijc -- triangle_multiplication_incoming
        """
        super().__init__()
        
        assert equation in ('ikc,jkc->ijc', 'kjc,kic->ijc')
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_intermediate_channel = num_intermediate_channel
        self.zero_init = zero_init
        self.equation = equation
        
        self.layer_norm_input = LayerNorm(input_dim)
        self.center_layer_norm = LayerNorm(num_intermediate_channel)
        
        self.left_projection = Linear(input_dim, num_intermediate_channel)
        self.right_projection = Linear(input_dim, num_intermediate_channel)
        
        self.left_gate  = Linear(input_dim, num_intermediate_channel, bias_init=1)
        self.right_gate = Linear(input_dim, num_intermediate_channel, bias_init=1)
        
        if zero_init:
            torch.nn.init.zeros_(self.left_gate.weights)
            torch.nn.init.zeros_(self.right_gate.weights)
        
        self.output_projection = Linear(num_intermediate_channel, output_dim)
        self.gating_linear = Linear(input_dim, output_dim, bias_init=1)
        if zero_init:
            torch.nn.init.zeros_(self.output_projection.weights)
            torch.nn.init.zeros_(self.gating_linear.weights)
    
    def forward(self, act, mask):
        """
        act:  [N_res, N_res, c_z]
        mask: [N_res, N_res]
        """
        assert act.ndim == 3
        assert mask.ndim == 2
        assert act.shape[2] == self.input_dim
        
        mask = mask[..., None] # [N_res, N_res, 1]
        
        act = self.layer_norm_input(act)
        input_act = act
        
        left_proj_act = mask * self.left_projection(act)
        right_proj_act = mask * self.right_projection(act)
        
        left_gate_values = torch.sigmoid(self.left_gate(act))
        right_gate_values = torch.sigmoid(self.right_gate(act))
        
        if self.training:
            left_proj_act = left_proj_act * left_gate_values
            right_proj_act = right_proj_act * right_gate_values
        else:
            left_proj_act *= left_gate_values
            right_proj_act *= right_gate_values
        del left_gate_values
        del right_gate_values
        del act
        del mask
        
        act = torch.einsum(self.equation, left_proj_act, right_proj_act)
        act = self.center_layer_norm(act)
        act = self.output_projection(act)
        
        gate_values = torch.sigmoid(self.gating_linear(input_act))
        act = act * gate_values

        return act

class EvoformerMSAIteration(nn.Module):
    def __init__(self, msa_dim=256, 
                       num_head=8, row_dropout=0.15,
                       col_dropout=0.0, trans_dropout=0.0,
                       gating=True, zero_init=True):
        super().__init__()
        self.msa_dim = msa_dim
        self.num_head = num_head
        self.row_dropout = row_dropout
        self.col_dropout = col_dropout
        self.trans_dropout = trans_dropout
        
        self.msa_row_attention = MSARowAttention(
                input_dim=msa_dim, 
                output_dim=msa_dim, 
                num_head=num_head, 
                gating=gating, 
                zero_init=zero_init)
        self.msa_column_attention = MSAColumnAttention(
                input_dim=msa_dim, 
                output_dim=msa_dim, 
                num_head=num_head, 
                gating=gating, 
                zero_init=zero_init)
        self.msa_transition = Transition(
            input_dim=msa_dim, 
            num_intermediate_factor=4, 
            zero_init=zero_init)
    
    def forward(self, msa_act, msa_mask):
        """
        msa_act:  [N_seq, N_res, c_m]
        msa_mask: [N_seq, N_res]
        """
        assert msa_act.ndim == 3
        assert msa_mask.ndim == 2
        assert msa_act.shape[2] == self.msa_dim
        
        msa_act = axis_dropout(self.msa_row_attention(msa_act, msa_mask), self.row_dropout, 0, self.training) + msa_act
        msa_act = axis_dropout(self.msa_column_attention(msa_act, msa_mask), self.col_dropout, 1, self.training) + msa_act
        msa_act = axis_dropout(self.msa_transition(msa_act, msa_mask), self.trans_dropout, None, self.training) + msa_act        
        return msa_act

#COUNT = 0

class EvoformerIteration(nn.Module):
    def __init__(self, msa_dim=256, pair_dim=128,
                       msa_num_head=8, 
                       pair_num_head=4,
                       row_dropout=0.15,
                       col_dropout=0.0, 
                       trans_dropout=0.0,
                       tri_start_dropout=0.25,
                       tri_end_dropout=0.25,
                       tri_out_dropout=0.25,
                       tri_in_dropout=0.25,
                       pair_trans_dropout=0.0,
                       is_extra_msa=False,
                       gating=True, 
                       zero_init=True,
                       outer_product_mean_first=False):
        super().__init__()
        
        self.msa_dim = msa_dim
        self.pair_dim = pair_dim
        self.msa_num_head = msa_num_head
        self.pair_num_head = pair_num_head
        self.row_dropout = row_dropout
        self.col_dropout = col_dropout
        self.trans_dropout = trans_dropout
        self.tri_start_dropout = tri_start_dropout
        self.tri_end_dropout = tri_end_dropout
        self.tri_out_dropout = tri_out_dropout
        self.tri_in_dropout = tri_in_dropout
        self.pair_trans_dropout = pair_trans_dropout
        self.is_extra_msa = is_extra_msa
        self.gating = gating
        self.outer_product_mean_first = outer_product_mean_first
        
        self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(
                        msa_input_dim=msa_dim, 
                        pair_input_dim=pair_dim,
                        output_dim=msa_dim, 
                        num_head=msa_num_head, 
                        gating=gating, 
                        zero_init=zero_init)
        if is_extra_msa:
            self.msa_column_global_attention = MSAColumnGlobalAttention(
                input_dim=msa_dim, 
                output_dim=msa_dim, 
                num_head=msa_num_head, 
                gating=gating, 
                zero_init=zero_init)
        else:
            self.msa_column_attention = MSAColumnAttention(
                input_dim=msa_dim, 
                        output_dim=msa_dim, 
                        num_head=msa_num_head, 
                        gating=gating, 
                        zero_init=zero_init)
        self.msa_transition = Transition(
            input_dim=msa_dim, 
            num_intermediate_factor=4, 
            zero_init=zero_init)
        self.outer_product_mean = OuterProductMean(
            input_dim=msa_dim, 
            output_dim=pair_dim, 
            num_outer_channel=32, 
            zero_init=zero_init)
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            input_dim=pair_dim, 
            output_dim=pair_dim, 
            equation='ikc,jkc->ijc', 
            num_intermediate_channel=128, 
            zero_init=zero_init)
        self.triangle_multiplication_incoming = TriangleMultiplication(
            input_dim=pair_dim, 
            output_dim=pair_dim, 
            equation='kjc,kic->ijc', 
            num_intermediate_channel=128, 
            zero_init=zero_init)
        self.triangle_attention_starting_node = TriangleAttention(
            input_dim=pair_dim, 
            output_dim=pair_dim, 
            num_head=pair_num_head, 
            gating=gating, 
            zero_init=zero_init, 
            orientation='per_row')
        self.triangle_attention_ending_node = TriangleAttention(
            input_dim=pair_dim, 
            output_dim=pair_dim, 
            num_head=pair_num_head, 
            gating=gating, 
            zero_init=zero_init, 
            orientation='per_column')
        self.pair_transition = Transition(
            input_dim=pair_dim, 
            num_intermediate_factor=4, 
            zero_init=zero_init)
    
    def forward(self, msa_act, msa_mask, pair_act, pair_mask):
        """
        msa_act:   [N_seq, N_res, c_m]
        msa_mask:  [N_seq, N_res]
        pair_act:  [N_res, N_res, c_z]
        pair_mask: [N_res, N_res]
        """
        assert msa_act.ndim == pair_act.ndim == 3
        assert msa_mask.ndim == pair_mask.ndim == 2
        assert msa_act.shape[2] == self.msa_dim
        assert pair_act.shape[2] == self.pair_dim
        
        #profiler.record(True, "Enter EvoformerIteration")
        
        if self.training:
            if self.outer_product_mean_first:
                pair_act = axis_dropout(self.outer_product_mean(msa_act, msa_mask), self.trans_dropout, None, self.training) + pair_act
            msa_act = axis_dropout(self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act), self.row_dropout, 0, self.training) + msa_act
            if self.is_extra_msa:
                msa_act = axis_dropout(self.msa_column_global_attention(msa_act, msa_mask), self.col_dropout, 1, self.training) + msa_act
            else:
                msa_act = axis_dropout(self.msa_column_attention(msa_act, msa_mask), self.col_dropout, 1, self.training) + msa_act
            msa_act = axis_dropout(self.msa_transition(msa_act, msa_mask), self.trans_dropout, None, self.training) + msa_act
            if not self.outer_product_mean_first:
                pair_act = axis_dropout(self.outer_product_mean(msa_act, msa_mask), self.trans_dropout, None, self.training) + pair_act
            pair_act = axis_dropout(self.triangle_multiplication_outgoing(pair_act, pair_mask), self.tri_out_dropout, 0, self.training) + pair_act
            pair_act = axis_dropout(self.triangle_multiplication_incoming(pair_act, pair_mask), self.tri_in_dropout, 0, self.training) + pair_act
            pair_act = axis_dropout(self.triangle_attention_starting_node(pair_act, pair_mask), self.tri_start_dropout, 0, self.training) + pair_act
            pair_act = axis_dropout(self.triangle_attention_ending_node(pair_act, pair_mask), self.tri_end_dropout, 1, self.training) + pair_act
            pair_act = axis_dropout(self.pair_transition(pair_act, pair_mask), self.pair_trans_dropout, 0, self.training) + pair_act
        else:
            if self.outer_product_mean_first:
                pair_act += axis_dropout(self.outer_product_mean(msa_act, msa_mask), self.trans_dropout, None, self.training)
            msa_act += axis_dropout(self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act), self.row_dropout, 0, self.training)
            if self.is_extra_msa:
                msa_act += axis_dropout(self.msa_column_global_attention(msa_act, msa_mask), self.col_dropout, 1, self.training)
            else:
                msa_act += axis_dropout(self.msa_column_attention(msa_act, msa_mask), self.col_dropout, 1, self.training)
            msa_act += axis_dropout(self.msa_transition(msa_act, msa_mask), self.trans_dropout, None, self.training)
            if not self.outer_product_mean_first:
                pair_act += axis_dropout(self.outer_product_mean(msa_act, msa_mask), self.trans_dropout, None, self.training)
            pair_act += axis_dropout(self.triangle_multiplication_outgoing(pair_act, pair_mask), self.tri_out_dropout, 0, self.training)
            pair_act += axis_dropout(self.triangle_multiplication_incoming(pair_act, pair_mask), self.tri_in_dropout, 0, self.training)
            pair_act += axis_dropout(self.triangle_attention_starting_node(pair_act, pair_mask), self.tri_start_dropout, 0, self.training)
            pair_act += axis_dropout(self.triangle_attention_ending_node(pair_act, pair_mask), self.tri_end_dropout, 1, self.training)
            pair_act += axis_dropout(self.pair_transition(pair_act, pair_mask), self.pair_trans_dropout, 0, self.training)
        
        return msa_act, pair_act


    
class EmbeddingsAndEvoformer(nn.Module):
    """Embeds the input data and runs Evoformer.
    
    Produces the MSA, single and pair representations.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5-18
    """
    
    def __init__(self, target_feat_dim=22, msa_feat_dim=49, 
                    msa_channel=256, evoformer_num_block=48,
                    extra_msa_channel=64, extra_msa_stack_num_block=4,
                    pair_channel=128, seq_channel=384, max_relative_feature=32,
                    max_relative_idx=32, use_chain_relative=True, max_relative_chain=2,
                    gating=True, zero_init=True, recycle_features=True, recycle_pos=True,
                    enable_template=False, embed_torsion_angles=True):
        super().__init__()
        self.target_feat_dim = target_feat_dim
        self.msa_feat_dim = msa_feat_dim
        self.msa_channel = msa_channel
        self.evoformer_num_block = evoformer_num_block
        self.extra_msa_channel = extra_msa_channel
        self.extra_msa_stack_num_block = extra_msa_stack_num_block
        self.pair_channel = pair_channel
        self.seq_channel = seq_channel
        self.max_relative_feature = max_relative_feature
        self.recycle_features = recycle_features
        self.recycle_pos = recycle_pos
        self.enable_template = enable_template
        self.embed_torsion_angles = embed_torsion_angles
        
        self.preprocess_1d = Linear(target_feat_dim, msa_channel)
        self.preprocess_msa = Linear(msa_feat_dim, msa_channel)
        self.left_single = Linear(target_feat_dim, pair_channel)
        self.right_single = Linear(target_feat_dim, pair_channel)
        
        self.pair_activiations = Linear(2 * max_relative_feature + 1, pair_channel)
        self.extra_msa_activations = Linear(25, extra_msa_channel)

        self.max_relative_idx = max_relative_idx
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if max_relative_idx:
            in_dim = (2*max_relative_idx+2)+1+(2*max_relative_chain+2) if use_chain_relative else (2*max_relative_idx+1)
            assert in_dim == 73
            self.position_activations = Linear(in_dim, pair_channel)


        self.gating = gating
        self.zero_init =zero_init
        
        self.extra_msa_stack = nn.ModuleList([
            EvoformerIteration(
                    msa_dim=extra_msa_channel, 
                    pair_dim=pair_channel,
                    msa_num_head=8, 
                    pair_num_head=4,
                    row_dropout=0.15,
                    col_dropout=0.0, 
                    trans_dropout=0.0,
                    tri_start_dropout=0.25,
                    tri_end_dropout=0.25,
                    tri_out_dropout=0.25,
                    tri_in_dropout=0.25,
                    pair_trans_dropout=0.0,
                    is_extra_msa=True,
                    gating=gating, 
                    zero_init=zero_init) for _ in range(extra_msa_stack_num_block)
        ])
        
        self.evoformer_iteration = nn.ModuleList([
            EvoformerIteration(
                    msa_dim=msa_channel, 
                    pair_dim=pair_channel,
                    msa_num_head=8, 
                    pair_num_head=4,
                    row_dropout=0.15,
                    col_dropout=0.0, 
                    trans_dropout=0.0,
                    tri_start_dropout=0.25,
                    tri_end_dropout=0.25,
                    tri_out_dropout=0.25,
                    tri_in_dropout=0.25,
                    pair_trans_dropout=0.0,
                    is_extra_msa=False,
                    gating=gating, 
                    zero_init=zero_init) for _ in range(evoformer_num_block)
        ])
        
        if self.enable_template:
            self.template_embedding = TemplateEmbedding(
                pair_dim=pair_channel, 
                num_channels=64, 
                num_block=2, 
                num_head=4,
                use_template_unit_vector=True,
                gating=gating, 
                zero_init=zero_init, 
                tri_start_dropout=0.25, 
                tri_end_dropout=0.25, 
                tri_out_dropout=0.25, 
                tri_in_dropout=0.25, 
                pair_trans_dropout=0.0)
            if embed_torsion_angles:
                self.template_single_embedding = Linear(22+14+14+7, msa_channel, initializer='relu')
                self.template_projection = Linear(msa_channel, msa_channel, initializer='relu')
        
        self.single_activations = Linear(msa_channel, seq_channel)
        
        if recycle_pos:
            self.prev_pos_linear = Linear(15, pair_channel)
        if recycle_features:
            self.prev_msa_first_row_norm = LayerNorm(msa_channel)
            self.prev_pair_norm = LayerNorm(pair_channel)


    def _relative_encoding(self, batch):
        """Add relative position encodings.

        For position (i, j), the value is (i-j) clipped to [-k, k] and one-hotted.

        When not using 'use_chain_relative' the residue indices are used as is, e.g.
        for heteromers relative positions will be computed using the positions in
        the corresponding chains.

        When using 'use_chain_relative' we add an extra bin that denotes
        'different chain'. Furthermore we also provide the relative chain index
        (i.e. sym_id) clipped and one-hotted to the network. And an extra feature
        which denotes whether they belong to the same chain type, i.e. it's 0 if
        they are in different heteromer chains and 1 otherwise.

        Parameters
        --------------
        batch: dict
            -- residue_index: [N_res]
            -- asym_id: [N_res]
            -- entity_id: [N_res]
            -- sym_id: [N_res]

        Returns
        --------------
        position_encoding: [N_res, N_res, pair_dim]
        """

        rel_feats = []
        pos = batch['residue_index']
        asym_id = batch['asym_id']
        asym_id_same = torch.eq(asym_id[:, None], asym_id[None, :])
        offset = pos[:, None] - pos[None, :]
        clipped_offset = torch.clip(offset+self.max_relative_idx, min=0, max=2 * self.max_relative_idx)

        if self.use_chain_relative:
            final_offset = torch.where(asym_id_same, clipped_offset, (2 * self.max_relative_idx + 1) * torch.ones_like(clipped_offset, device=pos.device))
            rel_pos = F.one_hot(final_offset.long(), 2 * self.max_relative_idx + 2)
            rel_feats.append(rel_pos) # 2 * max_relative_idx + 2

            entity_id = batch['entity_id']
            entity_id_same = torch.eq(entity_id[:, None], entity_id[None, :])
            rel_feats.append(entity_id_same.to(rel_pos)[..., None]) # 1

            sym_id = batch['sym_id']
            rel_sym_id = sym_id[:, None] - sym_id[None, :]
            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = torch.clip(rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain)
            final_rel_chain = torch.where(entity_id_same, clipped_rel_chain, (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain, device=pos.device))
            rel_chain = F.one_hot(final_rel_chain.long(), 2 * max_rel_chain + 2)
            rel_feats.append(rel_chain) # 2 * max_rel_chain + 2
        else:
            rel_pos = F.one_hot(clipped_offset.long(), 2 * self.max_relative_idx + 1)
            rel_feats.append(rel_pos) # 2 * self.max_relative_idx + 1

        rel_feat = torch.cat(rel_feats, dim=-1).float()
        position_encoding = self.position_activations(rel_feat)
        return position_encoding

    def forward(self, batch):
        """
        batch: dict
            target_feat -- [N_res, 22]
            msa_feat -- [N_seq, N_res, 49]
            seq_mask -- [N_res]
            residue_index -- [N_res], must be torch.long
            msa_mask -- [N_seq, N_res]
            
            extra_msa -- [N_extra_seq, N_res], must be torch.long
            extra_msa_mask -- [N_extra_seq, N_res]
            extra_has_deletion -- [N_extra_seq, N_res]
            extra_deletion_value -- [N_extra_seq, N_res]
        """
        assert batch['target_feat'].ndim == 2
        assert batch['msa_feat'].ndim == 3
        assert batch['seq_mask'].ndim == 1
        assert batch['residue_index'].ndim == 1
        assert batch['msa_mask'].ndim == 2
        assert batch['extra_msa'].ndim == 2
        assert batch['extra_msa_mask'].ndim == 2
        assert batch['extra_has_deletion'].ndim == 2
        assert batch['extra_deletion_value'].ndim == 2
        assert batch['target_feat'].shape[1] == self.target_feat_dim
        assert batch['msa_feat'].shape[2] == self.msa_feat_dim
        batch['residue_index'] = batch['residue_index'].long()
        batch['extra_msa'] = batch['extra_msa'].long()
        device = batch['msa_feat'].device

        # ======= alg. 3, Embeddings for initial representations (target and msa) ======= #
        preprocess_1d = self.preprocess_1d(batch['target_feat']) # [N_res, c_m]
        preprocess_msa = self.preprocess_msa(batch['msa_feat'])  # [N_seq, N_res, c_m]
        msa_activations = preprocess_1d[None,...] + preprocess_msa  # [N_seq, N_res, c_m]
        
        left_single = self.left_single(batch['target_feat'])   # [N_res, c_z]
        right_single = self.right_single(batch['target_feat']) # [N_res, c_z]
        # [N,1,pair_channel] [1,N,pair_channel] => [N,N,pair_channel]
        pair_activations = left_single[:, None, :] + right_single[None, :, :] # [N_res, N_res, c_z]
        mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :] # [N_res, N_res]


        # ======= update recycle features ======= #
        if self.recycle_pos and 'prev_pos' in batch:
            #print(batch['aatype'].shape, batch['prev_pos'].shape)
            prev_pseudo_beta = pseudo_beta_fn(batch['aatype'], batch['prev_pos'], None)
            dgram = dgram_from_positions(prev_pseudo_beta, num_bins=15, min_bin=3.25, max_bin=20.75)
            # [N,N,pair_channel] -> [N,N,pair_channel]
            pair_activations = self.prev_pos_linear(dgram) + pair_activations

        if self.recycle_features:
            if 'prev_msa_first_row' in batch:
                prev_msa_first_row = self.prev_msa_first_row_norm(batch['prev_msa_first_row'])
                msa_activations[0] += prev_msa_first_row
                # msa_activations = jax.ops.index_add(msa_activations, 0, prev_msa_first_row) # 用single更新MSA representation
            if 'prev_pair' in batch:
                pair_activations += self.prev_pair_norm(batch['prev_pair'])

        
        # ======== alg. 4, Relative position encoding ======= #
        if self.max_relative_idx:
            pair_activations += self._relative_encoding(batch)


        # ======== alg. 16-17, template embedding ======= #
        if self.enable_template:
            template_batch = {k: batch[k] for k in batch if k.startswith('template_')}
            #multichain_mask = batch['asym_id'][:, None] == batch['asym_id'][None, :]
            multichain_mask = (batch['template_pair_dist'] != 0).float()
            multichain_mask = multichain_mask[0]  # 2d
            template_pair_representation = self.template_embedding(pair_activations,template_batch,mask_2d,multichain_mask)
            pair_activations = pair_activations + template_pair_representation


        # ======== alg. 18-19, extra msa embedding ======= #
        extra_msa_feat = create_extra_msa_feature(batch)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        
        extra_evoformer_param = [extra_msa_activations, batch['extra_msa_mask'], pair_activations, mask_2d]
        for layer in tqdm(self.extra_msa_stack, desc='extra_msa_stack', dynamic_ncols=True, disable=DISABLE_TQDM):
            if self.training:
                extra_evoformer_output = checkpoint(layer, *extra_evoformer_param)
            else:
                extra_evoformer_output = layer(*extra_evoformer_param) # msa_act, pair_act
            extra_evoformer_param[0] = extra_evoformer_output[0]
            extra_evoformer_param[2] = extra_evoformer_output[1]
        
        pair_activations = extra_evoformer_output[1]

        # ========  embedd torch angles ======= #
        msa_mask = batch['msa_mask']
        if self.enable_template and self.embed_torsion_angles:
            num_templ, num_res = batch['template_aatype'].shape
            
            # Embed the templates aatypes.
            aatype_one_hot = F.one_hot(batch['template_aatype'].long(), 22)
            
            ret = all_atom.atom37_to_torsion_angles(
                aatype=batch['template_aatype'],
                all_atom_pos=batch['template_all_atom_positions'],
                all_atom_mask=batch['template_all_atom_masks'],
                # Ensure consistent behaviour during testing:
                placeholder_for_undefined=not self.zero_init)
            
            template_features = torch.cat([
                    aatype_one_hot, 
                    torch.reshape(ret['torsion_angles_sin_cos'], [num_templ, num_res, 14]),
                    torch.reshape(ret['alt_torsion_angles_sin_cos'], [num_templ, num_res, 14]),
                    ret['torsion_angles_mask']], dim=-1) # [ N_templ, N_res, 22+14+14+7 ]
            
            template_activations = self.template_single_embedding(template_features)
            template_activations = F.leaky_relu(template_activations)
            template_activations = self.template_projection(template_activations) # [N_templ, N_res, c_m]
            
            # Concatenate the templates to the msa.
            msa_activations = torch.cat([msa_activations, template_activations], dim=0) # [N_seq+N_templ, N_res, c_m]
            # Concatenate templates masks to the msa masks.
            # Use mask from the psi angle, as it only depends on the backbone atoms
            # from a single residue.
            torsion_angle_mask = ret['torsion_angles_mask'][:, :, 2] # [N_templ, N_res]
            torsion_angle_mask = torsion_angle_mask.to(msa_mask.dtype)
            msa_mask = torch.cat([msa_mask, torsion_angle_mask], dim=0) # [N_seq+N_templ, N_res]

        # ======== alg. 6,  EvoformerStack ======== #
        evoformer_param = [msa_activations, msa_mask, pair_activations, mask_2d]
        #for layer in self.evoformer_iteration:
        for layer in tqdm(self.evoformer_iteration, desc='evoformer_iteration', dynamic_ncols=True,
                          disable=DISABLE_TQDM):
            if self.training:
                evoformer_output = checkpoint(layer, *evoformer_param)
            else:
                evoformer_output = layer(*evoformer_param) # msa_act, pair_act
            evoformer_param[0] = evoformer_output[0]
            evoformer_param[2] = evoformer_output[1]
        
        msa_activations = evoformer_output[0]
        pair_activations = evoformer_output[1]
        
        single_activations = self.single_activations(msa_activations[0])


        num_sequences = batch['msa_feat'].shape[0]
        output = {
            'single': single_activations,
            'pair': pair_activations,
            'msa': msa_activations[:num_sequences, :, :],
            'msa_first_row': msa_activations[0].clone(),
        }

        return output



class Template_layer_stack_no_state(nn.Module):
    def __init__(self, pair_dim=128,
                        num_head=4, 
                        gating=True, 
                        zero_init=True,
                        tri_start_dropout=0.25,
                        tri_end_dropout=0.25,
                        tri_out_dropout=0.25,
                        tri_in_dropout=0.25,
                        pair_trans_dropout=0):
        super().__init__()
        self.tri_start_dropout = tri_start_dropout
        self.tri_end_dropout = tri_end_dropout
        self.tri_out_dropout = tri_out_dropout
        self.tri_in_dropout = tri_in_dropout
        self.pair_trans_dropout = pair_trans_dropout

        self.triangle_attention_starting_node = TriangleAttention(
            input_dim=pair_dim, 
            output_dim=pair_dim, 
            key_dim=64,
            value_dim=64,
            num_head=num_head, 
            gating=gating, 
            zero_init=zero_init, 
            orientation='per_row')
        self.triangle_attention_ending_node = TriangleAttention(
            input_dim=pair_dim, 
            output_dim=pair_dim, 
            key_dim=64,
            value_dim=64,
            num_head=num_head, 
            gating=gating, 
            zero_init=zero_init, 
            orientation='per_column')
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            input_dim=pair_dim, 
            output_dim=pair_dim, 
            equation='ikc,jkc->ijc', 
            num_intermediate_channel=64, 
            zero_init=zero_init)
        self.triangle_multiplication_incoming = TriangleMultiplication(
            input_dim=pair_dim, 
            output_dim=pair_dim, 
            equation='kjc,kic->ijc', 
            num_intermediate_channel=64, 
            zero_init=zero_init)
        self.pair_transition = Transition(
            input_dim=pair_dim, 
            num_intermediate_factor=2, 
            zero_init=zero_init)
    
    def forward(self, pair_act, pair_mask):
        pair_act = axis_dropout(self.triangle_attention_starting_node(pair_act, pair_mask), self.tri_start_dropout, 0, self.training) + pair_act
        pair_act = axis_dropout(self.triangle_attention_ending_node(pair_act, pair_mask), self.tri_end_dropout, 1, self.training) + pair_act
        pair_act = axis_dropout(self.triangle_multiplication_outgoing(pair_act, pair_mask), self.tri_out_dropout, 0, self.training) + pair_act
        pair_act = axis_dropout(self.triangle_multiplication_incoming(pair_act, pair_mask), self.tri_in_dropout, 0, self.training) + pair_act
        pair_act = axis_dropout(self.pair_transition(pair_act, pair_mask), self.pair_trans_dropout, 0, self.training) + pair_act
        return pair_act


class TemplatePairStack(nn.Module):
    """Pair stack for the templates.
    
    Jumper et al. (2021) Suppl. Alg. 16 "TemplatePairStack"
    """
    
    def __init__(self, pair_dim=128,
                        num_block=2,
                        num_head=4, 
                        gating=True, 
                        zero_init=True,
                        tri_start_dropout=0.25,
                        tri_end_dropout=0.25,
                        tri_out_dropout=0.25,
                        tri_in_dropout=0.25,
                        pair_trans_dropout=0):
        super().__init__()
    
        self.pair_dim = pair_dim
        self.num_block = num_block
        self.num_head = num_head
        self.gating = gating
        self.zero_init = zero_init
        self.tri_start_dropout = tri_start_dropout
        self.tri_end_dropout = tri_end_dropout
        self.tri_out_dropout = tri_out_dropout
        self.tri_in_dropout = tri_in_dropout
        self.pair_trans_dropout = pair_trans_dropout
        
        self.__layer_stack_no_state = nn.ModuleList([
            Template_layer_stack_no_state(
                        pair_dim=pair_dim,
                        num_head=num_head, 
                        gating=gating, 
                        zero_init=zero_init,
                        tri_start_dropout=tri_start_dropout,
                        tri_end_dropout=tri_end_dropout,
                        tri_out_dropout=tri_out_dropout,
                        tri_in_dropout=tri_in_dropout,
                        pair_trans_dropout=pair_trans_dropout) for _ in range(num_block)
        ])

    
    def forward(self, pair_act, pair_mask):
        """Builds TemplatePairStack module.
        
        Arguments:
          pair_act: [N_res, N_res, c_t]
          pair_mask: [N_res, N_res]
        
        Returns:
          Updated pair_act, shape [N_res, N_res, c_t].
        """
        assert pair_act.ndim == 3
        assert pair_mask.ndim == 2
        assert pair_act.shape[2] == self.pair_dim
        
        for layer in self.__layer_stack_no_state:
            if self.training:
                pair_act = checkpoint(layer, pair_act, pair_mask)
            else:
                pair_act = layer(pair_act, pair_mask)
        
        return pair_act


class SingleTemplateEmbedding(nn.Module):
    """Embeds a single template.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9+11
    """
    
    def __init__(self, num_channels=64,
                        use_template_unit_vector=False,
                        num_block=2,
                        num_head=4, 
                        gating=True, 
                        zero_init=True,
                        tri_start_dropout=0.25,
                        tri_end_dropout=0.25,
                        tri_out_dropout=0.25,
                        tri_in_dropout=0.25,
                        pair_trans_dropout=0):
        super().__init__()
        
        self.num_channels = num_channels
        self.use_template_unit_vector = use_template_unit_vector
        
        self.template_pair_stack = TemplatePairStack(pair_dim=num_channels,
                                                    num_block=num_block,
                                                    num_head=num_head, 
                                                    gating=gating, 
                                                    zero_init=zero_init,
                                                    tri_start_dropout=tri_start_dropout,
                                                    tri_end_dropout=tri_end_dropout,
                                                    tri_out_dropout=tri_out_dropout,
                                                    tri_in_dropout=tri_in_dropout,
                                                    pair_trans_dropout=pair_trans_dropout)
        self.embedding2d = Linear(88, num_channels, initializer='relu') # 64
        self.output_layer_norm = LayerNorm(num_channels)
    
    def forward(self, query_embedding, batch, mask_2d, multichain_mask_2d):
        """Build the single template embedding.
        
        Arguments:
          query_embedding: [N_res, N_res, c_z]
          batch: A batch of template features (note the template dimension has been stripped out as this module only runs over a single template).
                template_aatype -- [N_res]
                template_pseudo_beta_mask -- [N_res]
                template_pseudo_beta -- [N_res, 3]
                template_all_atom_positions -- [N_res, 37, 3]
                template_all_atom_masks -- [N_res, 37]
        
          mask_2d: Padding mask (Note: this doesn't care if a template exists, unlike the template_pseudo_beta_mask).
        
        Returns:
          A template embedding [N_res, N_res, c_z].
        """
        assert query_embedding.ndim == 3
        assert mask_2d.ndim == 2
        assert multichain_mask_2d.ndim == 2
        mask_2d = mask_2d.to(query_embedding.dtype)
        multichain_mask_2d = multichain_mask_2d.to(query_embedding.dtype)
        dtype, device = query_embedding.dtype, query_embedding.device
        
        assert batch['template_aatype'].ndim == 1
        assert batch['template_pseudo_beta_mask'].ndim == 1
        assert batch['template_pair_dist'].ndim == 2
        
        num_res = batch['template_aatype'].shape[0]
        template_mask = batch['template_pseudo_beta_mask']
        template_mask_2d = template_mask[:, None] * template_mask[None, :]
        template_mask_2d = template_mask_2d.to(dtype).to(device) # [N_res, N_res]
        template_mask_2d *= multichain_mask_2d

        template_dgram = dgram_from_pairs(batch['template_pair_dist'], num_bins=39, min_bin=3.25,
                                              max_bin=50.75)  # [N_res, N_res, 39]
        template_dgram = template_dgram.to(dtype).to(device)
        
        to_concat = [template_dgram, template_mask_2d[:, :, None]] # 39 + 1
        
        aatype = F.one_hot(batch['template_aatype'].long(), 22).to(dtype).to(device) # [N_res, 22]
        
        to_concat.append(aatype[None, :, :].repeat([num_res, 1, 1])) # 40 + 22
        to_concat.append(aatype[:, None, :].repeat([1, num_res, 1])) # 62 + 22
        
        unit_vector = batch['template_pair_orient']
        if not self.use_template_unit_vector:
            unit_vector = torch.zeros_like(unit_vector)

        unit_vector = unit_vector.to(dtype).to(device)
        template_mask_2d = template_mask_2d.to(dtype).to(device)
        
        to_concat.append(unit_vector) #
        to_concat.append(template_mask_2d[..., None])

        act = torch.cat(to_concat, dim=-1) # [N_res, N_res, 88]
        # Mask out non-template regions so we don't get arbitrary values in the
        # distogram for these regions.
        act = act * template_mask_2d[..., None]

        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 9
        act = self.embedding2d(act)
        
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 11
        act = self.template_pair_stack(act, mask_2d)
        
        # act = self.output_layer_norm(act)
        return act


class TemplateEmbedding(nn.Module):
    """Embeds a set of templates.
    
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-12
    Jumper et al. (2021) Suppl. Alg. 17 "TemplatePointwiseAttention"
    """
    
    def __init__(self, pair_dim=128, num_channels=64, num_block=2, num_head=4,
            use_template_unit_vector=False, gating=True, zero_init=True, 
            tri_start_dropout=0.25, tri_end_dropout=0.25, 
            tri_out_dropout=0.25, tri_in_dropout=0.25, pair_trans_dropout=0.0):
        super().__init__()
        self.pair_dim = pair_dim
        self.num_channels = num_channels
        self.use_template_unit_vector = use_template_unit_vector
        
        # gating is disabled here
        self.attention = Attention(query_dim=pair_dim, memo_dim=num_channels, output_dim=pair_dim, 
            num_head=4, key_dim=64, value_dim=64, gating=False, zero_init=zero_init)
        
        self.single_template_embedding = SingleTemplateEmbedding(
                        num_channels=num_channels,
                        use_template_unit_vector=use_template_unit_vector,
                        num_block=num_block,
                        num_head=num_head, 
                        gating=gating, 
                        zero_init=zero_init,
                        tri_start_dropout=tri_start_dropout,
                        tri_end_dropout=tri_end_dropout,
                        tri_out_dropout=tri_out_dropout,
                        tri_in_dropout=tri_in_dropout,
                        pair_trans_dropout=pair_trans_dropout)
        self.output_linear = Linear(num_channels, pair_dim, initializer='relu')

    def forward(self, query_embedding, template_batch, mask_2d, multichain_mask_2d):
        """Build TemplateEmbedding module.
        
        Arguments:
          query_embedding: Query pair representation, shape [N_res, N_res, c_z].
          template_batch: A batch of template features.
            -- template_mask -- [N_templ]
            -- template_aatype -- [N_templ, N_res]
            -- template_pseudo_beta_mask -- [N_templ, N_res]
            -- template_pseudo_beta -- [N_templ, N_res, 3]
            -- template_all_atom_positions -- [N_templ, N_res, 37, 3]
            -- template_all_atom_masks -- [N_templ, N_res, 37]
          mask_2d: Padding mask (Note: this doesn't care if a template exists,
            unlike the template_pseudo_beta_mask).
        
        Returns:
          A template embedding [N_res, N_res, c_z].
        """
        assert query_embedding.ndim == 3
        assert template_batch['template_mask'].ndim == 1
        assert template_batch['template_aatype'].ndim == 2
        assert template_batch['template_pseudo_beta_mask'].ndim == 2
        assert template_batch['template_pair_dist'].ndim == 3  # [N_temp, N_res, N_res]
        assert query_embedding.shape[2] == self.pair_dim
        assert multichain_mask_2d.ndim == 2

        num_templates = template_batch['template_pair_dist'].shape[0]
        num_res = query_embedding.shape[0]
        
        dtype, device = query_embedding.dtype, query_embedding.device
        template_mask = template_batch['template_mask']
        template_mask = template_mask.to(dtype).to(device)
                
        embedding = 0
        for i in range(num_templates):
            single_templete_batch = { name: content[i] for name,content in template_batch.items() }
            embedding = embedding + self.single_template_embedding(query_embedding, single_templete_batch, mask_2d, multichain_mask_2d)
        embedding = embedding / num_templates
        embedding = F.relu(embedding)
        embedding = self.output_linear(embedding)


        return embedding


class InterMultiplication(nn.Module):
    def __init__(self, input_dim, output_dim, aggr_by_row=True, num_intermediate_channel=128, zero_init=True):
        super().__init__()

        self.equation = 'ik,jk->ijk'
        self.aggr_axis = 0 if aggr_by_row else 1
        self.input_dim = input_dim

        self.left_projection = Linear(input_dim, num_intermediate_channel)
        self.right_projection = Linear(input_dim, num_intermediate_channel)

        self.left_gate = Linear(input_dim, num_intermediate_channel, bias_init=1)
        self.right_gate = Linear(input_dim, num_intermediate_channel, bias_init=1)

        self.layer_norm_input = LayerNorm(input_dim)
        self.center_layer_norm = LayerNorm(num_intermediate_channel)
        self.output_projection = Linear(num_intermediate_channel, output_dim)

    def forward(self, act_s, act_r, mask_r):
        """
        act_s:  [N_sup, N_sup, c_z]
        act_r:  [N_res, N_res, c_z]
        mask_r: [N_res, N_res]
        """
        assert act_s.ndim == act_r.ndim == 3
        assert mask_r.ndim == 2
        assert act_s.shape[2] == act_r.shape[2] == self.input_dim

        mask_r = mask_r[..., None]  # [N_res, N_res, 1]

        act_s = self.layer_norm_input(act_s)
        act_r = self.layer_norm_input(act_r)  # share the same layer norm affine parameters
        input_act_s = act_s
        input_act_r = act_r

        left_proj_act = self.left_projection(act_s)
        right_proj_act = mask_r * self.right_projection(act_r)

        left_gate_values = torch.sigmoid(self.left_gate(act_s))
        right_gate_values = torch.sigmoid(self.right_gate(act_r))

        left_proj_act = left_proj_act * left_gate_values
        right_proj_act = right_proj_act * right_gate_values

        del left_gate_values
        del right_gate_values
        del act_s
        del act_r
        del mask_r

        left_line_act = left_proj_act.sum(axis=self.aggr_axis)
        right_line_act = right_proj_act.sum(axis=self.aggr_axis)

        act = torch.einsum(self.equation, left_line_act, right_line_act)
        act = self.center_layer_norm(act)
        act = self.output_projection(act)  # [N_sup, N_res, c_x]

        return act



