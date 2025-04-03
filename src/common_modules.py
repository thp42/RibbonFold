import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers

try:
    #raise ImportError
    from apex.normalization import MixedFusedLayerNorm as LayerNorm
    LayerNorm(12)
    print("Use apex.normalization.MixedFusedLayerNorm as LayerNorm")
except ImportError:
    from torch.nn import LayerNorm as LayerNorm

def tensor_to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    assert isinstance(tensor, torch.Tensor)
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16 or tensor.dtype == torch.float16:
        tensor = tensor.float()
    return tensor.numpy()

def t2n(tensor):
    return tensor_to_numpy(tensor)

###############################
### Initiation functions
###############################

def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape."""
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        # Assuming convolution kernels (2D, 3D, or more.)
        # kernel_shape: (..., input_depth, depth)
        receptive_field_size = np.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out

class VarianceScaling:
    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal'):
        if scale < 0.0:
            raise ValueError('`scale` must be a positive float.')
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument:', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'truncated_normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument:', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
    
    def __call__(self, x):
        scale = self.scale
        shape = x.shape
        fan_in, fan_out = _compute_fans(shape)
        if self.mode == 'fan_in':
            scale /= max(1.0, fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        
        if self.distribution == 'truncated_normal':
            stddev = np.sqrt(scale)
            # Adjust stddev for truncation.
            # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            distribution_stddev = np.asarray(.87962566103423978)
            stddev = stddev / distribution_stddev
            return torch.nn.init.trunc_normal_(x, mean=0.0, std=stddev, a=-2.0, b=2.0)
        elif self.distribution == 'normal':
            stddev = np.sqrt(scale)
            return torch.nn.init.normal_(x, mean=0.0, std=stddev)
        else:
            limit = np.sqrt(3.0 * scale)
            return torch.nn.init.uniform_(x, a=-limit, b=limit)

###############################
### Dropout functions
###############################

def axis_dropout(tensor, p=0.15, broadcast_dim=1, is_training=True):
    """
    tensor: [N_seq, N_res, c_m]
    broadcast_dim: int or None
            0 -- Row-wise dropout
            1 -- Col-wise dropout
            None -- element-wise dropout
    """
    assert 0.0 <= p <= 1.0
    assert tensor.ndim == 3
    assert broadcast_dim in (0, 1, None)
    
    if is_training and p != 0.0:
        shape = list(tensor.shape)
        if broadcast_dim is not None:
            shape[broadcast_dim] = 1
        keep_rate = 1.0 - p
        
        probs = torch.empty(shape)
        torch.nn.init.constant_(probs, keep_rate)
        sample = torch.bernoulli(input=probs).to(tensor.device)
        return sample * tensor / keep_rate
    else:
        return tensor

###############################
### Linear layer
###############################

class Linear(nn.Module):
    """Protein folding specific Linear Module.
    
    This differs from the standard Haiku Linear in a few ways:
    * It supports inputs of arbitrary rank
    * Initializers are specified by strings
    """
    
    def __init__(self, input_dim, output_dim, initializer='linear', use_bias=True, bias_init=0, enable_autocast=True):
        """Constructs Linear Module.
        
        Args:
          num_output: number of output channels.
          initializer: What initializer to use, should be one of {'linear', 'relu', 'zeros'}
          use_bias: Whether to include trainable bias
          bias_init: Value used to initialize bias.
          name: name of module, used for name scopes.
        """
        
        super().__init__()
        
        if isinstance(output_dim, numbers.Integral):
            self.output_dim = (output_dim,)
        else:
            self.output_dim = tuple(output_dim)
        self.num_output_dims = len(self.output_dim)
        
        if isinstance(input_dim, numbers.Integral):
            self.input_dim = (input_dim,)
        else:
            self.input_dim = tuple(input_dim)
        self.num_input_dims = len(self.input_dim)
        
        # self.output_dim = output_dim
        self.initializer = initializer
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.enable_autocast = enable_autocast
        assert initializer in ('linear', 'relu', 'zeros')
        
        weight_shape = self.input_dim + self.output_dim
        self.weights = nn.Parameter(torch.randn(weight_shape))
        if initializer == 'linear':
            VarianceScaling(mode='fan_in', scale=1.)(self.weights)
        elif initializer == 'relu':
            VarianceScaling(mode='fan_in', scale=2.)(self.weights)
        elif initializer == 'zeros':
            torch.nn.init.zeros_(self.weights)
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
            torch.nn.init.constant_(self.bias, bias_init)
    
    def forward(self, inputs):
        """Connects Module.
        
        Args:
          inputs: Tensor of shape [..., num_channel]
        
        Returns:
          output of shape [..., num_output]
        """
        if self.num_input_dims > 0:
            assert inputs.shape[-self.num_input_dims:] == self.input_dim
        
        in_letters = 'abcde'[:self.num_input_dims]
        out_letters = 'hijkl'[:self.num_output_dims]
        equation = f'...{in_letters},{in_letters}{out_letters}->...{out_letters}'
        
        is_autocast_enabled = torch.is_autocast_enabled()
        autocast_gpu_dtype = torch.get_autocast_gpu_dtype()
        if not is_autocast_enabled:
            # For Float64 model (for test with Jax)
            inputs = inputs.to(self.weights)
        if is_autocast_enabled and (not self.enable_autocast):
            # For bfloat16 model (for AMP training or inference)
            inputs = inputs.to(self.weights)
        with torch.cuda.amp.autocast(self.enable_autocast and is_autocast_enabled, dtype=autocast_gpu_dtype):
            #inputs = torch.swapaxes(inputs, -1, -2)
            #output = torch.einsum('...cb,cd->...db', inputs, self.weights)
            #output = torch.swapaxes(output, -1, -2)
            output = torch.einsum(equation, inputs, self.weights)

        if self.use_bias:
            output += self.bias
        
        return output

###############################
### For AMP
###############################
    
def large_value(raw_value=1e9):
    if torch.is_autocast_enabled() and str(torch.get_autocast_gpu_dtype()) in ('torch.half', 'torch.float16'):
        return min(1e4, raw_value)
    else:
        return raw_value

def small_value(raw_value=1e-9):
    if torch.is_autocast_enabled() and str(torch.get_autocast_gpu_dtype()) in ('torch.half', 'torch.float16'):
        return max(1e-6, raw_value) 
    else:
        return raw_value

# Enable anomaly detection
DETECT_ANOMALY = False
def check_tensor(tensor, info=None):
    if DETECT_ANOMALY:
        if info is None:
            info = ""
        if torch.any(torch.isnan(tensor)):
            print(f"Input tensor contains nan: {info}")
            return False
        elif torch.any(torch.isinf(tensor)):
            print(f"Input tensor contains inf: {info}")
            return False
    return True

def revise_tensor(tensor, info=None):
    if not check_tensor(tensor, info):
        return torch.nan_to_num(tensor, nan=0.0, posinf=None, neginf=None)
    else:
        return tensor
