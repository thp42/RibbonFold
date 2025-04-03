import os, sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle, gzip
from .common_modules import *

def split_af2_params(af2_params):
    extra_msa_stack_params = {}
    evoformer_iteration_params = {}
    template_pairstack_params = {}
    evoformer_others = {}
    str_module_params = {}
    evo_loss_module_params = {}
    
    prefix = "alphafold/alphafold_iteration/"
    evo_prefix = prefix + "evoformer/"
    str_prefix = prefix + "structure_module/"
    evo_loss_prefix = [ prefix+module_name+"/" for module_name in ['masked_msa_head','distogram_head','predicted_lddt_head',
                                                               'predicted_aligned_error_head','experimentally_resolved_head'] ]
    evo_loss_prefix = tuple(evo_loss_prefix)

    for name, param in af2_params.items():
        if name.startswith(evo_prefix):
            p1 = evo_prefix+"extra_msa_stack/" # extra_msa_stack_params
            p2 = evo_prefix+"evoformer_iteration/" # evoformer_iteration_params
            p3 = evo_prefix+"template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/" # template_pairstack_params
            p3_2 = evo_prefix + "template_embedding/single_template_embedding/template_embedding_iteration/"
            if name.startswith(p1):
                extra_msa_stack_params[ name[len(p1):] ] = param
            elif name.startswith(p2):
                evoformer_iteration_params[ name[len(p2):] ] = param
            elif name.startswith(p3):
                template_pairstack_params[ name[len(p3):] ] = param
            elif name.startswith(p3_2):
                template_pairstack_params[ name[len(p3_2):] ] = param
            else:
                evoformer_others[ name[len(evo_prefix):] ] = param
        elif name.startswith(str_prefix):
            str_module_params[ name[len(str_prefix):] ] = param
        elif name.startswith(evo_loss_prefix):
            evo_loss_module_params[ name[len(prefix):] ] = param
    
    for k in list(evoformer_others.keys()):
        to_remove = '~_relative_encoding/'
        if k.startswith(to_remove):
            evoformer_others[ k[len(to_remove):] ] = evoformer_others[k]
            del evoformer_others[k]
    
    return {
        'extra_msa_stack_params': extra_msa_stack_params,
        'evoformer_iteration_params': evoformer_iteration_params,
        'template_pairstack_params': template_pairstack_params,
        'evoformer_others': evoformer_others,
        'str_module_params': str_module_params,
        'evo_loss_module_params': evo_loss_module_params
    }

def get_torch_module_weight(module, module_names, i=0):
    """
    Get parameter weight obj from module
    """
    if len(module_names) - 1 == i:
        if isinstance(module, LayerNorm):
            if module_names[i] == 'scale':
                return module.weight
            elif module_names[i] == 'offset':
                return module.bias
            else:
                raise RuntimeError(f"Unexpected name: {module_names[i]}")
        return getattr(module, module_names[i])
    else:
        return get_torch_module_weight( getattr(module, module_names[i]), module_names, i+1 )

def copy_weight_to_module(module, af_params, index=None, exclude_prefix=None, model_param_recoder=None):
    """
    Copy the AF2 params to the PyTorch modules
    
    Parameters
    ----------
    module:         torch.nn.Module
    af_params:      Dict of AF2 params
    index:          Index of slice of parameters
    exclude_prefix: Exclude some parameters with name prefix
    model_param_recoder: Record which parameters are loaded
    """
    tensor_count = 0
    for name, param in af_params.items():
        if isinstance(exclude_prefix, tuple) and name.startswith(exclude_prefix):
            continue
        items = name.split('//')
        assert len(items) == 2
        module_names = items[0].split('/') + [ items[1] ]
        try:
            pytorch_param_obj = get_torch_module_weight(module, module_names)
        except AttributeError:
            print(f"Warning: {name} weight obj not found in alphafold_model", file=sys.stderr)
            continue
        if index is not None:
            param_copy = param[index]
        else:
            param_copy = param
        af2_param = torch.tensor(param_copy, dtype=pytorch_param_obj.dtype)
        with torch.no_grad():
            try:
                pytorch_param_obj.copy_( af2_param )
            except RuntimeError as e:
                print(f"Error: {name} af_param={param.shape} pytorch_param={pytorch_param_obj.shape}", str(e), file=sys.stderr)
                continue
        
        tensor_count += 1
        if model_param_recoder is not None:
            model_param_recoder[ id(pytorch_param_obj) ] = 'loaded!'
    
    return tensor_count

def load_AlphaFoldIteration_params(module, af2_param_file):
    """
    Load the AF2 parameters to Evoformer.AlphaFoldIteration module
    """
    af2_params_raw = np.load(open(af2_param_file, 'rb'))
    af_params_dict = split_af2_params(af2_params_raw)
    model_param_recoder = {id(p):n for n,p in module.named_parameters()}
    
    tensor_count = 0
    # Load extra_msa_stack_params
    for i in range(4):
        tensor_count += copy_weight_to_module(module.evoformer.extra_msa_stack[i], 
                                              af_params_dict['extra_msa_stack_params'], 
                                              index=i, 
                                              model_param_recoder=model_param_recoder)
    # Load evoformer_iteration_params
    for i in range(48):
        tensor_count += copy_weight_to_module(module.evoformer.evoformer_iteration[i], 
                                             af_params_dict['evoformer_iteration_params'], 
                                             index=i, 
                                             model_param_recoder=model_param_recoder)
    # Load single_template_embedding
    for i in range(2):
        if not hasattr(module.evoformer, 'template_embedding'):
            break
        sub_module = module.evoformer.template_embedding.single_template_embedding
        if hasattr(sub_module, 'template_embedding_iteration'):
            sub_module = sub_module.template_embedding_iteration[i]
        elif hasattr(sub_module, 'template_pair_stack'):
            sub_module = sub_module.template_pair_stack._TemplatePairStack__layer_stack_no_state[i]
        else:
            raise RuntimeError("Error: expect module.evoformer.template_embedding.single_template_embedding.template_embedding_iteration or module.evoformer.template_embedding.single_template_embedding.template_pair_stack, but no one found")
        tensor_count += copy_weight_to_module(sub_module, 
                                              af_params_dict['template_pairstack_params'], 
                                              index=i, 
                                              model_param_recoder=model_param_recoder)
    #return af_params_dict['evoformer_others']
    # Load other params
    tensor_count += copy_weight_to_module(module.evoformer, 
                                          af_params_dict['evoformer_others'], 
                                          index=None, 
                                          exclude_prefix=None,
                                          model_param_recoder=model_param_recoder)
    
    # Load Evo loss
    tensor_count += copy_weight_to_module(module, 
                                          af_params_dict['evo_loss_module_params'], 
                                          index=None, 
                                          exclude_prefix=None,
                                          model_param_recoder=model_param_recoder)
    
    tensor_count += copy_weight_to_module(module.structure_module, 
                                          af_params_dict['str_module_params'], 
                                          index=None, 
                                          exclude_prefix=None, 
                                          model_param_recoder=model_param_recoder)
    
    print(f"Finish: {tensor_count} tensors are loaded!", file=sys.stderr)
    ## Check if all params are loaded
    for p_id,name in model_param_recoder.items():
        if name != 'loaded!':
            print(f"Warning: {name} not loaded", file=sys.stderr)

            
