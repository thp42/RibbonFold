import os, sys, pickle
import numpy as np
from typing import Dict, Optional
from . import quat_affine, r3
import torch.nn.functional as F


from alphafold.model import config
from alphafold.model.tf import proteins_dataset, input_pipeline
import tensorflow
import tensorflow.compat.v1 as tf
import numpy as np

os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1' 
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ["NUM_INTER_THREADS"]="1"
os.environ["NUM_INTRA_THREADS"]="1"
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


import gzip, copy
from pathlib import Path
import shutil
import torch

from . import all_atom, residue_constants

# from path_config import PROJECT_ROOT
from pathlib import Path
PROJECT_ROOT = Path.cwd()

# KALIGN = shutil.which("kalign")
# assert KALIGN is not None, "Error: kalign not found in PATH"

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'openfold/'))
import openfold.data.data_transforms as data_transforms


############################
### Features dtype and shape
############################

num_ens_placeholder = 'num of ens'
num_poly_placeholder = 'num of polys'
num_res_placeholder = 'num of res'
num_seq_placeholder = 'num of seq'
num_templ_placeholder = 'num of templates'

seq_features_X = { 
    'target_feat':              (torch.float, [num_ens_placeholder, num_res_placeholder, 22]),
    'msa_feat':                 (torch.float, [num_ens_placeholder, num_seq_placeholder, num_res_placeholder, 49]),
    'seq_mask':                 (torch.float, [num_ens_placeholder, num_res_placeholder]),
    'residue_index':            (torch.long,  [num_ens_placeholder, num_res_placeholder]),
    'msa_mask':                 (torch.float,  [num_ens_placeholder, num_seq_placeholder, num_res_placeholder]),
    'extra_msa':                (torch.float,  [num_ens_placeholder, num_res_placeholder, num_res_placeholder]),
    'extra_msa_mask':           (torch.float, [num_ens_placeholder, num_res_placeholder, num_res_placeholder]),
    'extra_has_deletion':       (torch.long, [num_ens_placeholder, num_res_placeholder, num_res_placeholder]),
    'extra_deletion_value':     (torch.float, [num_ens_placeholder, num_res_placeholder, num_res_placeholder]),
    'seq_length':               (torch.long,  [num_ens_placeholder]),
    'aatype':                   (torch.long,  [num_ens_placeholder, num_res_placeholder]),
    'atom14_atom_exists':       (torch.long,  [num_ens_placeholder, num_res_placeholder, 14]),
    'atom37_atom_exists':       (torch.long,  [num_ens_placeholder, num_res_placeholder, 37]),
    'residx_atom37_to_atom14':  (torch.long,  [num_ens_placeholder, num_res_placeholder, 37]),
    'residx_atom14_to_atom37':  (torch.long,  [num_ens_placeholder, num_res_placeholder, 14]),
    'asym_id':                  (torch.long, [num_ens_placeholder, num_res_placeholder]),
    'sym_id':                   (torch.long, [num_ens_placeholder, num_res_placeholder]),
    'entity_id':                (torch.long, [num_ens_placeholder, num_res_placeholder]),
    'num_sym':                  (torch.long, [num_ens_placeholder, num_res_placeholder])
}

templ_features_X = {
    'template_mask':                (torch.float, [num_ens_placeholder, num_templ_placeholder]),
    'template_aatype':              (torch.float, [num_ens_placeholder, num_templ_placeholder, num_res_placeholder]),
    'template_pseudo_beta_mask':    (torch.float, [num_ens_placeholder, num_templ_placeholder, num_res_placeholder]),
    'template_pseudo_beta':         (torch.float, [num_ens_placeholder, num_templ_placeholder, num_res_placeholder, 3]),
    'template_all_atom_positions':  (torch.float, [num_ens_placeholder, num_templ_placeholder, num_res_placeholder, 37, 3]),
    'template_all_atom_masks':      (torch.float, [num_ens_placeholder, num_templ_placeholder, num_res_placeholder, 37])
}

syth_templ_features_X = {
    'template_mask':                (torch.float, [num_ens_placeholder, num_templ_placeholder]),
    'template_aatype':              (torch.float, [num_ens_placeholder, num_templ_placeholder, num_res_placeholder]),
    'template_pseudo_beta_mask':    (torch.float, [num_ens_placeholder, num_templ_placeholder, num_res_placeholder]),
    'template_pair_dist':           (torch.float, [num_ens_placeholder, num_templ_placeholder, num_res_placeholder, num_res_placeholder]),
    'template_pair_orient':         (torch.float, [num_ens_placeholder, num_templ_placeholder, num_res_placeholder, num_res_placeholder, 3]),

}

features_Y = {
     'seq_mask':                    (torch.float, [num_res_placeholder]),
     'residue_index':               (torch.long, [num_res_placeholder]),
     'asym_id':                     (torch.long, [num_res_placeholder]),
     'sym_id':                      (torch.long, [num_res_placeholder]),
     'entity_id':                   (torch.long, [num_res_placeholder]),
     'num_sym':                     (torch.long, [num_res_placeholder]),
     'aatype':                      (torch.long, [num_res_placeholder]),
     'atom14_atom_exists':          (torch.long, [num_res_placeholder, 14]),
     'atom37_atom_exists':          (torch.long, [num_res_placeholder, 37]),
     'residx_atom37_to_atom14':     (torch.long, [num_res_placeholder, 37]),
     'pseudo_beta':                 (torch.float,[num_res_placeholder, 3]),
     'pseudo_beta_mask':            (torch.float, [num_res_placeholder]),
     'all_atom_positions':          (torch.float,[num_res_placeholder, 37, 3]),
     'all_atom_mask':               (torch.float, [num_res_placeholder, 37]),
     'backbone_affine_tensor':      (torch.float, [num_res_placeholder, 7]),
     'backbone_affine_mask':        (torch.float, [num_res_placeholder]),
     'residx_atom14_to_atom37':     (torch.long, [num_res_placeholder, 14]),
     'true_msa':                    (torch.long, [num_seq_placeholder, num_res_placeholder]),
     'bert_mask':                   (torch.float, [num_seq_placeholder, num_res_placeholder]),
     'atom14_gt_positions':         (torch.float, [num_res_placeholder, 14, 3]),
     'atom14_alt_gt_positions':     (torch.float, [num_res_placeholder, 14, 3]),
     'atom14_atom_is_ambiguous':    (torch.long, [num_res_placeholder, 14]),
     'atom14_gt_exists':            (torch.long, [num_res_placeholder, 14]),
     'atom14_alt_gt_exists':        (torch.long, [num_res_placeholder, 14]),
     'rigidgroups_gt_frames':       (torch.float, [num_res_placeholder, 8, 4, 4]),
     'rigidgroups_gt_exists':       (torch.long, [num_res_placeholder, 8]),
     'rigidgroups_group_exists':    (torch.long, [num_res_placeholder, 8]),
     'rigidgroups_group_is_ambiguous': (torch.long, [num_res_placeholder, 8]),
     'rigidgroups_alt_gt_frames':   (torch.float, [num_res_placeholder, 8, 4, 4]),
     'chi_mask':                    (torch.float, [num_res_placeholder, 4]),
     'chi_angles':                  (torch.float, [num_res_placeholder, 4])
}


features_YS = {
     'seq_mask':                    (torch.float, [num_poly_placeholder, num_res_placeholder]),
     'residue_index':               (torch.long, [num_poly_placeholder, num_res_placeholder]),
     'asym_id':                     (torch.long, [num_poly_placeholder, num_res_placeholder]),
     'sym_id':                      (torch.long, [num_poly_placeholder, num_res_placeholder]),
     'entity_id':                   (torch.long, [num_poly_placeholder, num_res_placeholder]),
     'num_sym':                     (torch.long, [num_poly_placeholder, num_res_placeholder]),
     'aatype':                      (torch.long, [num_poly_placeholder, num_res_placeholder]),
     'atom14_atom_exists':          (torch.long, [num_poly_placeholder, num_res_placeholder, 14]),
     'atom37_atom_exists':          (torch.long, [num_poly_placeholder, num_res_placeholder, 37]),
     'residx_atom37_to_atom14':     (torch.long, [num_poly_placeholder, num_res_placeholder, 37]),
     'pseudo_beta':                 (torch.float,[num_poly_placeholder, num_res_placeholder, 3]),
     'pseudo_beta_mask':            (torch.float, [num_poly_placeholder, num_res_placeholder]),
     'all_atom_positions':          (torch.float,[num_poly_placeholder, num_res_placeholder, 37, 3]),
     'all_atom_mask':               (torch.float, [num_poly_placeholder, num_res_placeholder, 37]),
     'backbone_affine_tensor':      (torch.float, [num_poly_placeholder, num_res_placeholder, 7]),
     'backbone_affine_mask':        (torch.float, [num_poly_placeholder, num_res_placeholder]),
     'residx_atom14_to_atom37':     (torch.long, [num_poly_placeholder, num_res_placeholder, 14]),
     'true_msa':                    (torch.long, [num_poly_placeholder, num_seq_placeholder, num_res_placeholder]),
     'bert_mask':                   (torch.float, [num_poly_placeholder, num_seq_placeholder, num_res_placeholder]),
     'atom14_gt_positions':         (torch.float, [num_poly_placeholder, num_res_placeholder, 14, 3]),
     'atom14_alt_gt_positions':     (torch.float, [num_poly_placeholder, num_res_placeholder, 14, 3]),
     'atom14_atom_is_ambiguous':    (torch.long, [num_poly_placeholder, num_res_placeholder, 14]),
     'atom14_gt_exists':            (torch.long, [num_poly_placeholder, num_res_placeholder, 14]),
     'atom14_alt_gt_exists':        (torch.long, [num_poly_placeholder, num_res_placeholder, 14]),
     'rigidgroups_gt_frames':       (torch.float, [num_poly_placeholder, num_res_placeholder, 8, 4, 4]),
     'rigidgroups_gt_exists':       (torch.long, [num_poly_placeholder, num_res_placeholder, 8]),
     'rigidgroups_group_exists':    (torch.long, [num_poly_placeholder, num_res_placeholder, 8]),
     'rigidgroups_group_is_ambiguous': (torch.long, [num_poly_placeholder, num_res_placeholder, 8]),
     'rigidgroups_alt_gt_frames':   (torch.float, [num_poly_placeholder, num_res_placeholder, 8, 4, 4]),
     'chi_mask':                    (torch.float, [num_poly_placeholder, num_res_placeholder, 4]),
     'chi_angles':                  (torch.float, [num_poly_placeholder, num_res_placeholder, 4])
}



def get_batchX(batch, use_template=True, use_syth_template=False):
    def valid_shape(obj_shape, valid_shape):
        if len(obj_shape) != len(valid_shape):
            return False
        for s1,s2 in zip(obj_shape, valid_shape):
            if isinstance(s2, int) and s1 != s2:
                return False
        return True
    batchX = {}
    for name in seq_features_X:
        if name not in batch:
            print( f"Error: {name} not in batch" )
            continue
        dtype, shape = seq_features_X[name]
        ## Check Shape
        assert valid_shape(batch[name].shape, shape), f"Error: {name} expect shape {shape}, but got {batch[name].shape}"
        batchX[name] = torch.tensor( batch[name], dtype=dtype )
    if use_template:
        for name in templ_features_X:
            if name not in batch:
                print( f"Error: {name} not in batch" )
                continue
            dtype, shape = templ_features_X[name]
            ## Check Shape
            assert valid_shape(batch[name].shape, shape), f"Error: {name} expect shape {shape}, but got {batch[name].shape}"
            batchX[name] = torch.tensor( batch[name], dtype=dtype )
    if use_syth_template:
        for name in syth_templ_features_X:
            if name not in batch:
                print( f"Error: {name} not in batch" )
                continue
            dtype, shape = syth_templ_features_X[name]
            ## Check Shape
            assert valid_shape(batch[name].shape, shape), f"Error: {name} expect shape {shape}, but got {batch[name].shape}"
            batchX[name] = torch.tensor( batch[name], dtype=dtype )
    return batchX

def get_batchY(batch, exclude=[]):
    def valid_shape(obj_shape, valid_shape):
        if len(obj_shape) != len(valid_shape):
            return False
        for s1,s2 in zip(obj_shape, valid_shape):
            if isinstance(s2, int) and s1 != s2:
                return False
        return True
    batchY = {}
    for name in features_Y:
        if name in exclude:
            continue
        assert name in batch, f"Error: {name} not in batch"
        dtype, shape = features_Y[name]
        ## Check Shape
        assert valid_shape(batch[name].shape, shape), f"Error: {name} expect shape {shape}, but got {batch[name].shape}"
        if isinstance(batch[name], np.ndarray):
            batchY[name] = torch.from_numpy( batch[name] ).to(dtype)
        elif isinstance(batch[name], torch.Tensor):
            batchY[name] = batch[name].to(dtype)
        else:
            raise f"Expect type np.ndarray or torch.Tensor, but got {type(batch[name])}"
    return batchY

def get_batchYS(batch, exclude=[]):
    def valid_shape(obj_shape, valid_shape):
        if len(obj_shape) != len(valid_shape):
            return False
        for s1,s2 in zip(obj_shape, valid_shape):
            if isinstance(s2, int) and s1 != s2:
                return False
        return True
    batchYS = {}
    for name in features_YS:
        if name in exclude:
            continue
        assert name in batch, f"Error: {name} not in batch"
        dtype, shape = features_YS[name]
        ## Check Shape
        assert valid_shape(batch[name].shape, shape), f"Error: {name} expect shape {shape}, but got {batch[name].shape}"
        if isinstance(batch[name], np.ndarray):
            batchYS[name] = torch.from_numpy( batch[name] ).to(dtype)
        elif isinstance(batch[name], torch.Tensor):
            batchYS[name] = batch[name].to(dtype)
        else:
            raise f"Expect type np.ndarray or torch.Tensor, but got {type(batch[name])}"
    return batchYS


############################
## Other functions
############################

# def parse_cif_file(cif_file, pdb_id, chain_id, query_seq):
#     """
#     Parse a cif file
#     """
#     from alphafold.data import mmcif_parsing, templates
#     from alphafold.data.templates import NoChainsError, NoAtomDataInTemplateError, TemplateAtomMaskAllZerosError
#
#     ######### Read a cif file
#     if cif_file.endswith('.gz'):
#         mmcif_string = gzip.open(cif_file).read().decode()
#     else:
#         mmcif_string = open(cif_file).read()
#
#     parsing_result = mmcif_parsing.parse(file_id=pdb_id, mmcif_string=mmcif_string)
#     if parsing_result is None or parsing_result.mmcif_object is None:
#         return None
#
#     ######### Generate the mapping
#     mapping = { i:i for i in range(len(query_seq)) }
#
#     ######### Extract
#     try:
#         features, realign_warning = templates._extract_template_features(
#             mmcif_object=parsing_result.mmcif_object,
#             pdb_id=pdb_id,
#             mapping=mapping,
#             template_sequence=query_seq,
#             query_sequence=query_seq,
#             template_chain_id=chain_id,
#             kalign_binary_path=KALIGN)
#     except (NoChainsError, NoAtomDataInTemplateError, TemplateAtomMaskAllZerosError) as e:
#         return None
#
#     return features


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    return data_transforms.pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks)


def get_protein_features_from_pos(query_seq, all_atom_positions, all_atom_masks):
    """
    Parameters
    --------------------
    batch: dict
        -- query_seq: str
        -- all_atom_positions: [N_res, 37, 3]
        -- all_atom_masks: [N_res, 37]
        
    Return
    --------------------
    batch: dict, the following features are added
        -- pseudo_beta:            [N_res, 3]
        -- pseudo_beta_mask:       [N_res]
        -- all_atom_positions:     [N_res, 37, 3]
        -- all_atom_mask:          [N_res, 37]
        -- backbone_affine_tensor: [N_res, 7]
        -- backbone_affine_mask:   [N_res]
    """

    # all_atom_positions = torch.from_numpy(all_atom_positions)
    # all_atom_masks = torch.from_numpy(all_atom_masks)

    assert all_atom_positions.ndim == 3
    assert all_atom_masks.ndim == 2
    assert len(query_seq) == all_atom_positions.shape[0] == all_atom_masks.shape[0]

    num_res = len(query_seq)
    aatype = np.array([residue_constants.restypes_with_x_and_gap.index(d) for d in query_seq])

    batch = {}
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(torch.from_numpy(aatype), torch.from_numpy(all_atom_positions), torch.from_numpy(all_atom_masks))
    batch['pseudo_beta'] = pseudo_beta
    batch['pseudo_beta_mask'] = pseudo_beta_mask
    batch['all_atom_positions'] = all_atom_positions
    batch['all_atom_mask'] = all_atom_masks
    all_atom_mask = batch['all_atom_mask']
    n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
    rot, trans = quat_affine.make_transform_from_reference(n_xyz=torch.from_numpy(batch['all_atom_positions'][:, n]),
                                                           ca_xyz=torch.from_numpy(batch['all_atom_positions'][:, ca]),
                                                           c_xyz=torch.from_numpy(batch['all_atom_positions'][:, c]))
    quaternion = quat_affine.rot_to_quat(rot, unstack_inputs=True)
    
    rot = rot.numpy()
    trans = trans.numpy()
    quaternion = quaternion.numpy()

    batch['backbone_affine_tensor'] = np.concatenate([quaternion, trans], 1)
    batch['backbone_affine_mask'] = ((all_atom_mask[:, n] + all_atom_mask[:, ca] + all_atom_mask[:, c]) == 3).astype(np.float32)
    batch['backbone_affine_tensor'] = torch.from_numpy(batch['backbone_affine_tensor'])
    batch['backbone_affine_mask'] = torch.from_numpy(batch['backbone_affine_mask'])
    
    tmp_batch = {
        'aatype': torch.from_numpy(aatype).long(),
        'all_atom_mask': torch.from_numpy(batch['all_atom_mask']).long(),
        'all_atom_positions': torch.from_numpy(batch['all_atom_positions'])
    }
    batch.update( make_atom14_positions(tmp_batch) )
    
    ############ O1 features
    o1 = data_transforms.atom37_to_frames(batch)
    o1 = { k:v.numpy() for k,v in o1.items() }
    batch.update(o1)
    
    ############ O2 features
    tmp_batch = {
        'aatype': torch.from_numpy(batch['aatype'])[None,...].long(),
        'all_atom_mask': torch.from_numpy(batch['all_atom_mask'])[None,...].long(),
        'all_atom_positions': torch.from_numpy(batch['all_atom_positions'])[None,...]
    }
    o2 = data_transforms.atom37_to_torsion_angles(tmp_batch)
    torsion_angles = torch.atan2(o2['torsion_angles_sin_cos'][0,:,:,0], 
                                 o2['torsion_angles_sin_cos'][0,:,:,1])
    batch['chi_angles'] = torsion_angles[..., 3:].numpy()
    batch['chi_mask'] = o2['torsion_angles_mask'][0,:,3:].numpy()
    
    return batch



def make_atom14_positions(prot):
    prot = data_transforms.make_atom14_masks(prot)
    prot = data_transforms.make_atom14_positions(prot)
    return prot



def seq_to_batchX(seq):
    """
    Covert the protein sequence to dict feed for model
    
    Parameters
    ------------
    seq: str
        Input protein sequence
    
    Return
    -----------
    batchX: dict
        -- seq_mask: [1, N_res]
        -- aatype: [1, N_res]
        -- atom14_atom_exists: [1, N_res, 14]
        -- atom37_atom_exists: [1, N_res, 37]
        -- target_feat: [1, N_res, 22]
        -- residue_index: [1, N_res]
    """
    
    aatype = np.array([residue_constants.restypes_with_x_and_gap.index(r) for r in seq])
    
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[residue_constants.restype_1to3[rt]]

        restype_atom14_to_atom37.append([(residue_constants.atom_order[name] if name else 0) for name in atom_names])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0) for name in residue_constants.atom_types])

        restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    # Add dummy mapping for restype 'UNK'.
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    # Create the corresponding mask.
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    # Create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein.
    residx_atom14_to_atom37 = restype_atom14_to_atom37[aatype]
    atom14_atom_exists = restype_atom14_mask[aatype]
    residx_atom37_to_atom14 = restype_atom37_to_atom14[aatype]

    atom37_atom_exists = restype_atom37_mask[aatype]
    
    batchX = {
        'seq_mask': torch.ones(len(seq))[None,...],
        'aatype': torch.tensor(aatype[None,...]),
        'atom14_atom_exists': torch.tensor(atom14_atom_exists[None,...]),
        'atom37_atom_exists': torch.tensor(atom37_atom_exists[None,...]),
        'residx_atom37_to_atom14': torch.tensor(residx_atom37_to_atom14[None,...])
    }
    
    batchX['target_feat'] = F.one_hot(batchX['aatype']+1, num_classes=22).float()
    batchX['residue_index'] = torch.tensor(range(len(seq)))[None,...]
    
    return batchX



def get_config(max_extra_msa=5120, crop_size=384, max_msa_clusters=512, num_ensemble=4, num_recycle=1, use_templates=False):
    """
    Get config object
    """
    model_config = config.model_config("model_1")
    data_config = model_config.data
    data_config.common.reduce_msa_clusters_by_max_templates = False
    data_config.common.max_extra_msa = max_extra_msa
    data_config.eval.crop_size = crop_size
    data_config.eval.num_ensemble = num_ensemble
    data_config.common.num_recycle = num_recycle
    data_config.common.resample_msa_in_recycling = False
    data_config.common.use_templates = use_templates
    data_config.eval.max_msa_clusters = max_msa_clusters
    
    feature_names = copy.deepcopy(data_config.common.unsupervised_features)
    if data_config.common.use_templates:
        feature_names += data_config.common.template_features
    
    return data_config, feature_names


def get_permitted_crop_start(pdb_ft, crop_size, min_exp_resolved_res_num):
    """
    Get a list of crop start positions to constraint the TF code cropping
    """
    exp_resolved_mask = (pdb_ft['all_atom_mask'].sum(1) > 0).astype(np.float32)
    assert crop_size > min_exp_resolved_res_num

    if exp_resolved_mask.shape[0] <= crop_size:
        permitted_crop_start = [0]
    else:
        exp_array = np.zeros_like(exp_resolved_mask)
        idx = exp_array.shape[0] - crop_size
        exp_array[idx] = exp_resolved_mask[idx:idx+crop_size].sum()
        idx -= 1
        while idx >= 0:
            exp_array[idx] = exp_array[idx+1] - exp_resolved_mask[idx+crop_size] + exp_resolved_mask[idx]
            idx -= 1
        permitted_crop_start = np.where(exp_array >= min_exp_resolved_res_num)[0].tolist()
    return permitted_crop_start


def process_np_example(np_example, data_config, feature_names, eagerly=True, seed=0):
    """
    Process the dict according the AlphaFold2 pipeline
    """
    np_example['deletion_matrix'] = np_example['deletion_matrix_int'].astype(np.float32)

    if not eagerly:
        tf.config.run_functions_eagerly(False)
        ## Define the graph
        tf_graph = tf.Graph()
        with tf_graph.as_default(), tf.device('/device:CPU:0'):
            tf.compat.v1.set_random_seed(seed)
            tensorflow.random.set_seed(seed)
            tensor_dict = proteins_dataset.np_to_tensor_dict(np_example=np_example, features=feature_names)
            if 'permitted_crop_start' in np_example:
                tensor_dict['permitted_crop_start'] = tf.constant(np_example['permitted_crop_start'])
            if 'crop_mask' in np_example:
                tensor_dict['crop_mask'] = tf.constant(np_example['crop_mask'])
            processed_batch = input_pipeline.process_tensors_from_config(tensor_dict, data_config)
        tf_graph.finalize()

        ## Run the graph
        config = tf.ConfigProto(device_count = {'GPU': 0, "CPU": 10})
        with tf.Session(graph=tf_graph, config=config) as sess:
            features = sess.run(processed_batch)
    else:
        tf.config.run_functions_eagerly(True)
        tensor_dict = proteins_dataset.np_to_tensor_dict(np_example=np_example, features=feature_names)
        if 'permitted_crop_start' in np_example:
            tensor_dict['permitted_crop_start'] = tf.constant(np_example['permitted_crop_start'])
        if 'crop_mask' in np_example:
            tensor_dict['crop_mask'] = tf.constant(np_example['crop_mask'])
        features = input_pipeline.process_tensors_from_config(tensor_dict, data_config)
        features = {k:v.numpy() for k,v in features.items()}

    return features



