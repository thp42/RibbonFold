import os, sys
import torch
import numpy as np
import tempfile, subprocess
import scipy, scipy.stats
from . import residue_constants
from .data_processing import save_as_pdb, ds_to_cuda

#########################
### Functions to calculate the LDDT
#########################

def _get_flattened(dmap):
    """
    Get the elements of the upper triagular of the matrix
    
    [0,1,1,1,1]
    [0,0,1,1,1]
    [0,0,0,1,1]
    [0,0,0,0,1]
    [0,0,0,0,0]
    
    dmap: numpy.ndarray
        [ N_res*(N_res-1)/2 ] or [N_res, N_res]
    """
    assert isinstance(dmap, np.ndarray)
    assert dmap.ndim == 1 or dmap.ndim == 2
    
    if dmap.ndim == 1:
        return dmap
    elif dmap.ndim == 2:
        return dmap[np.triu_indices_from(dmap, k=1)]

def _get_separations(dmap):
    """
    Get the distance between the row indice and col indice
    """
    t_indices = np.triu_indices_from(dmap, k=1)
    separations = np.abs(t_indices[0] - t_indices[1])
    return separations

def _get_sep_thresh_b_indices(dmap, thresh, comparator):
    """
    Return a 1D boolean array indicating where the sequence 
    separation in the upper triangle meets the threshold comparison
    
    dmap: numpy.ndarray
    thresh: float
    comparator: gt, lt, ge, le
    """
    assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
    separations = _get_separations(dmap)
    if comparator == 'gt':
        threshed = separations > thresh
    elif comparator == 'lt':
        threshed = separations < thresh
    elif comparator == 'ge':
        threshed = separations >= thresh
    elif comparator == 'le':
        threshed = separations <= thresh
    return threshed

def _get_dist_thresh_b_indices(dmap, thresh, comparator):
    """
    Return a 1D boolean array indicating where the distance in the
    upper triangle meets the threshold comparison
    
    dmap: numpy.ndarray
    thresh: float
    comparator: gt, lt, ge, le
    """
    assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
    dmap_flat = _get_flattened(dmap)
    if comparator == 'gt':
        threshed = dmap_flat > thresh
    elif comparator == 'lt':
        threshed = dmap_flat < thresh
    elif comparator == 'ge':
        threshed = dmap_flat >= thresh
    elif comparator == 'le':
        threshed = dmap_flat <= thresh
    return threshed


def calculate_LDDT(true_map, pred_map, R=15.0, sep_thresh=-1, T_set=[0.5, 1, 2, 4]):
    """
    lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests.
    doi: 10.1093/bioinformatics/btt473.
    
    Parameters
    -----------------------
    true_map: numpy.ndarray
        Distance map of the true structure
    pred_map: numpy.ndarray
        Distance map of the predicted structure
    R: float
        The threshold of the distance between residual pairs to consider
    sep_thresh: int
        The number of residues of seperation to consider
    T_set: list
        Maximum difference of distance map
    """
    assert true_map.ndim == 2
    assert pred_map.ndim == 2
    
    # Helper for number preserved in a threshold
    def get_n_preserved(ref_flat, mod_flat, thresh):
        err = np.abs(ref_flat - mod_flat)
        n_preserved = (err < thresh).sum()
        return n_preserved
    
    # flatten upper triangles
    true_flat_map = _get_flattened(true_map)
    pred_flat_map = _get_flattened(pred_map)
    
    # Find set L
    S_thresh_indices = _get_sep_thresh_b_indices(true_map, sep_thresh, 'gt')
    R_thresh_indices = _get_dist_thresh_b_indices(true_flat_map, R, 'lt')
    
    L_indices = S_thresh_indices & R_thresh_indices
    
    true_flat_in_L = true_flat_map[L_indices]
    pred_flat_in_L = pred_flat_map[L_indices]
    
    # Number of pairs in L
    L_n = L_indices.sum()
    
    # Calculated lDDT
    preserved_fractions = []
    for _thresh in T_set:
        _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, _thresh)
        _f_preserved = _n_preserved / (L_n + 0.00001)
        preserved_fractions.append(_f_preserved)
    
    lDDT = np.mean(preserved_fractions)
    return round(lDDT, 4)

def lddt_from_prediction(ret, batchY):
    """
    Calculate the lDDT from prediction result
    
    Parameters
    ------------
    ret: dict, Return from Structure Module
        -- final_atom_positions: [N_res, 37, 3]
        -- final_atom_mask: [N_res, 37]
    batchY: dict
        -- pseudo_beta: [N_res, 3]
        -- pseudo_beta_mask: [N_res]
    """
    
    assert ret['final_atom_positions'].ndim == 3
    assert ret['final_atom_mask'].ndim == 2
    assert batchY['pseudo_beta'].ndim == 2
    assert batchY['pseudo_beta_mask'].ndim == 1
    
    CB_ind = residue_constants.atom_types.index('CB')
    CA_ind = residue_constants.atom_types.index('CA')
    
    ## True distogram
    true_position = batchY['pseudo_beta'].cpu().detach().numpy()
    true_mask     = batchY['pseudo_beta_mask'].cpu().detach().numpy()
    
    ## Predicted distogram
    atom_CB_pred = ret['final_atom_positions'][:, CB_ind].cpu().detach().numpy()
    atom_CA_pred = ret['final_atom_positions'][:, CA_ind].cpu().detach().numpy()
    atom_CB_mask = ret['final_atom_mask'][:, CB_ind].cpu().detach().numpy()
    pred_position = np.where(atom_CB_mask[:, None]==1, atom_CB_pred, atom_CA_pred)
    pred_mask = (ret['final_atom_mask'].sum(1) > 0).float().cpu().detach().numpy()
    
    mask = (true_mask > 0) & (pred_mask > 0)
    
    true_dist = np.sqrt(np.sum((true_position[None, mask, :] - true_position[mask, None, :]) ** 2, axis=2))
    pred_dist = np.sqrt(np.sum((pred_position[None, mask, :] - pred_position[mask, None, :]) ** 2, axis=2))
    
    ## Calculate LDDT
    lddt = calculate_LDDT(true_dist, pred_dist)
    
    return lddt

#########################
### Functions to get RMSD, TM-score, GDT-TS-score, lDDT
#########################

def _parse_TMscore_outputs(output_txt):
    """
    output_txt: str
        Print by TMscore
    """
    
    metrics = {}
    lines = output_txt.strip().split('\n')
    for line in lines:
        if line.startswith('Number of residues in common'):
            metrics['res_num'] = int(line.split('=')[1].strip())
        elif line.startswith('RMSD of'):
            metrics['RMSD'] = float(line.split('=')[1].strip())
        elif line.startswith('TM-score'):
            metrics['TM-score'] = float(line.split('=')[1].split('(')[0].strip())
        elif line.startswith('GDT-TS-score'):
            metrics['GDT-TS-score'] = float(line.split('=')[1].split('%')[0].strip())
    return metrics

def metrics_from_prediction(ret, batchX, batchY):
    """
    Get the RMSD, TM-score, GDT-TS-score, lDDT
    
    Parameters
    ------------
    ret: dict. Return by StructureModule
        -- final_atom_positions: [N_res, 37, 3]
        -- final_atom_mask: [N_res, 37]
        
    batchX: dict. Input dict
        -- aatype: [1, N_res]
        -- residue_index: [1, N_res]
        
    batchY: dict. Label dict
        -- all_atom_positions: [N_res, 37, 3]
        -- all_atom_mask: [N_res, 37]
        -- pseudo_beta: [N_res,l 3]
        -- pseudo_beta_mask: [N_res]
    
    Return
    -----------
    metrics: dict
        -- lDDT: float
        -- RMSD: float
        -- TM-score: float
        -- GDT-TS-score: float
    """
    
    import distutils.spawn
    exec_path = distutils.spawn.find_executable('TMscore')
    if exec_path is None:
        print("Please install TMscore and add to PATH: https://zhanggroup.org//TM-score/TMscore.cpp")
        return None
    
    if 'structure_module' in ret:
        ret = ret['structure_module']
    N_res = batchY['all_atom_positions'].shape[0]
    lddt = lddt_from_prediction(ret, batchY)
    
    pdb_true = tempfile.mktemp(".pdb")
    pdb_pred = tempfile.mktemp(".pdb")
    
    #### Unify mask
    pdb_mask = (batchY['all_atom_mask'].sum(1) > 0).cpu() & (ret['final_atom_mask'].sum(1) > 0).cpu()
    pdb_mask_num = pdb_mask.sum()

    mask_pred = ret['final_atom_mask'].clone()
    mask_true = batchY['all_atom_mask'].clone()
    
    mask_true[~pdb_mask, :] = 0.0
    mask_pred[~pdb_mask, :] = 0.0
    
    ret_pred = {
        'structure_module':{
            'final_atom_positions': ret['final_atom_positions'],
            'final_atom_mask': mask_pred
    }}
    save_as_pdb(batchX, ret_pred, pdb_pred)
    
    ret_std = {
        'structure_module':{
            'final_atom_positions': batchY['all_atom_positions'],
            'final_atom_mask': mask_true
    }}
    save_as_pdb(batchX, ret_std, pdb_true)
    
    output = subprocess.getoutput(f"TMscore {pdb_pred} {pdb_true}")
    metrics = _parse_TMscore_outputs(output)
    if 'res_num' in metrics:
        if metrics['res_num'] != pdb_mask_num:
            print("Warning: different number of residues between TMscore and batchX:", metrics['res_num'], pdb_mask_num)
        del metrics['res_num']
    else:
        print("Warning: res_num not in metrics")
    
    if 'RMSD' not in metrics:
        print("Warning: RMSD not in metrics")
        metrics['RMSD'] = 0.0
    if 'TM-score' not in metrics:
        print("Warning: TM-score not in metrics")
        metrics['TM-score'] = 0.0
    if 'GDT-TS-score' not in metrics:
        print("Warning: GDT-TS-score not in metrics")
        metrics['GDT-TS-score'] = 0.0
    
    metrics['lDDT'] = lddt
    
    os.remove(pdb_true)
    os.remove(pdb_pred)
    
    return metrics


def metrics_for_dataset(dataset, alphafold_call_func):
    """
    Calculate the metrics for protein from dataset
    
    Parameters
    ------------
    dataset: iterable object, each obj must be (name,batchX,batchY)
    alphafold_call_func: function to call alphafold model, with single parameter batchX
    
    Return
    ------------
    ds_metric: dict
        { name: {'RMSD': 18.763, 'TM-score': 0.4003, 'GDT-TS-score': 0.251, 'lDDT': 0.3979} }
    """
    from tqdm.auto import tqdm
    
    ds_metric = {}
    bar = tqdm(total=len(dataset), leave=False)
    for name, batchX, batchY in dataset:
        bar.update(1)
        ds_to_cuda(batchX)
        ds_to_cuda(batchY)
        with torch.no_grad():
            ret = alphafold_call_func(batchX)
        ds_metric[name] = metrics_from_prediction(ret, batchX, batchY)
    bar.close()
    return ds_metric

#########################
### For distogram
### The distogram loss (catogory accurancy is not a good metric)
#########################

def get_dist_R(ret, batchY):
    """
    Get the Pearson correlation effiencient between the predicted distogram and true distogram
    
    Parameter
    -----------
    ret: dict
        -- distogram - logits: [N_res, N_res, 64]
    batchY: dict
        -- pseudo_beta:[N_res, 37, 3]
        -- pseudo_beta_mask: [N_res, 37]
        
    Return
    -----------
    R: float
        Pearson correlation coefficient
    """
    
    import scipy, scipy.stats
    
    ds_to_cuda(batchY)
    
    ### True distogram
    positions = batchY['pseudo_beta']
    mask = batchY['pseudo_beta_mask']
    dist_true = torch.sum(torch.square( torch.unsqueeze(positions, dim=-2) - torch.unsqueeze(positions, dim=-3)), dim=-1).sqrt()
    
    mask_2d = mask[None,...] * mask[..., None]
    mask_2d[ torch.arange(mask_2d.shape[0]), torch.arange(mask_2d.shape[0]) ] = 0
    mask_2d = mask_2d.bool()
    
    dist_true[~mask_2d] = 0
    dist_true = dist_true.cpu().numpy()
    
    dist_pred = ret['distogram']['logits'].argmax(-1) * 0.3125 + 2
    dist_pred[~mask_2d] = 0
    dist_pred = dist_pred.cpu().numpy()
    
    mask_2d = mask_2d.cpu().numpy()
    
    R = scipy.stats.pearsonr(dist_pred[mask_2d], dist_true[mask_2d])[0]
    
    return round(R, 3)

#########################
### Align 2 PDB complex
#########################

def pdb_align(source_pdb, target_pdb, dest_pdb, verbose=False):
    """
    Align Source PDB to Target PDB
    
    Parameters
    -------------
    source_pdb: str
    target_pdb: str
    dest_pdb: str
        Aligned PDB
    verbose: bool
        Show command
    
    Return
    ------------
    rot: np.ndarray
        [3, 4]
    """
    import shutil, tempfile
    exec_path = shutil.which('USalign')
    if exec_path is None:
        print("Please install USalign and add to PATH: https://zhanggroup.org/US-align/bin/module/USalign.cpp")
        return None
    
    work_dir = tempfile.mkdtemp(prefix="USalign_")
    cmd = f"{exec_path} {source_pdb} {target_pdb} -mol prot -mm 1 -ter 1 -m {work_dir}/rotation.txt -o {work_dir}/out"
    if verbose:
        print(cmd)
    assert os.system(cmd) == 0, f"Command error: {cmd}"
    shutil.copyfile( f"{work_dir}/out.pdb",  dest_pdb )
    
    ### Read rotation matrix ###
    rot = [ line.strip().split()[1:] for line in open(f"{work_dir}/rotation.txt").readlines()[2:5] ]
    rot = [ [float(d[0]), float(d[1]), float(d[2]), float(d[3])] for d in rot ]
    rot = np.array(rot)
    
    shutil.rmtree(work_dir, ignore_errors=True)
    
    return rot

def rotate_3d(src_dict, tgt_dict):
    """
    Get rotated source PDB according to the target positions
    
    Parameters
    ------------
    src_dict: dict
        -- aatype: [N_res]
        -- pos: [N_res, 37, 3]
        -- mask: [N_res, 37]
    
    tgt_dict: dict
        -- aatype: [N_res]
        -- pos: [N_res, 37, 3]
        -- mask: [N_res, 37]
    
    Return
    -----------
    dest_pos: [N_res, 37, 3]
    rot: [3, 3]
    trans: [3]
    """
    assert src_dict['aatype'].ndim == 1
    assert src_dict['pos'].ndim == 3
    assert src_dict['mask'].ndim == 2
    assert tgt_dict['aatype'].ndim == 1
    assert tgt_dict['pos'].ndim == 3
    assert tgt_dict['mask'].ndim == 2
    
    import shutil, tempfile
    from alphafold.common import protein
    from src.data_processing import save_as_pdb
    
    src_pdb_file = tempfile.mktemp(prefix="rotate_3d_src_", suffix=".pdb")
    tgt_pdb_file = tempfile.mktemp(prefix="rotate_3d_src_", suffix=".pdb")
    
    input_batch = {
        'aatype': src_dict['aatype'],
        'residue_index': np.arange(src_dict['aatype'].shape[0])
    }
    predict_ret = {
        'structure_module': {
            'final_atom_mask': src_dict['mask'],
            'final_atom_positions': src_dict['pos']
        }
    }
    save_as_pdb(input_batch, predict_ret, src_pdb_file, rem_leading_feat_dim=False)
    
    input_batch = {
        'aatype': tgt_dict['aatype'],
        'residue_index': np.arange(tgt_dict['aatype'].shape[0])
    }
    predict_ret = {
        'structure_module': {
            'final_atom_mask': tgt_dict['mask'],
            'final_atom_positions': tgt_dict['pos']
        }
    }
    save_as_pdb(input_batch, predict_ret, tgt_pdb_file, rem_leading_feat_dim=False)
    
    rot = pdb_align(src_pdb_file, tgt_pdb_file, tgt_pdb_file, verbose=False)
    rot, trans = rot[:, 1:], rot[:, 0]
    pdb_string = open(tgt_pdb_file).read()
    prot = protein.from_pdb_string(pdb_string, 'A')
    dest_pos = prot.atom_positions

    os.remove(src_pdb_file)
    os.remove(tgt_pdb_file)

    return dest_pos, rot, trans

def apply_rot(position, mask, rot, trans):
    """
    Parameters
    -------------
    position: [N_res, 37, 3]
    mask: [N_res, 37]
    rot: [3, 3]
    trans: [3]
    
    Return
    -------------
    position: [N_res. 37, 3]
    """
    
    assert position.ndim == 3
    assert mask.ndim == 2
    assert rot.ndim == 2
    assert trans.ndim == 1
    assert rot.shape[0] == rot.shape[1] == 3
    assert trans.shape[0] == 3
    
    # [N_res, 37, 3] -> [N_res*37, 1, 3]
    # [N_res*37, 1, 3] * [3, 3] -> [N_res*37, 3, 3]
    # [N_res*37, 3, 3] -> [N_res*37, 3, 1]
    position = (position.reshape([-1, 1, 3]) * rot).sum(axis=2) + trans
    position = position.reshape([-1, 37, 3]) # [N_res, 37, 3]
    position[ ~mask.astype(np.bool) ] = 0.0
    return position


