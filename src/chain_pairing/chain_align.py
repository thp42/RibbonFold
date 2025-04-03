import torch
from typing import Optional, Union, List
import os, sys, time, re, random, pickle, copy, gzip, io, configparser, math, shutil, pathlib, tempfile, hashlib, argparse, json, inspect, urllib, collections, subprocess, requests, platform, multiprocessing
from typing import Dict, Union, Optional
import multiprocessing as mp
import numpy as np
import pandas as pd

from .geometry import kabsch_rmsd, get_optimal_transform, compute_rmsd

from alphafold.common import residue_constants
from alphafold.common import protein as AF2_Protein_Module

def pairing_and_get_reordered_batchY(ret_dict, batchX, batchY):
    """
    Pairing prediction with groud truth and reorder the symmetric chains in batchY
    """
    import torch.utils._pytree as pytree
    
    assert 'gt_prot' in batchY
    gt_prot = batchY['gt_prot']
    
    asym_id_arr = batchX['asym_id'][0].cpu().numpy()
    #asym_id_arr = batchY['asym_id'][0].cpu().numpy()
    
    batchX0 = pytree.tree_map(lambda x: x[0], batchX)
    best_align, best_labels = multi_chain_perm_align(ret_dict['structure_module'], batchX0, gt_prot)
    best_align.sort(key=lambda x: x[0])

    ### Check
    pd_asym_array1 = np.array([ x[0] for x in best_align ])
    pd_asym_array2 = np.arange(asym_id_arr.max())
    assert np.all(pd_asym_array1 == pd_asym_array2), f"Expect same array, but got {pd_asym_array1} and {pd_asym_array2}"

    pd2gt = { int(x[0]):int(x[1]) for x in best_align }

    uniq_asym_id, uniq_asym_id_index = np.unique(asym_id_arr, return_index=True)
    argsort = np.argsort(uniq_asym_id_index)
    uniq_asym_id = uniq_asym_id[argsort].astype(np.int)

    index = []
    for asym_id in uniq_asym_id:
        if asym_id == 0:
            index.append( np.where(asym_id == asym_id_arr)[0] )
        else:
            gt_asym_id = pd2gt[asym_id-1] + 1
            index.append( np.where(gt_asym_id == gt_prot.chain_index)[0] )
    index = np.concatenate(index, 0)
    index = torch.from_numpy(index).long().to(batchY['aatype'].device)

    batchY_tmp = {}
    for k,v in batchY.items():
        if not isinstance(v, torch.Tensor):
            batchY_tmp[k] = v
            continue
        if k in ('resolution', 'support_positions'):
            batchY_tmp[k] = v
        elif k in ('true_msa', 'bert_mask'):
            batchY_tmp[k] = torch.index_select(v, 1, index)
        else:
            batchY_tmp[k] = torch.index_select(v, 0, index)
    
    return batchY_tmp

def multi_chain_perm_align(ret_dict: dict, batch: dict, gt_prot: AF2_Protein_Module.Protein, shuffle_times:int = 2):
    """
    Parameters
    --------------
    ret_dict: from AlphaFold Module
    batch: batchX dictionary
    gt_prot: Cropped prot object (unresolved intermediate residues must be contained)

    Return
    --------------
    best_labels: 
    """
    assert isinstance(gt_prot, AF2_Protein_Module.Protein)
    assert batch['aatype'].ndim == 1, f"Expect batch['aatype'].ndim == 1, but got batch['aatype'].ndim={batch['aatype'].ndim}"

    seq_mask = batch['seq_mask'].bool()
    labels = []
    ### Check gt_prot 
    unique_asym_ids = torch.unique(batch["asym_id"]).cpu().numpy()
    for cur_asym_id in unique_asym_ids:
        if cur_asym_id == 0:
            continue
        asym_mask = (batch["asym_id"] == cur_asym_id).bool()
        aatype = batch['aatype'][asym_mask].cpu().numpy()
        num_res = asym_mask.float().sum()
        prot = gt_prot.filter_by_mask( gt_prot.chain_index == cur_asym_id )
        assert num_res == prot.aatype.shape[0], f"Expect same length, but got {num_res} and {prot.aatype.shape[0]}"
        assert np.all(aatype == prot.aatype), f"Expect same aatype, but got {aatype} and {prot.aatype}"
        labels.append({
            'all_atom_positions': torch.from_numpy(prot.atom_positions).to(batch["asym_id"].device).float(),
            'all_atom_mask': torch.from_numpy(prot.atom_mask).to(batch["asym_id"].device).float()
        })

    ca_idx = residue_constants.atom_order["CA"]
    pred_ca_pos = ret_dict["final_atom_positions"][..., ca_idx, :].float()  # [bsz, nres, 3]
    pred_ca_mask = ret_dict["final_atom_mask"][..., ca_idx].float()  # [bsz, nres]
    true_ca_poses = [
        l["all_atom_positions"][..., ca_idx, :].float() for l in labels
    ]  # list([nres, 3])
    true_ca_masks = [
        l["all_atom_mask"][..., ca_idx].float() for l in labels
    ]  # list([nres,])

    #per_asym_residue_index = {}
    #for cur_asym_id in unique_asym_ids:
    #    asym_mask = (batch["asym_id"] == cur_asym_id).bool()
    #    per_asym_residue_index[int(cur_asym_id)] = batch["residue_index"][asym_mask]

    anchor_gt_asym, anchor_pred_asym = get_anchor_candidates(batch, true_ca_masks)
    # anchor_gt_asym: The asym_id choosed
    # anchor_pred_asym: The asym_id list with same entity_id with choosed asym_id

    anchor_gt_idx = int(anchor_gt_asym) - 1

    best_rmsd = 1e9
    best_labels = None
    best_align = None

    unique_entity_ids = torch.unique(batch["entity_id"])
    entity_2_asym_list = {} # { entity_id -> [asym_id1, asym_id2, ...] }
    for cur_ent_id in unique_entity_ids:
        ent_mask = batch["entity_id"] == cur_ent_id
        cur_asym_id = torch.unique(batch["asym_id"][ent_mask])
        entity_2_asym_list[int(cur_ent_id)] = cur_asym_id

    for cur_asym_id in anchor_pred_asym:
        asym_mask = (batch["asym_id"] == cur_asym_id).bool()
        # anchor_residue_idx = per_asym_residue_index[int(cur_asym_id)]
        anchor_true_pos = true_ca_poses[anchor_gt_idx] #[anchor_residue_idx]
        anchor_pred_pos = pred_ca_pos[asym_mask]
        anchor_true_mask = true_ca_masks[anchor_gt_idx] #[anchor_residue_idx]
        anchor_pred_mask = pred_ca_mask[asym_mask]

        r, x = get_optimal_transform(
            anchor_true_pos,
            anchor_pred_pos,
            (anchor_true_mask * anchor_pred_mask).bool(),
        )

        aligned_true_ca_poses = [ca @ r + x for ca in true_ca_poses]  # apply transforms
        for _ in range(shuffle_times):
            shuffle_idx = np.random.permutation(unique_asym_ids.shape[0])
            shuffled_asym_ids = unique_asym_ids[shuffle_idx]
            align = greedy_align(
                batch,
                # per_asym_residue_index, # { asym_id -> residue_index }
                shuffled_asym_ids, #      permutation of all asym_id
                entity_2_asym_list, #     { entity_id -> [asym_id1, asym_id2, ...] }
                pred_ca_pos,
                pred_ca_mask,
                aligned_true_ca_poses,
                true_ca_masks,
            )

            merged_labels = merge_labels(
                batch,
                # per_asym_residue_index,
                labels,
                align,
            )

            rmsd = kabsch_rmsd(
                merged_labels["all_atom_positions"][..., ca_idx, :] @ r + x,
                pred_ca_pos,
                (pred_ca_mask * merged_labels["all_atom_mask"][..., ca_idx]).bool(),
            )

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_labels = merged_labels
                best_align = align
    
    return best_align, best_labels


def get_anchor_candidates(batch, true_masks):
    """
    Get the best GT asym_id and best Pred asym_id

    batch: BatchX
    per_asym_residue_index: { asym_id: residue_index, ... }
    """
    def find_by_num_sym(min_num_sym):
        """
        Choose the longest chain with num_sym == min_num_sym
        """
        best_len = -1
        best_gt_asym = None
        asym_ids = torch.unique(batch["asym_id"][batch["num_sym"] == min_num_sym])
        for cur_asym_id in asym_ids:
            assert cur_asym_id > 0
            #cur_residue_index = per_asym_residue_index[int(cur_asym_id)] # residue_index
            j = int(cur_asym_id - 1)
            cur_true_mask = true_masks[j]# [cur_residue_index] # Mask
            cur_len = cur_true_mask.sum()
            if cur_len > best_len:
                best_len = cur_len
                best_gt_asym = cur_asym_id
        return best_gt_asym, best_len

    sorted_num_sym = batch["num_sym"][batch["num_sym"] > 0].sort()[0] # num_sym types
    best_gt_asym = None
    best_len = -1
    for cur_num_sym in sorted_num_sym:
        if cur_num_sym <= 0:
            continue
        cur_gt_sym, cur_len = find_by_num_sym(cur_num_sym)
        if cur_len > best_len:
            best_len = cur_len
            best_gt_asym = cur_gt_sym
        if best_len >= 3: # 3 points is enough to be used as anchor
            break
    best_entity = batch["entity_id"][batch["asym_id"] == best_gt_asym][0]
    best_pred_asym = torch.unique(batch["asym_id"][batch["entity_id"] == best_entity])
    # best_gt_asym: The asym_id choosed
    # best_pred_asym: The asym_id list with same entity_id with choosed asym_id
    return best_gt_asym, best_pred_asym


def greedy_align(
    batch,
    # per_asym_residue_index,
    unique_asym_ids,
    entity_2_asym_list,
    pred_ca_pos,
    pred_ca_mask,
    true_ca_poses,
    true_ca_masks,
    ):
    """
    batch: batchX
    per_asym_residue_index: { asym_id -> residue_index }
    unique_asym_ids: Permutation of all asym_id
    entity_2_asym_list: { entity_id -> [asym_id1, asym_id2, ...] }
    pred_ca_pos:
    pred_ca_mask:
    true_ca_poses:
    true_ca_masks: 
    """
    used = [False for _ in range(len(true_ca_poses))]
    align = []
    for cur_asym_id in unique_asym_ids:
        # skip padding
        if cur_asym_id == 0:
            continue
        i = int(cur_asym_id - 1)
        asym_mask = batch["asym_id"] == cur_asym_id
        num_sym = batch["num_sym"][asym_mask][0]
        # don't need to align
        if (num_sym) == 1:
            align.append((i, i))
            assert used[i] == False
            used[i] = True
            continue
        cur_entity_ids = batch["entity_id"][asym_mask][0]
        best_rmsd = 1e10
        best_idx = None
        cur_asym_list = entity_2_asym_list[int(cur_entity_ids)]
        # cur_residue_index = per_asym_residue_index[int(cur_asym_id)]
        cur_pred_pos = pred_ca_pos[asym_mask]
        cur_pred_mask = pred_ca_mask[asym_mask]
        for next_asym_id in cur_asym_list:
            if next_asym_id == 0:
                continue
            j = int(next_asym_id - 1)
            if not used[j]:  # posesible candidate
                cropped_pos = true_ca_poses[j]#[cur_residue_index]
                mask = true_ca_masks[j]#[cur_residue_index]
                rmsd = compute_rmsd(
                    cropped_pos, cur_pred_pos, (cur_pred_mask * mask).bool()
                )
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_idx = j

        assert best_idx is not None
        used[best_idx] = True
        align.append((i, best_idx))

    return align


def merge_labels(batch, labels, align):
    """
    batch:
    labels: list of label dicts, each with shape [nk, *]
    align: list of int, such as [2, None, 0, 1], each entry specify the corresponding label of the asym.
    """
    num_res = batch["msa_mask"].shape[-1]
    outs = {}
    for k, v in labels[0].items():
        if k in [
            "resolution",
        ]:
            continue
        cur_out = {}
        for i, j in align:
            label = labels[j][k]
            # to 1-based
            # cur_residue_index = per_asym_residue_index[i + 1]
            cur_out[i] = label# [cur_residue_index]
        cur_out = [x[1] for x in sorted(cur_out.items())]
        new_v = torch.concat(cur_out, dim=0)
        merged_nres = new_v.shape[0]
        assert (
            merged_nres <= num_res
        ), f"bad merged num res: {merged_nres} > {num_res}. something is wrong."
        if merged_nres < num_res:  # must pad
            pad_dim = new_v.shape[1:]
            pad_v = new_v.new_zeros((num_res - merged_nres, *pad_dim))
            new_v = torch.concat((new_v, pad_v), dim=0)
        outs[k] = new_v
    return outs
