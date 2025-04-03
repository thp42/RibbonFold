import numpy as np
import pandas as pd
import torch
import pickle, gzip
import os, sys, collections, subprocess, shutil
import random
from datetime import datetime
import copy
from Bio.PDB import PDBParser
import argparse

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


script_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(script_path)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "af2"))

from src.data_processing import prepare_input_pkl_file, ds_to_cuda, fill_random_vectors
from src import AlphaFold, param_loader
from src import protein_features, aux_heads
from src.chain_contact import set_contacts_for_inference


import af2_run
from alphafold.model.tf import shape_placeholders
from alphafold.common import protein as AF2_Protein_Module

DEVICE = 'cuda'



def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



def mask_msa_cols(matrix, mask_rate=0.2):
    '''
        column-wise MSA masking
    '''
    num_columns = matrix.shape[1]
    num_mask_columns = int(np.ceil(num_columns * mask_rate)) 
    start_column = np.random.randint(0, num_columns - num_mask_columns + 1)
    matrix[:, start_column:start_column + num_mask_columns] = 21

    return matrix


def fix_input_residue_index(input_data, pdbfile):
    old_residue_index = input_data['residue_index']

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("example", pdbfile)

    actual_residue_sequence = []
    for residue in structure.get_residues():
        actual_residue_sequence.append(residue.id[1] - 1)
    assert len(actual_residue_sequence) == len(old_residue_index)

    input_data['residue_index'] = np.array(actual_residue_sequence)
    return input_data



def add_num_sym_feature(raw_features):
    raw_features['num_sym'] = np.ones_like(raw_features['asym_id'])
    for entity_id in np.unique(raw_features['entity_id']):
        mask = raw_features['entity_id'] == entity_id
        raw_features['num_sym'][mask] = raw_features['sym_id'][mask].max()
    return raw_features


def get_hex_mask(n_chains, n_per_chain, chain_index_arr, temp_res_len=121, temp_chains=6):
    '''
    crop residues needed according to seq length and required num of copies, for forming an initial structure
    :return: mask   [num_hex_res]    hex structure residue length is now 726 (6 chains * 121 res)
    '''
    masks = []
    for chain in range(n_chains):
        mask = np.zeros(temp_res_len * temp_chains, dtype=bool)    # do mask on a per-chain basis to ensure the right order
        # start = temp_res_len * chain
        start = np.where(chain_index_arr == chain)[0][0]
        end = start + n_per_chain
        mask[start:end] = True
        # print(f'add mask: {start}-{end}')
        masks.append(mask)
    return masks


def process_data_for_inference(msa_features, pdb_features, use_syth_template=True, use_nes=0):

    batch_x = msa_features.copy()
    if pdb_features is not None:
        batch_y = pdb_features.copy()

    #. find batch y
    num_ensembl, num_res = batch_x['residue_index'].shape
    if use_nes:
        k = use_nes
    else:
        k = random.randint(0, num_ensembl - 1)

    if pdb_features is not None:
        batch_y['seq_mask'] = batch_x['seq_mask'][k]
        batch_y['residue_index'] = batch_x['residue_index'][k]
        batch_y['true_msa'] = batch_x['true_msa'][k]
        batch_y['bert_mask'] = batch_x['bert_mask'][k]
        # batch_y['resolution'] = np.array(pdb_data.pdb2resol.get(pdb_id, 3.5))
        batch_y['asym_id'] = batch_x['asym_id'][k]
        batch_y['sym_id'] = batch_x['sym_id'][k]
        batch_y['entity_id'] = batch_x['entity_id'][k]
        batch_y['num_sym'] = batch_x['num_sym'][k]

    for name in batch_x:
        batch_x[name] = batch_x[name][k:k + 1]

    #. syth template features
    templ_copies = 4
    def concat_multi_templ(matrix, templ_copies=4):
        matrix = np.expand_dims(matrix, axis=1)
        matrix_list = [matrix] * templ_copies
        result_matrix = np.concatenate(matrix_list, axis=1)
        return result_matrix

    if use_syth_template:
        batch_x['template_aatype'] = concat_multi_templ(batch_x['aatype'])
        batch_x['template_pseudo_beta_mask'] = concat_multi_templ(batch_x['seq_mask'])
        # batch_x['template_pair_dist'] = concat_multi_templ(find_residue_contacts(batch_y))
        pair_dist = set_contacts_for_inference(batch_x)   # (1, N, N)
        batch_x['template_pair_dist'] = concat_multi_templ(pair_dist)
        pair_orient = fill_random_vectors(pair_dist[0])   # (1, N, N, 3)
        #pair_orient = np.zeros_like(pair_orient)
        batch_x['template_pair_orient'] = concat_multi_templ(pair_orient)
        batch_x['template_mask'] = np.expand_dims(np.array([1.] * templ_copies), axis=0)

    #. final processing, make input as tensors, and check
    if use_syth_template:
        batchX = protein_features.get_batchX(batch_x, use_template=False, use_syth_template=True)
    else:
        batchX = protein_features.get_batchX(batch_x, use_template=True)
    if pdb_features is not None:
        batchY = protein_features.get_batchY(batch_y)
    else:
        batchY = None

    return batchX, batchY




def inference(batchX, alphafold_model, save_pdb_prefix, pred_t, ret_dict=None, start_recycle=0,
              num_recycle=3, early_stop=True, save_representations_data=False, LOG=sys.stdout):
    """
    Run Alphafold model with StructureModule after each Evoformer
    """

    N_ens, N_seq, N_res, emb_dim = batchX['msa_feat'].shape
    assert N_ens == 1
    alphafold_model.eval()
    assert start_recycle < num_recycle

    ##### Empty batchX
    if ret_dict is None:
        non_ensembled_batch = {
            'prev_pos': torch.zeros([N_res, 37, 3]).float().cuda(),
            'prev_msa_first_row': torch.zeros([N_res, 256]).float().cuda(),
            'prev_pair': torch.zeros([N_res, N_res, 128]).float().cuda()
        }
    else:
        non_ensembled_batch = {
            'prev_pos': ret_dict['structure_module']['final_atom_positions'],
            'prev_msa_first_row': ret_dict['representations']['msa_first_row'],
            'prev_pair': ret_dict['representations']['pair']
            # 'prev_msa_first_row': torch.zeros([N_res, 256]).float().cuda(),
            # 'prev_pair': torch.zeros([N_res, N_res, 128]).float().cuda()
        }

    highest_plddt = 0
    highest_plddt_idx = 0

    for i in range(start_recycle, num_recycle):
        print("num_recycle:", i, flush=True, file=LOG)
        with torch.no_grad():
            ret_dict = alphafold_model(batchX, non_ensembled_batch, ensemble_representations=False, recycle_id=i)

        if save_representations_data:
            data_out = {
                'pair_list': [x.cpu().numpy() for x in ret_dict['representations']['pair_list']],
                'msa_list': [x.cpu().numpy() for x in ret_dict['representations']['msa_list']],
                'msa_first_row': ret_dict['representations']['msa_first_row'].cpu().numpy(),
                'single_before_sm': ret_dict['representations']['single'].cpu().numpy(),
                'single_before_ipa': ret_dict['representations']['structure_module1'].cpu().numpy(),
                'single_final': ret_dict['representations']['structure_module'].cpu().numpy(),
            }
            out_data_file = save_pdb_prefix.replace('_infer', f'_rec{i}_data.pkl')
            pickle.dump(data_out, open(out_data_file, 'wb'))

        final_atom_positions = ret_dict['structure_module']['final_atom_positions'].cpu()
        final_atom_mask = ret_dict['structure_module']['final_atom_mask'].cpu()
        pd_file = save_pdb_prefix.replace('_infer', f'_{pred_t}_rec{i}_r.pdb')


        plddt = aux_heads.compute_plddt(ret_dict['predicted_lddt']['logits'].cpu().numpy())
        plddt_37 = np.tile(plddt[:, np.newaxis], 37)
        mean_plddt = np.mean(plddt)
        print(f"Mean of plddt: {mean_plddt:.3f}", file=LOG)
        if mean_plddt > highest_plddt:
            highest_plddt = mean_plddt
            highest_plddt_idx = i


        non_ensembled_batch = {
            'prev_pos': ret_dict['structure_module']['final_atom_positions'],
            'prev_msa_first_row': ret_dict['representations']['msa_first_row'],
            'prev_pair': ret_dict['representations']['pair'],
        }

        if i < num_recycle - 1:
            del ret_dict
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    return ret_dict




def process_and_inference(alphafold_model, test_file, savedir, pred_t, init_structure=None,
                          random_mode='cluster', ens_id=None, msa_max_cluster=8, msa_max_extra=8, run_dropout=False, recycles=5):
    data_dict = pickle.load(open(test_file, 'rb'))
    org_filename = data_dict['pdbfile']
    if org_filename is not None:
        chain_name = org_filename.split('/')[-1].split('.')[0]   # ribbon id
    else:
        chain_name = data_dict['item_id']
    raw_msa_dict = data_dict['features_x']
    if org_filename is not None:
        raw_msa_dict = fix_input_residue_index(raw_msa_dict, org_filename)
    seq_length = raw_msa_dict['seq_length'][0]
    num_chains = len(np.unique(raw_msa_dict['asym_id']))
    seq_len_per_chain = int(seq_length / num_chains)
    print('full sequence length: ', seq_length)
    print('monomer chain length: ', seq_len_per_chain)
    print('number of chains: ', num_chains)
    # print('start and end residue index: ', raw_msa_dict['residue_index'].min(), raw_msa_dict['residue_index'].max())


    if init_structure and (seq_len_per_chain < 120):
        temp_prot = af2_run.read_pdb(init_structure)
        masks = get_hex_mask(num_chains, int(seq_len_per_chain), temp_prot.chain_index)
        init_prot = None
        for mask in masks:
            cur_prot = temp_prot.filter_by_mask(mask)
            cur_prot = AF2_Protein_Module.from_protein(cur_prot)
            init_prot = cur_prot if init_prot is None else init_prot + cur_prot
        init_atom_positions = init_prot.atom_positions
        init_structure_dict = {
            'structure_module': {
                'final_atom_positions': torch.tensor(init_atom_positions).cuda()
            },
            'representations': {
                'msa_first_row': torch.zeros([seq_length, 256]).float().cuda(),
                'pair': torch.zeros([seq_length, seq_length, 128]).float().cuda(),
            }
        }
    else:
        init_structure_dict = None


    msa_max_cluster_applied = msa_max_cluster
    msa_max_extra_applied = msa_max_extra
    data_config, feature_names = protein_features.get_config(max_extra_msa=msa_max_extra_applied, crop_size=seq_length,
                                                             max_msa_clusters=msa_max_cluster_applied, num_ensemble=1, num_recycle=1,
                                                             use_templates=False)
    feature_names += ['asym_id', 'sym_id', 'entity_id', 'num_sym', 'crop_mask']
    data_config.eval.feat.asym_id   = [shape_placeholders.NUM_RES]
    data_config.eval.feat.sym_id    = [shape_placeholders.NUM_RES]
    data_config.eval.feat.entity_id = [shape_placeholders.NUM_RES]
    data_config.eval.feat.num_sym   = [shape_placeholders.NUM_RES]
    data_config.eval.feat.crop_mask = [shape_placeholders.NUM_RES]
    msa_ft = protein_features.process_np_example(raw_msa_dict, data_config, feature_names)
    msa_ft = add_num_sym_feature(msa_ft)
    batch_x = msa_ft.copy()
    processed_msa = copy.deepcopy(msa_ft)


    # select one cluster
    if random_mode == 'one_cluster':
        if ens_id is None:
            ens_id = random.randint(0, processed_msa['msa_row_mask'].sum() - 1)
        ck = ens_id
        batch_x['msa_feat'][0] = msa_ft['msa_feat'][0, ck:ck + 1]
        batch_x['msa_mask'][0] = msa_ft['msa_mask'][0, ck:ck + 1]
        batch_x['msa_row_mask'][0] = msa_ft['msa_row_mask'][0, ck:ck + 1]
        batch_x['bert_mask'][0] = msa_ft['bert_mask'][0, ck:ck + 1]
        batch_x['true_msa'][0] = msa_ft['true_msa'][0, ck:ck + 1]
        # fill extra features (non ensembled)
        batch_x['extra_msa'][0] = msa_ft['extra_msa'][0]
        batch_x['extra_msa_mask'][0] = msa_ft['extra_msa_mask'][0]
        batch_x['extra_msa_row_mask'][0] = msa_ft['extra_msa_row_mask'][0]
        batch_x['extra_has_deletion'][0] = msa_ft['extra_has_deletion'][0]
        batch_x['extra_deletion_value'][0] = msa_ft['extra_deletion_value'][0]


    ############## process input features ###########
    pdb_ft = data_dict['features_y']
    batchX, batchY = process_data_for_inference(batch_x, pdb_ft, use_nes=None)
    ds_to_cuda(batchX)
    if batchY is not None:
        ds_to_cuda(batchY)


    ############# inference ##############
    if run_dropout:
        alphafold_model = alphafold_model.train()
    save_prefix = os.path.join(savedir, f'{chain_name}_infer')
    ret = inference(batchX, alphafold_model, save_prefix, pred_t, ret_dict=init_structure_dict, start_recycle=0,
        num_recycle=recycles, early_stop=False)
    print('run inference finished')

    final_atom_positions = ret['structure_module']['final_atom_positions'].cpu()
    final_atom_mask = ret['structure_module']['final_atom_mask'].cpu()
    gt_file = os.path.join(savedir, f'{chain_name}_gt.pdb')
    pd_file = os.path.join(savedir, f'{chain_name}_pd_{pred_t}.pdb')

    # Save Prediction
    plddt = aux_heads.compute_plddt(ret['predicted_lddt']['logits'].cpu().numpy())
    plddt_37 = np.tile(plddt[:, np.newaxis], 37)
    af2_run.save_as_pdb(batchX['aatype'].cpu().numpy(), batchX['residue_index'].cpu().numpy(),
                        final_atom_positions.numpy(), final_atom_mask.numpy(), pd_file,
                        b_factors=plddt_37, asym_id=batchX['asym_id'].cpu().numpy()-1)
    # Save Ground Truth
    if batchY is not None:
        af2_run.save_as_pdb(batchY['aatype'].cpu().numpy(), batchY['residue_index'].cpu().numpy(),
                                 batchY['all_atom_positions'].cpu().numpy(),
                                 batchY['all_atom_mask'].cpu().numpy(), gt_file, asym_id=batchX['asym_id'].cpu().numpy()-1)




def parse_args():
    parser = argparse.ArgumentParser(description="Process command-line arguments.")

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--input_pkl', type=str, required=True, help='Input ribbon MSA pickle file')
    parser.add_argument('--ribbon_name', type=str, default='test_ribbon', help='Input ribbon name')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--rounds', type=int, default=10, help='Number of samples (default: 10)')
    parser.add_argument('--recycles', type=int, default=5, help='Number of recycles per round (default: 5)')
    parser.add_argument('--msa_random_mode', type=str, default='one_cluster',
                        help='MSA selecting method, one of: one_cluster or multi_cluster')
    parser.add_argument('--use_dropout', type=bool, default=True, help='Whether to use dropout when sampling (default: True)')
    parser.add_argument('--use_init_structure', type=bool, default=True, help='Whether to use initial structure (default: True)')

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()
    print("Checkpoint:", args.checkpoint)
    print("Input Ribbon data:", args.input_pkl)
    print("Output Directory:", args.output_dir)
    print("Rounds:", args.rounds)
    print("Recycles:", args.recycles)
    print("MSA Random Sampling Mode:", args.msa_random_mode)
    print("Use Dropout:", args.use_dropout)
    print("Use Initial Structure:", args.use_init_structure)


    # load model
    alphafold_model = AlphaFold.AlphaFoldIteration(enable_template=True,
                                                   use_predicted_aligned_error_head=False,
                                                   use_experimentally_resolved_head=False).cuda().eval()
    ckpt_filename = args.checkpoint
    state_dict = torch.load(ckpt_filename, map_location='cuda:0')
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict['model'].items():
        name = k[7:]    
        new_state_dict[name] = v
    alphafold_model.load_state_dict(new_state_dict, strict=False)
    print('load model checkpoint success: ', ckpt_filename)


    # out directory
    ckpt_name = ckpt_filename.split('/')[-1].split('.')[0]
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    savedir = os.path.join(args.output_dir, ckpt_name + '_' + timestamp)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    # settings
    init_struc_file = os.path.join('./data', 'hexamer_para.pdb') if args.use_init_structure else None
    run_dropout = args.use_dropout

    # preprocess input MSA features

    raw_msa_file = args.input_pkl
    single_msa_data = pickle.load(gzip.open(raw_msa_file))
    processed_feature_file = raw_msa_file.replace('.pkl.gz', '_processed.pkl')
    prepare_input_pkl_file(single_msa_data, args.ribbon_name, processed_feature_file, num_chains=5)


    # run inference
    for t in range(args.rounds):
        print(f'run sample {t}...')
        process_and_inference(alphafold_model, processed_feature_file, savedir, t, init_structure=init_struc_file,
                              random_mode=args.msa_random_mode, ens_id=None, msa_max_cluster=64, msa_max_extra=128,
                              run_dropout=run_dropout, recycles=args.recycles)








