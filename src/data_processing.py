import os, sys, pickle, time, random, gzip, io, collections, copy, json
import numpy as np
import torch



def ds_to_cuda(batch_data):
    for k in batch_data:
        if isinstance(batch_data[k], torch.Tensor):
            batch_data[k] = batch_data[k].cuda()


def ds_to_cpu(batch_data):
    for k in batch_data:
        if isinstance(batch_data[k], torch.Tensor):
            batch_data[k] = batch_data[k].cpu()


def ds_fp_to_dtype(batch_data, dtype=torch.float32):
    """
    Convert the floating type to dtype
    """
    for k in batch_data:
        if torch.is_floating_point(batch_data[k]):
            batch_data[k] = batch_data[k].to(dtype)


AMINO_ACIDS = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X'
]


CLASS_NAMES = {
    1: "ALA", 2: "ARG", 3: "ASN", 4: "ASP", 5: "CYS",
    6: "GLN", 7: "GLU", 8: "GLY", 9: "HIS", 10: "ILE",
    11: "LEU", 12: "LYS", 13: "MET", 14: "PHE", 15: "PRO",
    16: "SER", 17: "THR", 18: "TRP", 19: "TYR", 20: "VAL",
    0: "nul", 21: 'UNK'
}

NAME2ABBV = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "nul": "0", 'UNK': "X"
}

ID_TO_HHBLITS_AA = {
    0: 'A',
    1: 'C',  # Also U.
    2: 'D',  # Also B.
    3: 'E',  # Also Z.
    4: 'F',
    5: 'G',
    6: 'H',
    7: 'I',
    8: 'K',
    9: 'L',
    10: 'M',
    11: 'N',
    12: 'P',
    13: 'Q',
    14: 'R',
    15: 'S',
    16: 'T',
    17: 'V',
    18: 'W',
    19: 'Y',
    20: 'X',  # Includes J and O.
    21: '-',
}

def vector2seq(arr):
    '''for msa raw features'''
    return "".join(ID_TO_HHBLITS_AA.get(x) for x in arr)

def vector2seq_aatype(arr):
    '''for other seq features'''
    return ''.join([NAME2ABBV[CLASS_NAMES[x+1]] for x in arr])


def generate_random_unit_vector():
    vec = np.random.randn(3)
    return vec / np.linalg.norm(vec)


def fill_random_vectors(distance_matrix):
    # numpy version
    N_res = distance_matrix.shape[0]
    result_matrix = np.zeros((N_res, N_res, 3))
    non_zero_indices = np.argwhere(distance_matrix != 0)
    vec3 = generate_random_unit_vector()
    lower_triangle_indices = non_zero_indices[non_zero_indices[:, 0] < non_zero_indices[:, 1]]
    result_matrix[lower_triangle_indices[:, 0], lower_triangle_indices[:, 1]] = vec3
    upper_triangle_indices = non_zero_indices[non_zero_indices[:, 0] > non_zero_indices[:, 1]]
    result_matrix[upper_triangle_indices[:, 0], upper_triangle_indices[:, 1]] = -vec3
    result_matrix = np.expand_dims(result_matrix, axis=0)
    return result_matrix



def save_as_pdb(input_batch, predict_ret, out_file, b_factors=None, rem_leading_feat_dim=True):
    """
    input_batch: dict
        -- asym_id: [N_res] Optional
        -- aatype: [N_res]
        -- residue_index: [N_res]

    predict_ret: dict
        -- structure_module: dict
            -- final_atom_mask: [N_res, 37]
            -- final_atom_positions: [N_res, 37, 3]

    out_file: str
    b_factors: [N_res, 37]
    """
    from alphafold.common import protein
    from src import utils, aux_heads, residue_constants
    from src.common_modules import t2n

    if 'structure_module' not in predict_ret:
        predict_ret = {'structure_module': predict_ret}

    input_batch = utils.tree_map(lambda x: t2n(x), input_batch)
    predict_ret = utils.tree_map(lambda x: t2n(x), predict_ret)

    if b_factors is None and 'predicted_lddt' in predict_ret:
        plddt = aux_heads.compute_plddt(predict_ret['predicted_lddt']['logits'])
        b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)

    unrelaxed_protein = protein.from_prediction(input_batch,
                                                predict_ret,
                                                b_factors=b_factors,
                                                remove_leading_feature_dimension=rem_leading_feat_dim)
    print(protein.to_pdb(unrelaxed_protein), file=open(out_file, 'w'))


def repeat_concat(array, axis, n_times):
    expanded_array = np.repeat(array, n_times, axis=axis)
    return expanded_array


def repeat_along_dim(x, dim, n_times):
    expanded_array = np.expand_dims(x, axis=dim)
    result = np.repeat(expanded_array, n_times, axis=dim)
    new_shape = list(x.shape)
    new_shape[dim] *= n_times
    result = result.reshape(new_shape)
    return result


def concat_msa_multi(msa_dict, n_times):
    concat_dict = {}
    dims = {'aatype': 0,
            'between_segment_residues': 0,
            'residue_index': 0,
            'seq_length': 0,
            'deletion_matrix_int': 1,
            'msa': 1,
            'num_alignments': 0,
            'template_aatype': 1,
            'template_all_atom_masks': 1,
            'template_all_atom_positions': 1,
            'template_confidence_scores': 1
            }
    for k,v in msa_dict.items():
        if k not in dims:
            pass
        else:
            axis = dims[k]
            v = repeat_along_dim(v, axis, n_times)
        concat_dict[k] = v
    asym_id = []
    sym_id = []
    entity_id = []
    seq_len = msa_dict['aatype'].shape[0]
    # single_seq_len = seq_len // n_times
    single_seq_len = seq_len
    for i in range(n_times):
        asym_id += [i+1] * single_seq_len
        sym_id += [i+1] * single_seq_len
        entity_id += [1] * single_seq_len

    concat_dict['asym_id'] = np.array(asym_id)
    concat_dict['sym_id'] = np.array(sym_id)
    concat_dict['entity_id'] = np.array(entity_id)
    concat_dict['seq_length'] = concat_dict['seq_length'] * n_times
    return concat_dict


def get_seq_from_aatype(aatype):
    amino_acid_sequence = ''.join(AMINO_ACIDS[i] for i in aatype)
    return amino_acid_sequence


def process_single_item(single_msa, copies=5):
    msa_dict = single_msa
    seq = get_seq_from_aatype(msa_dict['aatype'].argmax(axis=1))
    msa_features = concat_msa_multi(msa_dict, copies)
    return msa_features, seq


def prepare_input_pkl_file(single_msa, ribbon_name, outfile, num_chains=5):

    msa_features, single_seq = process_single_item(single_msa, copies=num_chains)

    out = {'pdbfile': None, 'seq': single_seq, 'n_chains': num_chains, 'features_x': msa_features,
           'features_y': None, 'chain_splits': None, 'item_id': ribbon_name}

    # outfile = os.path.join(out_dir, f"{ribbon_name}.pkl")
    pickle.dump(out, open(outfile, mode='wb'))


if __name__ == '__main__':
    input_msa_file = './5oqv_bfd_single_msa.pkl.gz'
    single_msa_data = pickle.load(gzip.open(input_msa_file))
    prepare_input_pkl_file(single_msa_data, '5oqv', '../examples')
