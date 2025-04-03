import numpy as np
from collections import Counter
import random
import pickle


def most_frequent_element(residue_index):
    filtered_index = residue_index[residue_index != 0]
    count = Counter(filtered_index)
    if count:
        most_common_element = count.most_common(1)[0][0]
    else:
        most_common_element = None 
    return most_common_element


def most_frequent_non_zero_random(residue_index):
    # Filter out zeros
    non_zero_residues = residue_index[residue_index != 0]
    # Count occurrences of each element
    counter = Counter(non_zero_residues)
    # Find the maximum occurrence count
    max_count = max(counter.values())
    # Find all elements that have the maximum occurrence count
    most_frequent_elements = [elem for elem, count in counter.items() if count == max_count]
    # Randomly choose one of the most frequent elements
    return random.choice(most_frequent_elements)


def calculate_vector_differences(i, atom_positions, asym_id, residue_index, seq_mask):
    valid_positions = np.where(seq_mask == 1)[0]
    valid_atom_positions = atom_positions[valid_positions]
    valid_asym_id = asym_id[valid_positions]
    valid_residue_index = residue_index[valid_positions]
    _, c_idx = np.unique(valid_asym_id, return_index=True)
    chain_ids = valid_asym_id[np.sort(c_idx)]

    ca_positions = []
    for chain_id in chain_ids:
        chain_mask = valid_asym_id == chain_id
        chain_residues = valid_residue_index[chain_mask]
        if i in chain_residues:
            residue_mask = chain_residues == i
            ca_position = valid_atom_positions[chain_mask][residue_mask]
            ca_positions.append(ca_position[0])  
    if len(ca_positions) < len(chain_ids):
        print(f'residue {i} not found on all chains')
        print(ca_positions)
        return None
    ca_positions = np.array(ca_positions)
    num_chains = len(ca_positions)
    vector_diff_matrix = np.zeros((num_chains, num_chains, 3))
    for m in range(num_chains):
        for n in range(num_chains):
            if m != n:
                vector_diff_matrix[m, n] = ca_positions[m] - ca_positions[n]
    return chain_ids, vector_diff_matrix


def calculate_chain_contacts(atom_positions, asym_id, residue_index, seq_mask, use_random=True):
    if use_random:
        i = most_frequent_non_zero_random(residue_index)
    else:
        i = most_frequent_element(residue_index)
    chain_ids, vector_diff_matrix = calculate_vector_differences(i, atom_positions, asym_id, residue_index, seq_mask)
    contact_matrix = np.linalg.norm(vector_diff_matrix, axis=2)     # [n_chains, n_chains]
    return chain_ids, contact_matrix


def calculate_residue_distance_matrix(asym_id, residue_index, seq_mask, contact, chain_ids):
    num_residues = len(asym_id)
    distance_matrix = np.zeros((num_residues, num_residues))
    valid_positions = np.where(seq_mask == 1)[0]
    valid_asym_id = asym_id[valid_positions]
    valid_residue_index = residue_index[valid_positions]
    chain_id_to_index = {chain_id: idx for idx, chain_id in enumerate(chain_ids)}
    for i in range(len(valid_positions)):
        for j in range(len(valid_positions)):
            if valid_residue_index[i] == valid_residue_index[j]:
                chain_i = valid_asym_id[i]
                chain_j = valid_asym_id[j]
                if chain_i in chain_id_to_index and chain_j in chain_id_to_index:
                    index_i = chain_id_to_index[chain_i]
                    index_j = chain_id_to_index[chain_j]
                    distance_matrix[valid_positions[i], valid_positions[j]] = contact[index_i, index_j]
    return distance_matrix


def find_residue_contacts(batchy, pdb_id=None):
    atom_positions = batchy['all_atom_positions'][:,1,:]
    asym_id = batchy['asym_id']
    residue_index = batchy['residue_index']
    seq_mask = batchy['seq_mask']

    chain_ids, chain_contacts = calculate_chain_contacts(atom_positions, asym_id, residue_index, seq_mask)
    distance_matrix = calculate_residue_distance_matrix(asym_id, residue_index, seq_mask, chain_contacts, chain_ids)
    distance_matrix = np.expand_dims(distance_matrix, axis=0)
    return distance_matrix


def apply_dropout(seq_mask, dropout=0.25):
    ones_indices = np.where(seq_mask == 1)[0]
    num_dropout = int(len(ones_indices) * dropout)
    dropout_indices = np.random.choice(ones_indices, size=num_dropout, replace=False)
    seq_mask[dropout_indices] = 0

    return seq_mask



def set_contacts_for_inference(batchx, dropout=0.25):
    asym_id = batchx['asym_id'][0]
    residue_index = batchx['residue_index'][0]
    seq_mask = batchx['seq_mask'][0]
    assert len(seq_mask.shape) == 1

    valid_positions = np.where(seq_mask == 1)[0]
    valid_asym_id = asym_id[valid_positions]
    chain_ids = np.unique(valid_asym_id)

    contact_matrix = np.zeros((len(chain_ids), len(chain_ids)))
    for i in range(len(chain_ids)):
        for j in range(len(chain_ids)):
            contact_matrix[i][j] = 4.85 * abs(i - j)

    distance_matrix = calculate_residue_distance_matrix(asym_id, residue_index, seq_mask, contact_matrix, chain_ids)
    distance_matrix = np.expand_dims(distance_matrix, axis=0)

    return distance_matrix

