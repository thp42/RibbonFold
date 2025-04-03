# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, time, re, random, pickle, copy, gzip, io, configparser, math, shutil, pathlib, collections
import numpy as np


if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'bool'):
    np.bool = np.bool_


from typing import Optional, Union, List

os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '4.0'

cur_path = str(pathlib.Path(__file__).parent.resolve())


from alphafold.common import protein, residue_constants
from alphafold.common import residue_constants as rc
from alphafold.common import protein as AF2_Prot_Module
from alphafold.common.protein import Protein as AF2_Protein


restype_1to3 = residue_constants.restype_1to3
restype_3to1 = residue_constants.restype_3to1
restypes_with_x = residue_constants.restypes_with_x
restype_order_with_x = residue_constants.restype_order_with_x
atom_order = residue_constants.atom_order
PDB_CHAIN_IDS = protein.PDB_CHAIN_IDS


join = os.path.join
HOME = os.environ.get('HOME', 'HOME')
if 'PDB_DATA_DIR' in os.environ:
    PDB_DATA_DIR = os.environ['PDB_DATA_DIR']
else:
    PDB_DATA_DIR = join(HOME, ".cache", "pdb")

PhmmerDomain = collections.namedtuple('PhmmerDomain',
                                      ['q_start', 'q_seq', 'q_end', 'match', 'domain_score', 't_id', 't_start', 't_seq',
                                       't_end'])



######################
## Util functions
######################


def aatype2seq(aatype_or_protein: Union[np.ndarray, AF2_Protein],
               molecular_type: str = 'protein',
               from_hhblits: bool = False):
    """
    Convert aatype to Protein/DNA/RNA sequence

    Parameters
    ---------------
    aatype_or_protein: aatype array or AF2_Protein object
    molecular_type: protein, DNA or RNA
    from_hhblits: is template aatype. Not compatable with DNA/RNA or input protein object

    Return
    ---------------
    seq_or_seq_dict: sequence string (aatype_or_protein is a ndarray) or sequence dict (aatype_or_protein is a Protein)
    """

    def _aatype2seq_(aatype):
        if molecular_type == 'protein':
            if from_hhblits:
                restypes_with_x = ['?'] * 22
                for k, v in rc.HHBLITS_AA_TO_ID.items():
                    if k in rc.restypes_with_x_and_gap:
                        restypes_with_x[v] = k
            else:
                restypes_with_x = rc.restypes_with_x_and_gap
        else:
            restypes_with_x = rc.nuctypes_with_x

        if aatype.ndim == 1:
            seq = "".join([restypes_with_x[d] for d in aatype])
        elif aatype.ndim == 2:
            seq = "".join([restypes_with_x[d] for d in aatype.argmax(1)])
        else:
            raise RuntimeError(f"aatype.ndim = {aatype.ndim}")
        return seq

    assert isinstance(aatype_or_protein, (np.ndarray,
                                          AF2_Protein)), f"Expect aatype_or_protein be one of np.ndarray or AF2_Protein, but got {type(aatype_or_protein)}"
    assert molecular_type in (
    'protein', 'DNA', 'RNA'), f"Expect molecular_type be one of protein, DNA, RNA. Nut got {molecular_type}"
    if from_hhblits:
        assert isinstance(aatype_or_protein,
                          np.ndarray), f"aatype_or_protein must be np.ndarray if from_hhblits is True"
        assert molecular_type == 'protein', f"molecular_type must be protein if from_hhblits is True"
    if isinstance(aatype_or_protein, AF2_Protein):
        assert aatype_or_protein.molecular_type == molecular_type, f"Protein object has molecular_type of {molecular_type}, but provide molecular_type of {molecular_type}"

    if isinstance(aatype_or_protein, AF2_Protein):
        chain2seq = {}
        for ch_idx in np.unique(aatype_or_protein.chain_index):
            ch_name = AF2_Prot_Module.PDB_CHAIN_IDS[ch_idx]
            chain2seq[ch_name] = _aatype2seq_(aatype_or_protein[ch_name].aatype)
        return chain2seq
    else:
        return _aatype2seq_(aatype_or_protein)


def seq2aatype(seq: str, molecular_type: str = 'protein', to_hhblits: bool = False, one_hot_encoding: bool = False):
    """
    Convert sequence to aatype object

    Parameters
    ---------------
    seq: sequence
    molecular_type: protein, DNA or RNA
    to_hhblits: is template aatype. Not compatable with DNA/RNA
    one_hot_encoding: output one-hot encoded array

    Return
    ---------------
    aatype: nd.array
    """
    assert isinstance(seq, str), f"Expect seq be str type, but got {type(seq)}"
    assert molecular_type in (
    'protein', 'DNA', 'RNA'), f"Expect molecular_type be one of protein, DNA, RNA. Nut got {molecular_type}"
    if to_hhblits:
        assert molecular_type == 'protein', f"molecular_type must be protein if to_hhblits is True"

    if molecular_type == 'protein':
        if to_hhblits:
            restype_order_with_x = rc.HHBLITS_AA_TO_ID
        else:
            restype_order_with_x = {a: i for i, a in enumerate(rc.restypes_with_x_and_gap)}
    else:
        restype_order_with_x = rc.nuctype_order_with_x

    aatype = np.array([restype_order_with_x.get(r.upper(), 'X') for r in seq], dtype=np.int64)

    if one_hot_encoding:
        num_items = np.max(list(restype_order_with_x.values()))
        aatype = np.eye(num_items + 1)[aatype]

    return aatype


def write_pdb_seq(sequence: str, chain_id: str, molecular_type='protein'):
    """
    Write Sequence to PDB SEQRES field

    Parameters
    -----------
    sequence: str
    chain_id: str
    molecular_type: protein, RNA, DNA

    Return
    -----------
    seqres_lines: list
        Lines to save in PDB file. e.g. SEQRES   1 A   21  GLY ILE VAL GLU GLN CYS CYS THR SER ILE CYS SER LEU
    """
    assert molecular_type in ('protein', 'RNA', 'DNA')

    seq_len = len(sequence)
    assert seq_len <= 9999
    assert len(chain_id) == 1
    seqres_lines = []
    for row_idx, start_idx in enumerate(range(0, seq_len, 13)):
        seq_1 = sequence[start_idx:start_idx + 13]
        if molecular_type == 'protein':
            seq_3 = [restype_1to3.get(res, 'UNK') for res in seq_1]
        elif molecular_type == 'RNA':
            seq_3 = ['  ' + res for res in seq_1]
        else:
            seq_3 = [' D' + res for res in seq_1]
        line = f"SEQRES {row_idx + 1:3d} {chain_id} {seq_len:4d}  {' '.join(seq_3)}"
        seqres_lines.append(line)
    return seqres_lines


def read_pdb_seq(pdb_file: str):
    """
    Read SEQRES field from PDB file
    Return
    -----------
    seq_dict: dict
        Dict of chain_id to seq. e.g. {'A': 'EYTISHTGGTLGSSKVTTA'}
    """
    seq_dict = {}
    seq_len_dict = {}
    for line in open(pdb_file):
        if line.startswith('SEQRES '):
            chain_id = line[11]
            seq_len = int(line[13:13 + 4].strip())
            res_list = [res for res in line[19:].strip().split()]
            if len(res_list[0]) == 3:
                ## Protein
                res_list = [restype_3to1.get(res, 'X') for res in res_list]
            elif len(res_list[0]) == 2:
                ## DNA
                res_list = [res[-1] for res in res_list]
            else:
                ## RNA
                res_list = [res for res in res_list]
            frag = "".join(res_list)
            seq_dict[chain_id] = seq_dict.get(chain_id, '') + frag
            seq_len_dict[chain_id] = seq_len
    for chain_id in seq_len_dict:
        if seq_len_dict[chain_id] != len(seq_dict[chain_id]):
            print(
                f"Warning: expect sample length for chain {chain_id}, but got {seq_len_dict[chain_id]} and {len(seq_dict[chain_id])}")
    return seq_dict


def read_pdb_atom_line(atom_line: str):
    """
    Parse ATOM line of PDB file

    Return
    ---------------
    list: [atom_idx, atom_name, restype, chain, res_index, x, y, z, occ, temp ]
    """
    if len(atom_line) > 78:
        atom_line = atom_line.strip()
    assert atom_line.startswith('ATOM')
    atom_idx = atom_line[6:11].strip()
    atom_name = atom_line[12:16].strip()
    restype = atom_line[17:20].strip()
    chain = atom_line[21]
    res_index = atom_line[22:26].strip()
    x, y, z = float(atom_line[30:38].strip()), float(atom_line[38:46].strip()), float(atom_line[46:54].strip())
    occ = float(atom_line[54:60].strip())
    temp = float(atom_line[60:66].strip())
    return [atom_idx, atom_name, restype, chain, res_index, x, y, z, occ, temp]


def write_pdb_atom_line(idx: int, atom_name: str, restype: str, chain: str, res_index: int,
                        x: float, y: float, z: float, occ: float = 0.0, temp: float = 0.0):
    """
    Convert to atom
    """
    assert len(atom_name) <= 4
    assert len(restype) <= 3
    assert len(chain) == 1
    assert len(str(res_index)) <= 4
    atom_mark = atom_name[0]
    if res_index < 1:
        print(f"Warning: res_index should greater than 0")
    if len(restype) < 3:
        return f"ATOM  {idx:5d}  {atom_name:<3s} {restype:>3s} {chain}{res_index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}           {atom_mark:1s}"
    else:
        return f"ATOM  {idx:5d} {atom_name.center(4)} {restype:>3s} {chain}{res_index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}           {atom_mark:1s}"


def write_pdb_helix_line(serial: int, identifier: str, resname1: str, chain1: str, res_index1: int,
                         resname2: str, chain2: str, res_index2: int):
    """
    HELIX    1 AA1 THR A   90  GLY A  109  1                                  20
    """
    assert len(str(serial)) <= 3
    assert len(identifier) <= 3
    assert len(resname1) == len(resname2) == 3
    assert len(chain1) == len(chain2) == 1
    assert len(str(res_index1)) <= 4
    assert len(str(res_index2)) <= 4

    length = res_index2 - res_index1 + 1
    return f"HELIX  {serial:3d} {identifier:3s} {resname1} {chain1} {res_index1:4d}  {resname2} {chain2} {res_index2:4d}  1                               {length:5d}"


def write_pdb_sheet_line(serial: int, identifier: str, resname1: str, chain1: str, res_index1: int,
                         resname2: str, chain2: str, res_index2: int):
    """
    SHEET    1 AA1 3 PHE A 125  LYS A 127  0
    """
    assert len(str(serial)) <= 3
    assert len(identifier) <= 3
    assert len(resname1) == len(resname2) == 3
    assert len(chain1) == len(chain2) == 1
    assert len(str(res_index1)) <= 4
    assert len(str(res_index2)) <= 4

    length = res_index2 - res_index1 + 1
    assert len(str(length)) <= 2
    return f"SHEET  {serial:3d} {identifier:3s}{length:2d} {resname1} {chain1}{res_index1:4d}  {resname2} {chain2}{res_index2:4d}  0"


def save_as_pdb(aatype, residue_index, atom_positions, atom_position_mask, out_file,
                b_factors=None, asym_id=None, full_seq=None, occupancies=None, ss_coding=[1, 2, 3],
                write_ss=False, gap_ter_threshold=600, molecular_type='protein', remark_lines=None):
    """
    Save PDB file from positions

    Warning
    -----------
    residue_index must be zero-based

    Parameters
    -----------
    aatype: [N_res]
    residue_index: [N_res]
    atom_positions: [N_res, atom_num, 3]
    atom_position_mask: [N_res, atom_num]
    out_file: str
    b_factors: [N_res, atom_num] or None
    occupancies: occupancies of protein
    asym_id: [N_res] or None
    full_seq: str or dict
        full sequence, write to SEQRES field
        str type: single chain
        dict type: chain_id -> str mapping
    ss_coding: 3 elements, for index of loop, helix and sheet
    write_ss: bool
        Write the SS information
    molecular_type: protein, DNA or RNA
    """

    assert molecular_type in ('protein', 'RNA', 'DNA')
    if molecular_type == 'protein':
        restypes_with_x = residue_constants.restypes_with_x
    else:
        restypes_with_x = residue_constants.nuctypes_with_x

    if aatype.shape[0] == 1:
        aatype = aatype[0]
    if aatype.ndim == 2:
        aatype = aatype.argmax(1)

    if residue_index.shape[0] == 1:
        residue_index = residue_index[0]

    assert aatype.ndim == 1, f"Expect aatype.ndim == 1, but got {aatype.ndim}"
    assert residue_index.ndim == 1, f"Expect residue_index.ndim == 1, but got {residue_index.ndim}"
    assert atom_positions.ndim == 3, f"Expect atom_positions.ndim == 3, but got {atom_positions.ndim}"
    assert atom_position_mask.ndim == 2, f"Expect atom_position_mask.ndim == 2, but got {atom_position_mask.ndim}"
    if b_factors is not None:
        assert b_factors.ndim == 2, f"Expect b_factors.ndim == 2, but got {b_factors.ndim}"
    else:
        b_factors = np.zeros_like(atom_position_mask)
    if asym_id is not None:
        if asym_id.shape[0] == 1:
            asym_id = asym_id[0]
        assert asym_id.ndim == 1, f"Expect asym_id.ndim == 1, but got {asym_id.ndim}"
    else:
        asym_id = np.zeros_like(aatype)
    asym_id = asym_id.astype(np.int32)
    if write_ss:
        assert len(ss_coding) == 3, f"Expect len(ss_coding) == 3, but got ss_coding={ss_coding}"
        assert occupancies is not None
    if full_seq is not None:
        assert isinstance(full_seq, (str, dict))

    ## Convert call from from_prediction to protein.Protein
    unrelaxed_protein = protein.Protein(atom_positions, aatype, atom_position_mask,
                                        residue_index + 1, asym_id, b_factors, occupancies, molecular_type)

    def check_seq(ch_full_seq, ch_aatype, ch_residx):
        for res_aatype, res_residx in zip(ch_aatype, ch_residx):
            fseq_res = ch_full_seq[res_residx]
            aatype_res = restypes_with_x[res_aatype]
            if fseq_res != aatype_res:
                print(
                    _w(f"Warning: different sequence for full_seq and aatype ({res_residx}): {fseq_res} -- {aatype_res}"))

    SEQRES = []
    if full_seq is not None:
        if isinstance(full_seq, str):
            assert len(set(asym_id)) == 1
            check_seq(full_seq, aatype, residue_index)
            SEQRES = write_pdb_seq(full_seq, protein.PDB_CHAIN_IDS[asym_id[0]], molecular_type=molecular_type)
        else:
            for ch_key in sorted(full_seq.keys()):
                if isinstance(ch_key, (int, np.int32, np.int64)):
                    ch_idx = int(ch_key)
                elif isinstance(ch_key, str) and len(ch_key) == 1:
                    ch_idx = protein.PDB_CHAIN_IDS.index(ch_key)
                else:
                    raise RuntimeError(f"Wrong chain index: {ch_key}")
                mask = (ch_idx == asym_id)
                if mask.sum() > 0:
                    check_seq(full_seq[ch_key], aatype[mask], residue_index[mask])
                SEQRES += write_pdb_seq(full_seq[ch_key], protein.PDB_CHAIN_IDS[ch_idx], molecular_type=molecular_type)

    if write_ss:
        helix_lines, sheet_lines = parse_prot_ss(unrelaxed_protein, ss_coding=ss_coding)
        ss_line = "\n".join(helix_lines) + "\n" + "\n".join(sheet_lines) + "\n"
    else:
        ss_line = ""

    if SEQRES is None or len(SEQRES) == 0:
        SEQRES = ""
    else:
        SEQRES = "\n".join(SEQRES) + "\n"

    if remark_lines is not None:
        assert isinstance(remark_lines, (list, tuple)), f"Expect remark_lines be list, but got {type(remark_lines)}"
        remark = ""
        for line in remark_lines:
            line = line.rstrip()
            remark += "REMARK   3 " + line + "\n"
    else:
        remark = ""

    pdb_str = remark + SEQRES + ss_line + protein.to_pdb(unrelaxed_protein, gap_ter_threshold=gap_ter_threshold)
    print(pdb_str, file=open(out_file, 'w'))


def save_prot_as_pdb(prot: protein.Protein, out_file, full_seq=None, ss_coding=[1, 2, 3], write_ss=False,
                     gap_ter_threshold=600, remark_lines=None):
    """
    Save protein.Protein object as PDB file

    residue_index is 1-based
    """
    save_as_pdb(prot.aatype, prot.residue_index - 1, prot.atom_positions, prot.atom_mask, out_file,
                prot.b_factors, prot.chain_index, full_seq, prot.occupancies, ss_coding,
                write_ss, gap_ter_threshold, prot.molecular_type, remark_lines)



def parse_prot_ss(prot, ss_coding=[1, 2, 3]):
    L_TYPE = ss_coding[0]
    H_TYPE = ss_coding[1]
    S_TYPE = ss_coding[2]
    cur_ss_type = None
    start_ch = None
    start_res_idx = None
    last_res_idx = None
    start_restype = None
    last_restype = None

    helix_lines = []
    sheet_lines = []
    for ch_idx, res_idx, aatype, occ in zip(prot.chain_index, prot.residue_index, prot.aatype, prot.occupancies[:, 1]):
        # ch_idx, res_idx, aatype, occ = ch_idx.item(), res_idx.item(), aatype.item(), occ.item()
        if int(occ) != cur_ss_type or start_ch != ch_idx or abs(res_idx - last_res_idx) > 3:
            if cur_ss_type == H_TYPE and last_res_idx - start_res_idx >= 3:
                helix_lines.append([start_ch, start_res_idx, start_restype, last_res_idx, last_restype])
            elif cur_ss_type == S_TYPE and last_res_idx - start_res_idx >= 3:
                sheet_lines.append([start_ch, start_res_idx, start_restype, last_res_idx, last_restype])
            start_res_idx, start_restype, start_ch = res_idx, restype_1to3.get(restypes_with_x[aatype], 'UNK'), ch_idx
            cur_ss_type = int(occ)
        last_res_idx, last_restype = int(res_idx), restype_1to3.get(restypes_with_x[aatype], 'UNK')
    if cur_ss_type == H_TYPE and last_res_idx - start_res_idx >= 3:
        helix_lines.append([start_ch, start_res_idx, start_restype, last_res_idx, last_restype])
    elif cur_ss_type == S_TYPE and last_res_idx - start_res_idx >= 3:
        sheet_lines.append([start_ch, start_res_idx, start_restype, last_res_idx, last_restype])

    for idx in range(len(helix_lines)):
        start_ch, start_res_idx, start_restype, last_res_idx, last_restype = helix_lines[idx]
        ch_name = PDB_CHAIN_IDS[start_ch]
        helix_lines[idx] = write_pdb_helix_line(min(idx + 1, 999), 'AA1', start_restype, ch_name, start_res_idx,
                                                last_restype, ch_name, last_res_idx)
    for idx in range(len(sheet_lines)):
        start_ch, start_res_idx, start_restype, last_res_idx, last_restype = sheet_lines[idx]
        ch_name = PDB_CHAIN_IDS[start_ch]
        sheet_lines[idx] = write_pdb_sheet_line(min(idx + 1, 999), 'AA1', start_restype, ch_name, start_res_idx,
                                                last_restype, ch_name, last_res_idx)

    return helix_lines, sheet_lines


def fill_protein_to_nongapped(prot, seq_dict, full_padding=False):
    """
    Fill the protein.Protein object as a non-gapped protein with zero padding

    Parameters
    --------------
    prot: protein.Protein object
    seq_dict: dict
        A mapping from chain name (or chain ID) to sequence
    full_padding: bool
        Padding the protein according to the full sequence (including both end region)

    Return
    --------------
    prot: protein.Protein object
    """
    from alphafold.common import protein, residue_constants
    PDB_CHAIN_IDS = protein.PDB_CHAIN_IDS

    if prot.molecular_type == 'protein':
        restypes_with_x = residue_constants.restypes_with_x
    else:
        restypes_with_x = residue_constants.nuctypes_with_x

    aatype, atom_positions, atom_mask, residue_index, chain_index, b_factors, occs = [], [], [], [], [], [], []
    chain_list = sorted(list(set(prot.chain_index)))
    # print(f"Info: {len(chain_list)} chains to padding")

    if prot.occupancies is None:
        prot = protein.from_protein(prot, occupancies=np.zeros_like(prot.b_factors))

    if prot.molecular_type == 'protein':
        atom_num = residue_constants.atom_type_num
    else:
        atom_num = residue_constants.na_atom_type_num

    last_size = 0
    for ch_idx in chain_list:
        if ch_idx in seq_dict:
            chain_seq = seq_dict[ch_idx]
        else:
            chain_seq = seq_dict[PDB_CHAIN_IDS[ch_idx]]

        prot_aatype = prot.aatype[prot.chain_index == ch_idx]
        prot_atom_positions = prot.atom_positions[prot.chain_index == ch_idx]
        prot_atom_mask = prot.atom_mask[prot.chain_index == ch_idx]
        prot_residue_index = prot.residue_index[prot.chain_index == ch_idx]
        prot_chain_index = prot.chain_index[prot.chain_index == ch_idx]
        prot_b_factors = prot.b_factors[prot.chain_index == ch_idx]
        prot_occupancies = prot.occupancies[prot.chain_index == ch_idx]

        def fill_a_res(chain_seq, res_idx):
            aa = restypes_with_x.index(chain_seq[res_idx - 1])
            aatype.append(aa)
            atom_positions.append(np.zeros([atom_num, 3]))
            atom_mask.append(np.zeros([atom_num]))
            residue_index.append(res_idx)
            chain_index.append(ch_idx)
            b_factors.append(np.zeros([atom_num]))
            occs.append(np.zeros([atom_num]))

        if full_padding:
            for res_idx in range(1, prot_residue_index[0]):
                fill_a_res(chain_seq, res_idx)

        for aa, pos, mask, res_idx, ch_idx, b_fac, occ in zip(prot_aatype, prot_atom_positions, prot_atom_mask,
                                                              prot_residue_index, prot_chain_index, prot_b_factors,
                                                              prot_occupancies):
            restype = restypes_with_x[aa]
            assert restype == chain_seq[
                res_idx - 1], f"Expect same aatype, but got {restype} and {chain_seq[res_idx - 1]} in res_idx ({res_idx});prot: {prot}, seq_dict: {seq_dict}"
            if len(residue_index) != 0:
                # Filling the gap
                for res_idx_ in range(residue_index[-1] + 1, res_idx):
                    fill_a_res(chain_seq, res_idx_)
            aatype.append(aa)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res_idx)
            chain_index.append(ch_idx)
            b_factors.append(b_fac)
            occs.append(occ)

        if full_padding:
            for res_idx in range(residue_index[-1] + 1, len(chain_seq) + 1):
                fill_a_res(chain_seq, res_idx)

            assert len(atom_positions) - last_size == len(
                chain_seq), f"Expect same length, but got {len(atom_positions) - last_size} and {len(chain_seq)} in chain {PDB_CHAIN_IDS[ch_idx]}. prot_residue_index: {prot_residue_index}"
            last_size = len(atom_positions)

    prot = protein.Protein(
        np.stack(atom_positions),
        np.array(aatype),
        np.stack(atom_mask),
        np.array(residue_index),
        np.array(chain_index),
        np.stack(b_factors),
        np.stack(occs),
        prot.molecular_type)

    return prot


def read_pdb(pdbfile, padding=False, full_padding=False, molecular_type=None, insertion_code_process='error',
             stderr=sys.stderr):
    """
    Read pdb file

    Parameters
    --------------
    pdbfile: PDB file path
    padding: bool
        Padding the pdb file as continous chains
    full_padding: bool
        Full padding each chains in PDB, overwrite padding mode
    insertion_code_process: error, insert or ignore

    Return
    --------------
    prot: protein.Protein object
    """

    assert insertion_code_process in ('error', 'insert', 'ignore')

    if pdbfile.endswith('.cif'):
        prot, chidx2seq = read_cif_as_prot(pdbfile, molecular_type=molecular_type, stderr=stderr)
    else:
        prot = protein.from_pdb_string(open(pdbfile).read(), molecular_type=molecular_type,
                                       insertion_code_process=insertion_code_process, stderr=stderr)
        chidx2seq = read_pdb_seq(pdbfile)
    if padding or full_padding:
        assert len(chidx2seq) > 0, f"Expect SEQRES in {pdbfile} with padding mode, but no sequence found"
    if padding or full_padding:
        prot = fill_protein_to_nongapped(prot, chidx2seq, full_padding=full_padding)
    return prot


def read_gapped_pdb(pdbfile: str, min_prot_size: int = 1, full_padding: bool = True):
    """
    Read fragmented PDB file

    Return
    -----------------
    seq_dict: dict
    prot_frags: List[AF2_Protein]
    """
    seq_dict = read_pdb_seq(pdbfile)

    def add_new_frag():
        nonlocal atom_lines
        nonlocal last_chain_idx
        if len(atom_lines) >= min_prot_size:
            prot = protein.from_pdb_string("".join(atom_lines))
            if len(seq_dict) > 0:
                prot = fill_protein_to_nongapped(prot, seq_dict, full_padding=full_padding)
            prot_frags.append(prot)
        atom_lines = []
        last_chain_idx = None

    seqres_lines = []
    atom_lines = []
    prot_frags = []
    last_chain_idx = None
    for line in open(pdbfile):
        if line.startswith('ATOM'):
            (atom_idx, atom_name, restype, chain, res_index, x, y, z, occ, temp) = read_pdb_atom_line(line)
            res_index = int(res_index)
            if len(seq_dict) > 0:
                seqres_aa = seq_dict[chain][res_index - 1]
                if seqres_aa != restype_3to1[restype]:
                    print(
                        f"Warning: different sequence for residue and SEQRES ({chain}, {res_index}): {restype_3to1[restype]} -- {seqres_aa}")
            if last_chain_idx is not None and last_chain_idx != chain:
                add_new_frag()
            atom_lines.append(line)
            last_chain_idx = chain
        elif line.startswith('SEQRES'):
            seqres_lines.append(line)
        elif line.startswith('TER'):
            add_new_frag()
    add_new_frag()
    return prot_frags


def to_pdb(pdb_id, filename, apply_symmetry=False, only_ca: Union[str, bool] = 'auto', raise_error=False,
           molecular_type='protein'):
    """
    Parse cif file and as save as PDB file

    only_ca: str or bool
        'auto' -- Automatic discriminate use or not use all atom
        True   -- Force to print Ca atoms only
        False  -- Force to print all atoms
    raise_error: Raise error when number of residues exceed the threshold
    molecular_type: protein, RNA, DNA, NA (RNA and DNA)
        when molecular_type is NA, filename should be a list of two files (RNA_save_file and DNA_save_file)
    """
    import pdb_features
    from alphafold.common.protein import PDB_MAX_CHAINS, PDB_CHAIN_IDS

    assert molecular_type in ('protein', 'RNA', 'DNA', 'NA')
    assert isinstance(only_ca, (str, bool))
    if isinstance(only_ca, str):
        assert only_ca == 'auto'

    if molecular_type == 'NA':
        assert isinstance(filename, (list,
                                     tuple)), "when molecular_type is NA, filename should be a list of two files (RNA_save_file and DNA_save_file)"
    else:
        assert isinstance(filename, str)

    def save_pdb_3d(pdb_3d, filename, apply_symmetry, only_ca, raise_error, molecular_type, atom_num):
        chain_ids = list(range(len(PDB_CHAIN_IDS)))
        for ch_name in pdb_3d:
            chain = ch_name.split('_')[1]
            if len(chain) == 1:
                chain_ids.remove(PDB_CHAIN_IDS.index(chain))

        chain2seq = {}
        atom_positions = []
        all_atom_masks = []
        aatype = []
        residue_index = []
        chain_index = []
        b_factors = []

        ch_idx = 0
        for ch_name, p in pdb_3d.items():
            chain = ch_name.split('_')[1]
            num_res = p['all_atom_positions'].shape[0]
            atom_positions.append(p['all_atom_positions'])
            if 'aatype' in p:
                aatype.append(p['aatype'].argmax(1))
            else:
                aatype.append(p['natype'].argmax(1))
            all_atom_masks.append(p['all_atom_masks'])
            residue_index.append(np.arange(num_res))
            if len(chain) == 1:
                chain_index.append(np.ones(num_res, dtype=np.int32) * PDB_CHAIN_IDS.index(chain))
                chain2seq[PDB_CHAIN_IDS.index(chain)] = p['sequence'].decode()
            else:
                chain_index.append(np.ones(num_res, dtype=np.int32) * chain_ids[ch_idx])
                chain2seq[chain_ids[ch_idx]] = p['sequence'].decode()
                ch_idx += 1
            b_factors.append(np.zeros([num_res, atom_num]))

        if apply_symmetry:
            import pdb_data
            symmetry = pdb_data.get_cif_symmetry(pdb_id)
            for sym in symmetry:
                if sym.is_identity():
                    continue
                rot = sym.matrix
                trans = sym.vector
                for ch_name, p in pdb_3d.items():
                    num_res = p['all_atom_positions'].shape[0]
                    all_atom_positions = np.dot(p['all_atom_positions'], rot.T) + trans
                    atom_positions.append(all_atom_positions)
                    if 'aatype' in p:
                        aatype.append(p['aatype'].argmax(1))
                    else:
                        aatype.append(p['natype'].argmax(1))
                    all_atom_masks.append(p['all_atom_masks'])
                    residue_index.append(np.arange(num_res))
                    chain_index.append(np.ones(num_res, dtype=np.int32) * chain_ids[ch_idx])
                    b_factors.append(np.zeros([num_res, atom_num]))
                    chain2seq[chain_ids[ch_idx]] = p['sequence'].decode()
                    ch_idx += 1

        atom_positions = np.concatenate(atom_positions)
        all_atom_masks = np.concatenate(all_atom_masks)
        aatype = np.concatenate(aatype)
        residue_index = np.concatenate(residue_index)
        chain_index = np.concatenate(chain_index)
        b_factors = np.concatenate(b_factors)

        if only_ca == 'auto':
            only_ca = True if (all_atom_masks.sum() > 99999) else False

        if all_atom_masks.sum() > 99999 and not only_ca:
            print(f"Expect atom num less than 99999, but got {all_atom_masks.sum()}, suggest use only_ca=True")
            if raise_error:
                raise RuntimeError("Error")

        if only_ca:
            all_atom_masks[:, 0] = 0
            all_atom_masks[:, 2:] = 0
            if all_atom_masks.sum() > 99999:
                print(f"TOO MANY RESIDUES: Expect atom num less than 99999, but got {all_atom_masks.sum()}")
                if raise_error:
                    raise RuntimeError("Error")

        save_as_pdb(aatype,
                    residue_index,
                    atom_positions,
                    all_atom_masks,
                    filename,
                    b_factors=b_factors,
                    asym_id=chain_index,
                    full_seq=chain2seq,
                    molecular_type=molecular_type)

    if molecular_type == 'protein':
        atom_num = residue_constants.atom_type_num
        pdb_3d = pdb_features.get_pdb_3D_info(pdb_id)
        if pdb_3d is not None:
            assert len(
                pdb_3d) <= PDB_MAX_CHAINS, f"Number chain exceed PDB_MAX_CHAINS: {len(pdb_3d)} > {PDB_MAX_CHAINS} use extend_PDB_Chains to extend the chain range"
            save_pdb_3d(pdb_3d, filename, apply_symmetry, only_ca, raise_error, molecular_type, atom_num)
    else:
        atom_num = residue_constants.na_atom_type_num
        pdb_3d = pdb_features.get_pdb_na_3D_info(pdb_id)
        assert len(
            pdb_3d) <= PDB_MAX_CHAINS, f"Number chain exceed PDB_MAX_CHAINS: {len(pdb_3d)} > {PDB_MAX_CHAINS} use extend_PDB_Chains to extend the chain range"
        if molecular_type in ('DNA', 'RNA'):
            pdb_3d = {k: v for k, v in pdb_3d.items() if v['molecular_type'] == molecular_type}
            if len(pdb_3d) > 0:
                save_pdb_3d(pdb_3d, filename, apply_symmetry, only_ca, raise_error, molecular_type, atom_num)
        else:
            for molecular_type, save_file in zip(('RNA', 'DNA'), filename):
                pdb_3d_ = {k: v for k, v in pdb_3d.items() if v['molecular_type'] == molecular_type}
                if len(pdb_3d_) > 0:
                    save_pdb_3d(pdb_3d_, save_file, apply_symmetry, only_ca, raise_error, molecular_type, atom_num)



def read_cif_as_prot(cif_file, molecular_type=None, stderr=sys.stderr):
    """
    Read mmCIF file as Protein object (only support protein not nucleic acid now)
    """
    from Bio.PDB import MMCIF2Dict
    import alphafold.common.residue_constants as rc
    import alphafold.common.protein as AF2_Protein_Module

    atom_types = None
    atom_order = None
    restype_order_with_x = None
    restype_3to1 = None

    def set_rc_items():
        nonlocal atom_types
        nonlocal atom_order
        nonlocal restype_order_with_x
        nonlocal restype_3to1
        if molecular_type == 'protein':
            atom_types = rc.atom_types
            atom_order = rc.atom_order
            restype_order_with_x = rc.restype_order_with_x
            restype_3to1 = rc.restype_3to1
        elif molecular_type in ('RNA', 'DNA'):
            atom_types = rc.na_atom_types
            atom_order = rc.na_atom_order
            restype_order_with_x = rc.nuctype_order_with_x
            restype_3to1 = {n: n for n in rc.nuctypes_with_x} if molecular_type == 'RNA' else {'D' + n: n for n in
                                                                                               rc.nuctypes_with_x}

    set_rc_items()
    cif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)
    chain_atoms = {}
    for idx, (atom_group, atom_type, aa_type, auch, res_idx, x, y, z) in enumerate(zip(
            cif_dict['_atom_site.group_PDB'],
            cif_dict['_atom_site.label_atom_id'],
            cif_dict['_atom_site.label_comp_id'],
            cif_dict['_atom_site.auth_asym_id'],
            cif_dict['_atom_site.auth_seq_id'],
            cif_dict['_atom_site.Cartn_x'],
            cif_dict['_atom_site.Cartn_y'],
            cif_dict['_atom_site.Cartn_z'])):
        if atom_group != 'ATOM':
            continue
        if '_atom_site.pdbx_PDB_model_num' in cif_dict and int(cif_dict['_atom_site.pdbx_PDB_model_num'][idx]) != 1:
            print(f"Expect one model in {cif_file}, but got {cif_dict['_atom_site.pdbx_PDB_model_num'][idx]}",
                  file=stderr)
            continue
        if molecular_type == 'protein':
            if len(aa_type) != 3:
                print(f"Expect len(aa_type)==3, but got {aa_type}", file=stderr)
                continue
        elif molecular_type == 'RNA':
            if len(aa_type) != 1:
                print(f"Expect len(aa_type)==1, but got {aa_type}", file=stderr)
                continue
        elif molecular_type == 'DNA':
            if len(aa_type) != 2:
                print(f"Expect len(aa_type)==2, but got {aa_type}", file=stderr)
                continue
        else:
            if len(aa_type) == 3:
                molecular_type = 'protein'
            elif len(aa_type) == 1:
                molecular_type = 'RNA'
            elif len(aa_type) == 2:
                molecular_type = 'DNA'
            set_rc_items()

        b_fac = cif_dict['_atom_site.B_iso_or_equiv'][idx] if '_atom_site.B_iso_or_equiv' in cif_dict else 0
        occupancy = cif_dict['_atom_site.occupancy'][idx] if '_atom_site.occupancy' in cif_dict else 0

        try:
            res_idx = int(res_idx)
        except:
            continue
        if auch not in chain_atoms:
            chain_atoms[auch] = {}
        if res_idx not in chain_atoms[auch]:
            chain_atoms[auch][res_idx] = [aa_type,
                                          np.zeros([len(atom_types), 3]),  # xyz position
                                          np.zeros([len(atom_types)]),  # mask
                                          np.zeros([len(atom_types)]),  # Occupancy
                                          np.zeros([len(atom_types)])  # B factor
                                          ]
        assert chain_atoms[auch][res_idx][0] == aa_type
        order = atom_order[atom_type]
        chain_atoms[auch][res_idx][1][order] = [x, y, z]
        chain_atoms[auch][res_idx][2][order] = 1
        chain_atoms[auch][res_idx][3][order] = occupancy
        chain_atoms[auch][res_idx][4][order] = b_fac

    auch2seq = {}
    if '_entity_poly.pdbx_seq_one_letter_code_can' in cif_dict:
        for auchs, seq in zip(cif_dict['_entity_poly.pdbx_strand_id'],
                              cif_dict['_entity_poly.pdbx_seq_one_letter_code_can']):
            for auch in auchs.split(','):
                auch2seq[auch] = seq

    aatypes = []
    chain_index = []
    atom_positions = []
    atom_masks = []
    occupancies = []
    b_factors = []
    residue_index = []
    chidx2seq = {}
    for chain_idx, auch in enumerate(chain_atoms):
        if auch in auch2seq:
            chidx2seq[chain_idx] = auch2seq[auch]
        for res_idx in sorted(chain_atoms[auch].keys()):
            aatype, res_atom_position, res_atom_mask, res_atom_occ, res_atom_bfac = chain_atoms[auch][res_idx]
            aatypes.append(restype_order_with_x[restype_3to1[aatype]])
            chain_index.append(chain_idx)
            atom_positions.append(res_atom_position)
            atom_masks.append(res_atom_mask)
            occupancies.append(res_atom_occ)
            b_factors.append(res_atom_bfac)
            residue_index.append(res_idx)

    aatypes = np.array(aatypes)
    chain_index = np.array(chain_index)
    atom_positions = np.array(atom_positions)
    atom_masks = np.array(atom_masks)
    occupancies = np.array(occupancies)
    b_factors = np.array(b_factors)
    residue_index = np.array(residue_index)

    prot = AF2_Protein_Module.Protein(atom_positions, aatypes, atom_masks, residue_index, chain_index,
                                      b_factors, occupancies=occupancies, molecular_type=molecular_type)
    return prot, chidx2seq


def chidx2auch(chidx, auch_length=2):
    """
    Convert chain index to auth chain name. e.g. 0 -> AA, 899 -> Of
    """
    assert auch_length > 0, f"Expect auch_length > 0, but got {auch_length}"
    chidx = int(chidx)

    auth_chain_head_list = [chr(ord('A') + i) for i in range(26)] + [chr(ord('a') + i) for i in range(26)]
    auth_chain_list = [chr(ord('A') + i) for i in range(26)] + [chr(ord('a') + i) for i in range(26)] + [
        chr(ord('0') + i) for i in range(10)]

    max_auch = len(auth_chain_head_list) * len(auth_chain_list) ** (auch_length - 1)
    assert max_auch > chidx, f"Expect max_auch {max_auch}, but got chidx {chidx}"

    leaved_auch_length = auch_length
    auth_ch = ""
    while leaved_auch_length > 0:
        dim = len(auth_chain_list) ** (leaved_auch_length - 1)
        if leaved_auch_length == auch_length:
            auth_ch += auth_chain_head_list[chidx // dim]
        else:
            auth_ch += auth_chain_list[chidx // dim]
        chidx -= dim * (chidx // dim)
        leaved_auch_length -= 1
    return auth_ch


def get_cif_atom_head():
    """
    Get mmCIF _atom_site list, same format as RCSB PDB
    """
    identifier = '_atom_site'
    data_list = ['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id', 'label_comp_id', 'label_asym_id', \
                 'label_entity_id', 'label_seq_id', 'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy',
                 'B_iso_or_equiv', \
                 'pdbx_formal_charge', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id',
                 'pdbx_PDB_model_num']
    data_list = [identifier + '.' + d for d in data_list]
    return data_list


def write_cif_atom_line(idx: int, atom_name: str, restype: str, entity_id: int, chain: str, res_index: int,
                        x: float, y: float, z: float, auth_chain: str, occ: float = 0.0, temp: float = 0.0):
    """
    Get mmCIF ATOM line
    """
    type_symbol = atom_name[0]
    if "'" in atom_name:
        atom_name = f'"{atom_name}"'
    return f"ATOM {idx} {type_symbol} {atom_name} . {restype} {chain} {entity_id} {res_index} ? {x:.3f} {y:.3f} {z:.3f} {occ:.3f} {temp:.3f} ? {res_index} {restype} {auth_chain} {atom_name} 1"


def to_cif(prot, chidx2entityidx, auch_length=2):
    """
    Convert prot object to mmCIF String
    """
    cif_str = "loop_\n"
    cif_atom_heads = get_cif_atom_head()
    for item in cif_atom_heads:
        cif_str += '' + item + '\n'
    if prot.molecular_type == 'protein':
        aatypes = rc.restypes_with_x
        atom_types = rc.atom_types
    else:
        aatypes = rc.nuctypes_with_x
        atom_types = rc.na_atom_types

    num_res = prot.aatype.shape[0]
    occupancies = np.ones_like(prot.atom_mask) if prot.occupancies is None else prot.occupancies
    b_factors = np.zeros_like(prot.atom_mask) if prot.b_factors is None else prot.b_factors

    for atom_idx, (aatype, res_idx, ch_idx, pos, mask, occ, b_fac) in enumerate(
            zip(prot.aatype, prot.residue_index, prot.chain_index, prot.atom_positions, \
                prot.atom_mask, occupancies, b_factors)):
        aatype = aatypes[aatype]
        if prot.molecular_type == 'protein':
            aatype = rc.restype_1to3[aatype]
        elif prot.molecular_type == 'DNA':
            aatype = 'D' + aatype
        # ch     = AF2_Prot_Module.PDB_CHAIN_IDS[ch_idx]
        auch = chidx2auch(ch_idx, auch_length=auch_length)
        for idx, ((x, y, z), m, occ_, b_fac_) in enumerate(zip(pos, mask, occ, b_fac)):
            if m > 0:
                atom_name = atom_types[idx]
                entity_id = chidx2entityidx[ch_idx]
                line = write_cif_atom_line(atom_idx + 1, atom_name, aatype, entity_id, auch, res_idx, x, y, z, auch,
                                           occ_, b_fac_)
                cif_str += '' + line + "\n"
    cif_str += "#\n"
    return cif_str


def check_prot_seq_match(prot, chidx2seq):
    """
    Check protein aatype - input sequence match
    """
    for chidx in np.unique(prot.chain_index):
        chidx = int(chidx)
        chseq = chidx2seq[chidx]
        # ch_name = AF2_Prot_Module.PDB_CHAIN_IDS[chidx]
        auch = chidx2auch(chidx, auch_length=2)
        ch_prot = prot.filter_by_chain(int(chidx))
        for res_idx, res_aatype in zip(ch_prot.residue_index, ch_prot.aatype):
            res_type = rc.nuctypes_with_x[res_aatype] if prot.molecular_type != 'protein' else rc.restypes_with_x[
                res_aatype]
            if chseq[res_idx - 1] != res_type:
                print(
                    f"Chain {auch}, chain index {chidx}, expect same restype ({res_idx + 1}) but got {chseq[res_idx - 1]} and {res_type}")


def save_prot_as_cif(prot, out_file, full_seq=None):
    """
    Save protein.Protein object as mmCIF file

    residue_index is 1-based
    """
    if prot.chain_index.max() < 52:
        auch_length = 1
    elif prot.chain_index.max() < 3224:
        auch_length = 2
    else:
        auch_length = 3

    auch2seq = {}  # auth_chain_index -> Sequence
    chidx2seq = {}  # chain_index -> Sequence
    if full_seq is not None:
        if isinstance(full_seq, str):
            assert len(np.unique(
                prot.chain_index)) == 1, f"Expect single chain in prot when full_seq==None, bit got {np.unique(prot.chain_index)}"
            full_seq = {int(prot.chain_index[0].item()): full_seq}

        for ch, seq in full_seq.items():
            if isinstance(ch, str):
                chidx = AF2_Prot_Module.PDB_CHAIN_IDS.index(ch)
            else:
                chidx = ch
            auch = chidx2auch(chidx, auch_length=auch_length)
            auch2seq[auch] = seq
            chidx2seq[chidx] = seq
    else:
        for chidx in np.unique(prot.chain_index):
            ch_prot = prot.filter_by_chain(int(chidx))
            seq = ['X'] * ch_prot.residue_index.max()
            for aatype, res_idx in zip(ch_prot.aatype, ch_prot.residue_index):
                seq[res_idx - 1] = rc.nuctypes_with_x[aatype] if prot.molecular_type != 'protein' else \
                rc.restypes_with_x[aatype]
            seq = "".join(seq)
            auch = chidx2auch(chidx, auch_length=auch_length)
            auch2seq[auch] = seq
            chidx2seq[chidx] = seq

    check_prot_seq_match(prot, chidx2seq)
    seq_str, seq2entityidx = write_cif_seq(auch2seq, molecular_type=prot.molecular_type)
    chidx2entityidx = {}
    for chidx in np.unique(prot.chain_index):
        auch = chidx2auch(chidx, auch_length=auch_length)
        seq = auch2seq[auch]
        entityidx = seq2entityidx[seq]
        chidx2entityidx[chidx] = entityidx
    cif_str = to_cif(prot, chidx2entityidx, auch_length=auch_length)
    print(f"data_1\n#\n" + seq_str + cif_str, file=open(out_file, 'w'))


def write_cif_seq(auch2seq: dict, molecular_type=None):
    """
    Write Sequence to mmCIF _entity_poly field

    Parameters
    -----------
    auch2seq: dict
    molecular_type: RNA, DNA, protein

    Return
    -----------
    cif_seq_string
    """
    if molecular_type is None:
        long_seq = "".join(auch2seq.values())
        if 'U' in long_seq:
            molecular_type = 'RNA'
        elif (long_seq.count('A') + long_seq.count('T') + long_seq.count('C') + long_seq.count('G')) / len(
                long_seq) > 0.8:
            molecular_type = 'DNA'
        else:
            molecular_type = 'protein'

    cif_seq_string = "loop_\n"
    identifier = '_entity_poly'
    data_list = ['entity_id', 'type', 'nstd_linkage', 'nstd_monomer', 'pdbx_seq_one_letter_code',
                 'pdbx_seq_one_letter_code_can', \
                 'pdbx_strand_id', 'pdbx_target_identifier']
    data_list = [identifier + '.' + d for d in data_list]
    for item in data_list:
        cif_seq_string += item + "\n"

    seq2auch = {}
    for auch, seq in auch2seq.items():
        try:
            seq2auch[seq].append(auch)
        except KeyError:
            seq2auch[seq] = [auch]
    _entity_poly_seq_str = "loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n_entity_poly_seq.hetero\n"
    seq2entityidx = {}
    for idx, seq in enumerate(sorted(list(seq2auch.keys()), key=lambda x: len(x), reverse=True)):
        type_ = {'protein': "'polypeptide(L)'", 'RNA': 'polyribonucleotide', 'DNA': 'polydeoxyribonucleotide'}[
            molecular_type]
        cif_seq_string += f"{idx + 1} {type_} no no\n"
        seq_list = list(seq)
        seq_str = "".join([f"(D{s})" for s in seq_list]) if molecular_type == 'DNA' else seq
        cif_seq_string += ';' + seq_str + '\n;\n;' + seq + '\n;\n'
        auchs = seq2auch[seq]
        auchs = ",".join(auchs)
        cif_seq_string += f'{auchs} ?\n'
        for res_idx, res in enumerate(seq):
            if molecular_type == 'protein':
                _entity_poly_seq_str += f"{idx + 1} {res_idx + 1} {rc.restype_1to3[res]} n\n"
            elif molecular_type == 'RNA':
                _entity_poly_seq_str += f"{idx + 1} {res_idx + 1} {res} n\n"
            elif molecular_type == 'DNA':
                _entity_poly_seq_str += f"{idx + 1} {res_idx + 1} D{res} n\n"
        seq2entityidx[seq] = idx + 1
    cif_seq_string += "#\n"
    _entity_poly_seq_str += "#\n"
    return cif_seq_string + _entity_poly_seq_str, seq2entityidx


def fix_pdb_file(input_pdb_file, verbose=True) -> protein.Protein:
    """
    Fix PDB file with missing atoms

    Parameters
    ------------------
    input_pdb_file: Input PDB file, fix in place

    Return
    ------------------
    protein.Protein object
    """
    assert shutil.which('pdbfixer') is not None

    if input_pdb_file.endswith('.cif'):
        prot, seq_dict = read_cif_as_prot(input_pdb_file)
    else:
        seq_dict = read_pdb_seq(input_pdb_file)
        assert len(seq_dict) == 1, "Multiple sequence found"
        prot = read_pdb(input_pdb_file, full_padding=True)
    resol_mask = (prot.atom_mask.sum(1) > 0)
    theoretical_mask = residue_constants.restype_atom37_mask[prot.aatype][resol_mask]
    true_mask = prot.atom_mask[resol_mask]
    if np.any(true_mask - theoretical_mask < 0):
        cmd = f"CUDA_VISIBLE_DEVICES= pdbfixer {input_pdb_file} --add-atoms=heavy --keep-heterogens=none --output={input_pdb_file}"
        if verbose:
            print(cmd)
        os.system(cmd)
        prot_ = read_pdb(input_pdb_file)
        prot_ = fill_protein_to_nongapped(prot_, seq_dict, full_padding=True)
        prot_ = protein.from_protein(prot_, b_factors=prot.b_factors)
        if input_pdb_file.endswith('.cif'):
            save_prot_as_cif(prot_, input_pdb_file, seq_dict)
        else:
            save_prot_as_pdb(prot_, input_pdb_file, seq_dict)
        return prot_
    return prot


def deduplicate_pdb_files(pdb_file_list, dedup_rmsd=3.0):
    """
    Deduplicate similar PDBs with superimpose.
    Warning: All pdb files must be full padded and have same length.

    Parameters
    -------------
    pdb_file_list: list of template information dict
    dedup_rmsd: RMSD cutoff for similar templates

    Return
    -------------
    Index list of selected pdb files.
    """
    from Bio.SVDSuperimposer import SVDSuperimposer
    def temp_rms(position_1, mask_1, position_2, mask_2):
        mask = mask_1[:, 1].astype(np.bool_) & mask_2[:, 1].astype(np.bool_)
        if mask.sum() < 5:  # Overlapped residues count < 5
            return 0.0
        sup = SVDSuperimposer()
        sup.set(position_1[mask, 1], position_2[mask, 1])
        sup.run()
        rms = sup.get_rms()
        return rms

    if len(pdb_file_list) == 0:
        return []
    if len(pdb_file_list) == 1:
        return [0]

    prot_list = [read_pdb(file, full_padding=True) for file in pdb_file_list]
    assert len(set([prot.aatype.shape[0] for prot in prot_list])) == 1, "Multiple size protein found"
    dist_matrix = np.zeros([len(prot_list), len(prot_list)], dtype=np.float32)

    for i_ in range(len(prot_list)):
        prot_i = prot_list[i_]
        for j_ in range(0, i_):
            prot_j = prot_list[j_]
            dist_matrix[i_, j_] = dist_matrix[j_, i_] = temp_rms(prot_i.atom_positions, prot_i.atom_mask,
                                                                 prot_j.atom_positions, prot_j.atom_mask)
    dist_matrix_ = dist_matrix.copy()
    selected_temp_index = []
    while dist_matrix_.max() > dedup_rmsd:
        index = np.argmax(dist_matrix_)
        x_idx = index // dist_matrix_.shape[0]
        y_idx = index % dist_matrix_.shape[0]
        if all([dist_matrix[idx, x_idx] > dedup_rmsd for idx in selected_temp_index]):
            selected_temp_index.append(x_idx)
        if all([dist_matrix[idx, y_idx] > dedup_rmsd for idx in selected_temp_index]):
            selected_temp_index.append(y_idx)
        dist_matrix_[x_idx, :] = 0.0
        dist_matrix_[:, x_idx] = 0.0
        dist_matrix_[:, y_idx] = 0.0
        dist_matrix_[y_idx, :] = 0.0

    if len(selected_temp_index) == 0:
        selected_temp_index.append(0)

    selected_temp_index = sorted(selected_temp_index)
    return selected_temp_index


