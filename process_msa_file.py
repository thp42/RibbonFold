import pickle, gzip
import sys
import argparse

sys.path.append('./af2/')

from alphafold.data.pipeline import make_msa_features, make_sequence_features
from alphafold.data import parsers



def make_pkl_gz(input_fasta_path, a3m_path, output_pklgz):
    with open(input_fasta_path) as f:
        input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
        raise ValueError(
            f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)


    with open(a3m_path) as f1:
        input_msa_str = f1.read()
    msa = parsers.parse_a3m(input_msa_str)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)
    msa_features = make_msa_features((msa,))

    features = {**sequence_features, **msa_features}

    pickle.dump(features, gzip.open(output_pklgz, 'wb'), protocol=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert FASTA and MSA (a3m) to PKL.GZ file.")
    
    parser.add_argument("--input_fasta", type=str, required=True, help="Input FASTA file path")
    parser.add_argument("--msa_file", type=str, required=True, help="Input MSA (a3m) file path")
    parser.add_argument("--output", type=str, required=True, help="Output PKL.GZ file path")
    
    args = parser.parse_args()
    
    make_pkl_gz(args.input_fasta, args.msa_file, args.output)
