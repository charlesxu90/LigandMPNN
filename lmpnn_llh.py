# adapted from ProteinMPNN/protein_mpnn_utils.py

import sys
import os
import warnings
import torch
import copy
import argparse
from Bio import SeqIO

from tqdm import tqdm
from Bio.PDB import PDBParser
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def load_mpnn_model(args, device):
    if args.model_type == "protein_mpnn":
        checkpoint_path = args.checkpoint_protein_mpnn
    elif args.model_type == "ligand_mpnn":
        checkpoint_path = args.checkpoint_ligand_mpnn
    elif args.model_type == "per_residue_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_per_residue_label_membrane_mpnn
    elif args.model_type == "global_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_global_label_membrane_mpnn
    elif args.model_type == "soluble_mpnn":
        checkpoint_path = args.checkpoint_soluble_mpnn
    else:
        print("Choose one of the available models")
        sys.exit()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if args.model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
        k_neighbors = checkpoint["num_edges"]
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
        k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, atom_context_num


def read_fasta(path, **kwargs):
    """ Read fasta file and return as a list of sequences and their descriptions """
    seqs = [str(fa.seq) for fa in SeqIO.parse(path, "fasta")]
    descriptions = [fa.description for fa in SeqIO.parse(path, "fasta")]
    return seqs, descriptions


def score_llhs(args, pdb_path, model, atom_context_num, device):

    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
        pdb_path,
        device=device,
        chains=[args.chain],
        parse_all_atoms=args.ligand_mpnn_use_side_chain_context,
        parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy
    )

    # make chain_letter + residue_idx + insertion_code mapping to integers
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
    chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
    encoded_residues = []
    for i, R_idx_item in enumerate(R_idx_list):
        tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
        encoded_residues.append(tmp)
    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    encoded_residue_dict_rev = dict(
        zip(list(range(len(encoded_residues))), encoded_residues)
    )


    chains_to_design_list = [args.chain]
    chain_mask = torch.tensor(
        np.array([item in chains_to_design_list for item in protein_dict["chain_letters"]],
            dtype=np.int32,),
        device=device,)

    # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
    protein_dict["chain_mask"] = chain_mask
    

    # set other atom bfactors to 0.0
    if other_atoms:
        other_bfactors = other_atoms.getBetas()
        other_atoms.setBetas(other_bfactors * 0.0)

    # adjust input PDB name by dropping .pdb if it does exist
    name = pdb_path[pdb_path.rfind("/") + 1 :]
    if name[-4:] == ".pdb":
        name = name[:-4]

    with torch.no_grad():
        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
            use_atom_context=args.ligand_mpnn_use_atom_context,
            number_of_ligand_atoms=atom_context_num,
            model_type=args.model_type,
        )
        feature_dict["batch_size"] = args.batch_size
        B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
        feature_dict["symmetry_residues"] = [[]]

        logits_list = []
        probs_list = []
        log_probs_list = []
        decoding_order_list = []
        for _ in range(10):
            feature_dict["randn"] = torch.randn(
                [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                device=device,
            )
            # print("feature_dict['randn'].shape:", feature_dict["randn"].shape)
            if args.autoregressive_score:
                    score_dict = model.score(feature_dict, use_sequence=args.use_sequence)
            else:
                score_dict = model.single_aa_score(feature_dict, use_sequence=args.use_sequence)
            
            logits_list.append(score_dict["logits"])
            log_probs_list.append(score_dict["log_probs"])
            probs_list.append(torch.exp(score_dict["log_probs"]))
            decoding_order_list.append(score_dict["decoding_order"])

        log_probs_stack = torch.cat(log_probs_list, 0)
        logits_stack = torch.cat(logits_list, 0)
        probs_stack = torch.cat(probs_list, 0)
        decoding_order_stack = torch.cat(decoding_order_list, 0)

        out_dict = {}
        out_dict["logits"] = logits_stack.cpu().numpy()
        out_dict["probs"] = probs_stack.cpu().numpy()
        out_dict["log_probs"] = log_probs_stack.cpu().numpy()
        out_dict["decoding_order"] = decoding_order_stack.cpu().numpy()
        out_dict["native_sequence"] = feature_dict["S"][0].cpu().numpy()
        out_dict["mask"] = feature_dict["mask"][0].cpu().numpy()
        out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu().numpy() #this affects decoding order
        out_dict["alphabet"] = alphabet
        out_dict["residue_names"] = encoded_residue_dict_rev

        mean_probs = np.mean(out_dict["probs"], 0)
        std_probs = np.std(out_dict["probs"], 0)
        sequence = [restype_int_to_str[AA] for AA in out_dict["native_sequence"]]
        mean_dict = {}
        std_dict = {}
        for residue in range(L):
            mean_dict_ = dict(zip(alphabet, mean_probs[residue]))
            mean_dict[encoded_residue_dict_rev[residue]] = mean_dict_
            std_dict_ = dict(zip(alphabet, std_probs[residue]))
            std_dict[encoded_residue_dict_rev[residue]] = std_dict_

        out_dict["sequence"] = sequence
        out_dict["mean_of_probs"] = mean_dict
        out_dict["std_of_probs"] = std_dict
    return out_dict


def main(args):
    seq_path = args.input
    
    # load wt seq
    dir_path = os.path.dirname(seq_path)
    wt_seq_path = os.path.join(dir_path, 'wt.fasta')
    seqs, _ = read_fasta(wt_seq_path)
    if len(seqs) == 0:
        raise ValueError(f"wt_seq not found in {wt_seq_path}")
    wt_seq = seqs[0]
    
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    model, atom_context_num = load_mpnn_model(args, device)
    out_dict = score_llhs(args, args.pdb, model, atom_context_num, device)

    # obtain mut probabilities
    NAT_AAS = list('ACDEFGHIKLMNPQRSTVWY')
    pos = []
    pos_aa_probs = []
    pos_aa_probs_std = []
    for aa_pos in out_dict['mean_of_probs'].keys():
        aa_probs = out_dict['mean_of_probs'][aa_pos]
        pos_aa_probs.append([aa_probs[aa] for aa in NAT_AAS])
        pos_aa_probs_std.append([out_dict['std_of_probs'][aa_pos][aa] for aa in NAT_AAS])
        pos.append(aa_pos[1:])  # remove chain letter 

    df_mut_probs = pd.DataFrame(pos_aa_probs, columns=NAT_AAS, index=pos)
    df_mut_probs['pos'] = pos
    df_mut_probs['ref'] = [wt_seq[int(p) - 1] for p in pos]
    
    df_mut_probs.to_csv(f"{args.pdb}_ligandmpnn_mut_probs.csv", index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate log likelihood of mutants using LigandMPNN")
    parser.add_argument('--pdb', '-p', type=str, required=True, help='location of the PDB file')
    parser.add_argument('--input', '-i', type=str, default='./data/fireprot_mapped.csv', help='location of the mutant dataset')

    parser.add_argument('--chain', type=str, default='A', help='chain ID of the PDB file')
    parser.add_argument('--lmpnn_loc', type=str, default='./', help='path to ligandmpnn dir')
    parser.add_argument("--batch_size", type=int, default=10, help="Number of sequence to generate per one pass.",)
    parser.add_argument("--model_type", type=str, default="ligand_mpnn", help="Choose your model: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn",)
    parser.add_argument("--checkpoint_protein_mpnn", type=str, default="./model_params/proteinmpnn_v_48_020.pt", help="Path to model weights.",)
    parser.add_argument("--checkpoint_ligand_mpnn", type=str, default="./model_params/ligandmpnn_v_32_010_25.pt", help="Path to model weights.",)
    parser.add_argument("--checkpoint_per_residue_label_membrane_mpnn", type=str, default="./model_params/per_residue_label_membrane_mpnn_v_48_020.pt", help="Path to model weights.",)
    parser.add_argument("--checkpoint_global_label_membrane_mpnn", type=str, default="./model_params/global_label_membrane_mpnn_v_48_020.pt", help="Path to model weights.",)
    parser.add_argument("--checkpoint_soluble_mpnn", type=str, default="./model_params/solublempnn_v_48_020.pt", help="Path to model weights.",)
    parser.add_argument("--autoregressive_score", type=int, default=0, help="1 - run autoregressive scoring function; p(AA_1|backbone); p(AA_2|backbone, AA_1) etc, 0 - False",)
    parser.add_argument("--use_sequence", type=int, default=1, help="1 - get scores using amino acid sequence info; 0 - get scores using backbone info only",)

    parser.add_argument("--ligand_mpnn_use_side_chain_context", type=int, default=0,
        help="Flag to use side chain atoms as ligand context for the fixed residues",)
    parser.add_argument("--ligand_mpnn_cutoff_for_score", type=float, default=8.0,
        help="Cutoff in angstroms between protein and context atoms to select residues for reporting score.",)
    parser.add_argument("--ligand_mpnn_use_atom_context", type=int, default=1,
        help="1 - use atom context, 0 - do not use atom context.",)
    parser.add_argument("--parse_atoms_with_zero_occupancy", type=int, default=0, help="To parse atoms with zero occupancy in the PDB input files. 0 - do not parse, 1 - parse atoms with zero occupancy",)

    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for computation, efault is cuda:0')

    args = parser.parse_args()
    sys.path.append(args.lmpnn_loc)
    # from model_utils import tied_featurize, parse_PDB, StructureDatasetPDB
    from model_utils import ProteinMPNN
    from data_utils import (
        element_dict_rev,
        alphabet,
        restype_int_to_str,
        featurize,
        parse_PDB,
    )
    from model_utils import ProteinMPNN

    main(args)
