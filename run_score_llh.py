import subprocess
import torch
import os

def calculate_log_likelihood(pdb_path, out_folder, model_type="ligand_mpnn", number_of_batches=10):
    """
    Calculate the average log-likelihood for a protein sequence in a PDB file using LigandMPNN.
    
    Args:
        pdb_path (str): Path to the PDB file containing the variant sequence.
        out_folder (str): Directory to save the output scores.
        model_type (str): Model type, default is 'ligand_mpnn'.
        number_of_batches (int): Number of batches for scoring (default: 10).
    
    Returns:
        float: Average log-likelihood across batches.
    """
    # Ensure output folder exists
    os.makedirs(out_folder, exist_ok=True)
    
    # Run the score.py script
    command = (
        f"python score.py "
        f"--model_type '{model_type}' "
        f"--seed 111 "
        f"--autoregressive_score 1 "
        f"--pdb_path '{pdb_path}' "
        f"--out_folder '{out_folder}' "
        f"--use_sequence 1 "
        f"--batch_size 1 "
        f"--number_of_batches {number_of_batches}"
    )
    # subprocess.run(command, shell=True, check=True)
    
    # Load the output dictionary
    output_file = os.path.join(out_folder, "2gi9.pt")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file {output_file} not found. Check if score.py ran successfully.")
    
    output = torch.load(output_file)
    log_probs = output['log_probs']  # Shape: [number_of_batches, length]
    print(type(log_probs))
    
    # Compute total log-likelihood for each batch
    log_likelihoods = log_probs.sum(axis=1)  # Shape: [number_of_batches]
    
    
    return log_likelihoods

if __name__ == "__main__":
    # Example usage
    variant_pdb = "./example/GB1/2gi9.pdb"  # Replace with your PDB file path
    out_folder = "example/GB1/lmpnn_llh"
    
    try:
        log_likelihood = calculate_log_likelihood(variant_pdb, out_folder)
        print(f"The average log-likelihood for the variant is: {log_likelihood}")
    except Exception as e:
        print(f"Error: {e}")