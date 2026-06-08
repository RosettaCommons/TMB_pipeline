import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_paired_and_masked_boxes(n, residues_per_strand):
    """
    Generate an adjacency matrix with paired boxes.
    """
    total_length = n * residues_per_strand
    adj_matrix = np.zeros((total_length, total_length))

    for i in range(n):
        start = i * residues_per_strand
        end = start + residues_per_strand
        center_row = start + residues_per_strand // 2
        center_col_left = start - residues_per_strand + residues_per_strand // 2
        center_col_right = end + residues_per_strand // 2

        if start - residues_per_strand >= 0:
            adj_matrix[start:end, start - residues_per_strand:start] = 2
            adj_matrix[center_row - 1:center_row + 1, center_col_left - 1:center_col_left + 1] = 3

        if end + residues_per_strand <= total_length:
            adj_matrix[start:end, end:end + residues_per_strand] = 2
            adj_matrix[center_row - 1:center_row + 1, center_col_right - 1:center_col_right + 1] = 3

    bottom_start = total_length - residues_per_strand
    adj_matrix[bottom_start:, :residues_per_strand] = 2
    bottom_center_row = total_length - residues_per_strand // 2
    bottom_center_col = residues_per_strand // 2
    adj_matrix[bottom_center_row - 1:bottom_center_row + 1, bottom_center_col - 1:bottom_center_col + 1] = 3

    adj_matrix[:residues_per_strand, -residues_per_strand:] = 2
    top_center_row = residues_per_strand // 2
    top_center_col = total_length - residues_per_strand // 2
    adj_matrix[top_center_row - 1:top_center_row + 1, top_center_col - 1:top_center_col + 1] = 3

    return torch.tensor(adj_matrix)

def generate_flat_masked_ss_tensor(n, residues_per_strand):
    """
    Generate a flat ss tensor with masked regions.
    """
    total_length = n * residues_per_strand
    ss_tensor = np.full(total_length, 3.0)

    for i in range(n):
        start = i * residues_per_strand
        center_start = start + residues_per_strand // 2 - 1
        center_end = center_start + 2
        ss_tensor[center_start:center_end] = 1.0

    return torch.tensor(ss_tensor)

# Assuming generate_paired_and_masked_boxes() and generate_flat_masked_ss_tensor() exist

def generate_glycine_twist_positions(n, residues_per_strand, twist_indices):
    """
    Generate positions for glycine twists based on user-defined indices.
    """
    total_length = n * residues_per_strand
    twist_positions = np.zeros(total_length)

    for twist_idx in twist_indices:
        strand_start = (twist_idx - 1) * residues_per_strand
        twist_center = strand_start + residues_per_strand // 2
        twist_positions[twist_center - 2:twist_center + 3] = 1  # Mark 5 residues for the twist

    return torch.tensor(twist_positions)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate adjacency and secondary structure tensors.")
    parser.add_argument("--n", type=int, required=True, help="Number of strands.")
    parser.add_argument("--residues_per_strand", type=int, required=True, help="Number of residues per strand.")
    parser.add_argument("--twist_indices", nargs='+', type=int, help="Indices of strands for glycine twists.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated files.")
    args = parser.parse_args()

    # Generate adjacency and ss tensors
    n = args.n
    residues_per_strand = args.residues_per_strand
    twist_indices = args.twist_indices
    output_dir = args.output_dir

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate tensors
    adj_matrix = generate_paired_and_masked_boxes(n, residues_per_strand)
    ss_tensor = generate_flat_masked_ss_tensor(n, residues_per_strand)

    twist_tensor = None
    if twist_indices:
        twist_tensor = generate_glycine_twist_positions(n, residues_per_strand, twist_indices)

    # Save tensors
    adj_file = os.path.join(output_dir, f"n{n}_nres{residues_per_strand}_adj.pt")
    ss_file = os.path.join(output_dir, f"n{n}_nres{residues_per_strand}_ss.pt")
    twist_file = os.path.join(output_dir, f"n{n}_nres{residues_per_strand}_twist.pt") if twist_tensor is not None else None

    torch.save(adj_matrix, adj_file)
    torch.save(ss_tensor, ss_file)
    if twist_tensor is not None:
        torch.save(twist_tensor, twist_file)

    print(f"Adjacency matrix saved to: {adj_file}")
    print(f"Secondary structure tensor saved to: {ss_file}")
    if twist_tensor is not None:
        print(f"Glycine twist tensor saved to: {twist_file}")

if __name__ == "__main__":
    main()