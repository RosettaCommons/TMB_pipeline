#!/bin/bash
# after setting your parameters run this script by:
# singularity exec ../../ENV_DIR/sif/modulus_latest.sif bash run_inference.sh
# Set the PYTHONPATH environment variable
export PYTHONPATH=./TMB_pipeline/TMB_RFD2:$PYTHONPATH

# Lists of parameters for adjacency and secondary structure generation
# Define a mapping of number of strands to average number of residues per strand
# value in [] is number of strands, value in "" lists average number of residues per strand you want
declare -A n_strands_residues_map
# n_strands_residues_map[10]="16"
# n_strands_residues_map[12]="16"
n_strands_residues_map[14]="16"
# n_strands_residues_map[16]="16"

# Wrapper script for generating adjacency and secondary structure files
adj_ss_wrapper_script="./adj_ss_wrapper_glykink.py"

# Set checkpoint
models=(
    "../../WEIGTHS_DIR/RFD_TMBB_bulges_twists_masked_withPDB__10.pt"
)

# Static parameters for inference
config_name="aa_ppi"
has_termini="[True]"
contigs="['1-1']"
input_pdb="./dummy.pdb"
center_type="all"

# Output file for commands
output_file="inference_commands.txt"
> "$output_file"  # Clear the file if it exists

# Define a mapping of n_strands values to their respective glycine_twist_combinations
# Specify combinations of strands for glycine twists (specify what strands should have glycine twists)
# ex,: put twists on strand 2 and 7 for 14 stranded designs
# glycine_twist_map[14]="2 7"

declare -A glycine_twist_map
glycine_twist_map[14]="2 7"

# broader sampling example:
# glycine_twist_map[14]="1 2 3 4; 1 2 4 5; 1 2 5 6; 1 2 6 7; 1 2 7 8; 1 2 8 9; 1 2 9 10; 1 2 10 11; 1 2 11 12; 1 2 12 13; 1 2 13 14; 2 3 4 5; 2 3 5 6; 2 3 6 7; 2 3 7 8; 2 3 8 9; 2 3 9 10; 2 3 10 11; 2 3 11 12; 2 3 12 13; 2 3 13 14; 3 4 5 6; 3 4 6 7; 3 4 7 8; 3 4 8 9; 3 4 9 10; 3 4 10 11; 3 4 11 12; 3 4 12 13; 3 4 13 14; 4 5 6 7; 4 5 7 8; 4 5 8 9; 4 5 9 10; 4 5 10 11; 4 5 11 12; 4 5 12 13; 4 5 13 14; 5 6 7 8; 5 6 8 9; 5 6 9 10; 5 6 10 11; 5 6 11 12; 5 6 12 13; 5 6 13 14; 6 7 8 9; 6 7 9 10; 6 7 10 11; 6 7 11 12; 6 7 12 13; 6 7 13 14; 7 8 9 10; 7 8 10 11; 7 8 11 12; 7 8 12 13; 7 8 13 14; 8 9 10 11; 8 9 11 12; 8 9 12 13; 8 9 13 14; 9 10 11 12; 9 10 12 13; 9 10 13 14; 10 11 12 13; 10 11 13 14; 11 12 13 14"

# Loop over n_strands and get the corresponding glycine_twist_combinations
# Iterate over each n_strands value and get its valid residues_per_strand options
for n_strands in "${!n_strands_residues_map[@]}"; do
    IFS=' ' read -ra residues_per_strand_list <<< "${n_strands_residues_map[$n_strands]}"
    for residues_per_strand in "${residues_per_strand_list[@]}"; do

        # Extract glycine twist combinations for this specific n_strands
        IFS=';' read -ra glycine_twist_combinations <<< "${glycine_twist_map[$n_strands]}"
        
        for glycine_twists in "${glycine_twist_combinations[@]}"; do


            # Generate glycine twist residue indices
            twist_residues=()
            for strand in $glycine_twists; do
                start=$(( (strand - 1) * residues_per_strand + residues_per_strand / 2 - 2 ))
                for offset in {0..4}; do
                    twist_residues+=($((start + offset)))
                done
            done
            glycine_twists_res=$(IFS=,; echo "${twist_residues[*]}")

            # Dynamically name the output directory
            adj_ss_output_dir="./glycine_twist_position_sampling_adj_ss_out_dir/n${n_strands}_nres${residues_per_strand}_glycine_twists_${glycine_twists// /_}"

            # Ensure the output directory exists
            if [ ! -d "$adj_ss_output_dir" ]; then
                echo "Directory $adj_ss_output_dir does not exist. Creating it..."
                mkdir -p "$adj_ss_output_dir"
            fi

            # Check if adjacency and secondary structure files exist
            adj_file="${adj_ss_output_dir}/n${n_strands}_nres${residues_per_strand}_adj.pt"
            ss_file="${adj_ss_output_dir}/n${n_strands}_nres${residues_per_strand}_ss.pt"
            twist_file="${adj_ss_output_dir}/n${n_strands}_nres${residues_per_strand}_twist.pt"

            if [ ! -f "$adj_file" ] || [ ! -f "$ss_file" ] || [ ! -f "$twist_file" ]; then
                echo "Generating adjacency, secondary structure, and glycine twist files fo
                n=${n_strands}, residues=${residues_per_strand}, twists=${glycine_twists}..."

                # Use the same Python environment that respects the current PYTHONPATH
                python "$adj_ss_wrapper_script" --n "$n_strands" --residues_per_strand "$residues_per_strand" --twist_indices $glycine_twists --output_dir "$adj_ss_output_dir"
                if [ $? -ne 0 ]; then
                    echo "Error: Failed to generate files. Exiting."
                    exit 1
                fi
            else
                echo "Files already exist for n=${n_strands}, residues=${residues_per_strand}, twists=${glycine_twists}. Skipping generation."
            fi

            # Prepare output and scaffold directories
            output_prefix="./glycine_twist_position_sampling_backbones_out_dir_n18_2twists_in_row_twice_no_bulges_checkpoint/glycine_twists_no_masking_model_n${n_strands}_nres${residues_per_strand}_${glycine_twists// /_}"

            scaffold_dir="$adj_ss_output_dir"

            for model_ckpt in "${models[@]}"; do
                # Create output directory
                model_output_prefix="${output_prefix}/$(basename $model_ckpt .pt)"
                if [ ! -d "$model_output_prefix" ]; then
                    mkdir -p "$model_output_prefix"
                fi

                # Construct the inference command
                cmd="/home/aphilo/TMBB_twists_and_bulges_rf_diffusion/rf_diffusion/run_inference.py \
                    --config-name=\"$config_name\" \
                    contigmap.has_termini=\"$has_termini\" \
                    inference.output_prefix=\"$model_output_prefix/n${n_strands}nres${residues_per_strand}\" \
                    scaffoldguided.scaffold_dir=\"$scaffold_dir\" \
                    contigmap.contigs=\"$contigs\" \
                    inference.ckpt_path=\"$model_ckpt\" \
                    inference.input_pdb=\"$input_pdb\" \
                    inference.num_designs=\"10\" \
                    ++transforms.configs.CenterPostTransform.center_type=\"$center_type\" \
                    ++glycine_twists.glycine_twists_res='\"$glycine_twists_res\"'"

                # Append command to the output file
                echo "$cmd" >> "$output_file"
            done
        done
    done
done

echo "Commands saved to $output_file."
