#!/bin/bash

# Run from the TMB_RFD2 directory with:
# apptainer exec "$MODULUS_APPTAINER_PATH" bash make_diffusion_commands.sh
#
# Or, if paths are not exported:
# apptainer exec ../environments/modulus_latest.sif bash make_diffusion_commands.sh
#
# Then run diffusion separately from the host with:
# bash inference_commands.txt
#
# Expected directory layout:
# TMB_pipeline/
# ├── TMB_RFD2/
# │   ├── make_diffusion_commands.sh
# │   ├── run_inference.sh
# │   ├── adj_ss_wrapper_glytwist.py
# │   ├── dummy.pdb
# │   └── TMB_RFD2_code/
# │       └── rf_diffusion/
# ├── weights/
# │   └── TMB_RFD2.pt
# └── environments/
#     ├── modulus_latest.sif
#     └── TMB_RFD2.sif

set -e

############################################
# Paths
############################################

# Directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# TMB_RFD2 directory
REPO_ROOT="$SCRIPT_DIR"

# TMB_pipeline directory
TMB_PIPELINE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"

# Code directory containing rf_diffusion/
RF_CODE_DIR="${REPO_ROOT}/TMB_RFD2_code"

# Default locations, override if needed:
# export WEIGHTS_DIR=/path/to/weights
# export ENV_DIR=/path/to/environments
WEIGHTS_DIR="${WEIGHTS_DIR:-${TMB_PIPELINE_DIR}/weights}"
ENV_DIR="${ENV_DIR:-${TMB_PIPELINE_DIR}/environments}"

# Containers
MODULUS_SIF="${MODULUS_APPTAINER_PATH:-${ENV_DIR}/modulus_latest.sif}"
DIFFUSION_SIF="${TMB_RFD2_APPTAINER_PATH:-${ENV_DIR}/TMB_RFD2.sif}"

# Python path for local RFdiffusion code
export PYTHONPATH="${RF_CODE_DIR}:${PYTHONPATH}"

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "REPO_ROOT: $REPO_ROOT"
echo "TMB_PIPELINE_DIR: $TMB_PIPELINE_DIR"
echo "RF_CODE_DIR: $RF_CODE_DIR"
echo "WEIGHTS_DIR: $WEIGHTS_DIR"
echo "ENV_DIR: $ENV_DIR"
echo "MODULUS_SIF: $MODULUS_SIF"
echo "DIFFUSION_SIF: $DIFFUSION_SIF"

############################################
# Sanity checks
############################################

if [ ! -d "$RF_CODE_DIR/rf_diffusion" ]; then
    echo "Error: rf_diffusion directory not found:"
    echo "$RF_CODE_DIR/rf_diffusion"
    exit 1
fi

if [ ! -f "$MODULUS_SIF" ]; then
    echo "Error: modulus SIF not found:"
    echo "$MODULUS_SIF"
    exit 1
fi

if [ ! -f "$DIFFUSION_SIF" ]; then
    echo "Error: diffusion SIF not found:"
    echo "$DIFFUSION_SIF"
    exit 1
fi

############################################
# User parameters
############################################

declare -A n_strands_residues_map

# n_strands_residues_map[10]="16"
# n_strands_residues_map[12]="16"
n_strands_residues_map[14]="16"
# n_strands_residues_map[16]="16"

# Wrapper script for generating adjacency/SS/glycine twist files
adj_ss_wrapper_script="${REPO_ROOT}/adj_ss_wrapper_glytwist.py"

if [ ! -f "$adj_ss_wrapper_script" ]; then
    echo "Error: adjacency/SS wrapper script not found:"
    echo "$adj_ss_wrapper_script"
    exit 1
fi

############################################
# Checkpoint models
############################################

models=(
    "${WEIGHTS_DIR}/TMB_RFD2.pt"
)

for model_ckpt in "${models[@]}"; do
    if [ ! -f "$model_ckpt" ]; then
        echo "Error: checkpoint not found:"
        echo "$model_ckpt"
        echo "Expected weights dir:"
        echo "$WEIGHTS_DIR"
        exit 1
    fi
done

############################################
# Static parameters for inference
############################################

config_name="aa_ppi"
has_termini="[True]"
contigs="['1-1']"
input_pdb="${REPO_ROOT}/dummy.pdb"
center_type="all"
num_designs="10"

if [ ! -f "$input_pdb" ]; then
    echo "Error: input PDB not found:"
    echo "$input_pdb"
    exit 1
fi

# Output file for generated commands
output_file="${REPO_ROOT}/inference_commands.txt"
> "$output_file"

############################################
# Glycine twist combinations
############################################

declare -A glycine_twist_map

# Example: put glycine twists on strands 2 and 7 for 14-stranded designs
glycine_twist_map[14]="2 7"

# Broader sampling example:
# glycine_twist_map[14]="1 2 3 4; 1 2 4 5; 1 2 5 6; 1 2 6 7; 1 2 7 8; 1 2 8 9; 1 2 9 10; 1 2 10 11; 1 2 11 12; 1 2 12 13; 1 2 13 14; 2 3 4 5; 2 3 5 6; 2 3 6 7; 2 3 7 8; 2 3 8 9; 2 3 9 10; 2 3 10 11; 2 3 11 12; 2 3 12 13; 2 3 13 14; 3 4 5 6; 3 4 6 7; 3 4 7 8; 3 4 8 9; 3 4 9 10; 3 4 10 11; 3 4 11 12; 3 4 12 13; 3 4 13 14; 4 5 6 7; 4 5 7 8; 4 5 8 9; 4 5 9 10; 4 5 10 11; 4 5 11 12; 4 5 12 13; 4 5 13 14; 5 6 7 8; 5 6 8 9; 5 6 9 10; 5 6 10 11; 5 6 11 12; 5 6 12 13; 5 6 13 14; 6 7 8 9; 6 7 9 10; 6 7 10 11; 6 7 11 12; 6 7 12 13; 6 7 13 14; 7 8 9 10; 7 8 10 11; 7 8 11 12; 7 8 12 13; 7 8 13 14; 8 9 10 11; 8 9 11 12; 8 9 12 13; 8 9 13 14; 9 10 11 12; 9 10 12 13; 9 10 13 14; 10 11 12 13; 10 11 13 14; 11 12 13 14"

############################################
# Move into repo root
############################################

cd "$REPO_ROOT"

############################################
# Generate adj/ss/glytwist files and inference_commands.txt
############################################

for n_strands in "${!n_strands_residues_map[@]}"; do
    IFS=' ' read -ra residues_per_strand_list <<< "${n_strands_residues_map[$n_strands]}"

    for residues_per_strand in "${residues_per_strand_list[@]}"; do
        IFS=';' read -ra glycine_twist_combinations <<< "${glycine_twist_map[$n_strands]}"

        for glycine_twists in "${glycine_twist_combinations[@]}"; do

            glycine_twists="$(echo "$glycine_twists" | xargs)"

            if [ -z "$glycine_twists" ]; then
                echo "Warning: no glycine twists specified for n=${n_strands}. Skipping."
                continue
            fi

            # Generate glycine twist residue indices
            twist_residues=()

            for strand in $glycine_twists; do
                start=$(( (strand - 1) * residues_per_strand + residues_per_strand / 2 - 2 ))

                for offset in {0..4}; do
                    twist_residues+=($((start + offset)))
                done
            done

            glycine_twists_res=$(IFS=,; echo "${twist_residues[*]}")
            twist_name="${glycine_twists// /_}"

            adj_ss_output_dir="${REPO_ROOT}/adj_ss_glytwist_pt_files_out_dir/n${n_strands}_nres${residues_per_strand}_glycine_twists_${twist_name}"

            mkdir -p "$adj_ss_output_dir"

            adj_file="${adj_ss_output_dir}/n${n_strands}_nres${residues_per_strand}_adj.pt"
            ss_file="${adj_ss_output_dir}/n${n_strands}_nres${residues_per_strand}_ss.pt"
            twist_file="${adj_ss_output_dir}/n${n_strands}_nres${residues_per_strand}_twist.pt"

            if [ ! -f "$adj_file" ] || [ ! -f "$ss_file" ] || [ ! -f "$twist_file" ]; then
                echo "Generating adjacency, secondary structure, and glycine twist files for n=${n_strands}, residues=${residues_per_strand}, twists=${glycine_twists}..."

                python "$adj_ss_wrapper_script" \
                    --n "$n_strands" \
                    --residues_per_strand "$residues_per_strand" \
                    --twist_indices $glycine_twists \
                    --output_dir "$adj_ss_output_dir"
            else
                echo "Files already exist for n=${n_strands}, residues=${residues_per_strand}, twists=${glycine_twists}. Skipping generation."
            fi

            output_prefix="${REPO_ROOT}/glycine_twist_position_sampling_backbones_out_dir_n${n_strands}/glycine_twists_no_masking_model_n${n_strands}_nres${residues_per_strand}_${twist_name}"
            scaffold_dir="$adj_ss_output_dir"

            for model_ckpt in "${models[@]}"; do
                model_output_prefix="${output_prefix}/$(basename "$model_ckpt" .pt)"
                mkdir -p "$model_output_prefix"

                cmd="singularity exec --nv ${DIFFUSION_SIF} \
                    env PYTHONPATH=${RF_CODE_DIR}:\$PYTHONPATH \
                    python ${RF_CODE_DIR}/rf_diffusion/run_inference.py \
                    --config-name=\"$config_name\" \
                    contigmap.has_termini=\"$has_termini\" \
                    inference.output_prefix=\"$model_output_prefix/n${n_strands}nres${residues_per_strand}\" \
                    scaffoldguided.scaffold_dir=\"$scaffold_dir\" \
                    contigmap.contigs=\"$contigs\" \
                    inference.ckpt_path=\"$model_ckpt\" \
                    inference.input_pdb=\"$input_pdb\" \
                    inference.num_designs=\"$num_designs\" \
                    ++transforms.configs.CenterPostTransform.center_type=\"$center_type\" \
                    ++glycine_twists.glycine_twists_res='\"$glycine_twists_res\"'"

                echo "$cmd" >> "$output_file"
            done
        done
    done
done

echo "Commands saved to $output_file."
echo "To run diffusion, now run:"
echo "bash $output_file"