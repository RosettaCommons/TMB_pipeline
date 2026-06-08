#!/bin/bash

# Run from motif_scaffolding_example with:
# singularity exec ../../../ENV_DIR/modulus_latest.sif bash make_motif_diffusion_commands.sh
#
# Then run diffusion from the host with:
# bash inference_commands.txt
#
# Expected layout:
# parent_dir/
# ├── TMB_pipeline/
# │   └── TMB_RFD2/
# │       ├── TMB_RFD2_code/
# │       └── motif_scaffolding_example/
# │           ├── make_motif_diffusion_commands.sh
# │           ├── adj_ss_wrapper_glytwist.py
# │           ├── sample_motif_orientations.py
# │           ├── guidepost_diff_cmd_multiPDBinp.sh
# │           ├── align_barrels_Z.py
# │           └── motif.pdb
# ├── WEIGHTS_DIR/
# └── ENV_DIR/
#     ├── modulus_latest.sif
#     └── bakerlab_rf_diffusion_aa.sif

set -e

# User parameters

N=10
nres=16

# Motif PDB inside motif_scaffolding_example
motif_pdb="./motif.pdb"

GPlen=3 #how many amino acids is the motif (3 in this example)
gly_twists="2 7"
Num_samples=20 #number of orientations of the motif to sample for diffusion

outdir="./motif_diffusion_outputs"
output_file="inference_commands.txt"

# Paths

# Directory where this script lives:
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# TMB_RFD2 root:
TMB_RFD2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Code directory containing rf_diffusion/
RF_CODE_DIR="${TMB_RFD2_ROOT}/TMB_RFD2_code"

# Parent directory containing TMB_pipeline, WEIGHTS_DIR, ENV_DIR:
PROJECT_PARENT="$(cd "${TMB_RFD2_ROOT}/../.." && pwd)"

WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_PARENT}/WEIGHTS_DIR}"
ENV_DIR="${ENV_DIR:-${PROJECT_PARENT}/ENV_DIR}"

DIFFUSION_SIF="${ENV_DIR}/bakerlab_rf_diffusion_aa.sif"

export PYTHONPATH="${RF_CODE_DIR}:${PYTHONPATH}"

# Motif-scaffolding scripts are in this folder
adj_ss_wrapper_script="${TMB_RFD2_ROOT}/adj_ss_wrapper_glytwist.py"
motif_sampling_script="${SCRIPT_DIR}/sample_motif_orientations.py"
guidepost_settings_script="${SCRIPT_DIR}/guidepost_diff_cmd_multiPDBinp.sh"
center_script="${SCRIPT_DIR}/align_barrels_Z.py"

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "TMB_RFD2_ROOT: $TMB_RFD2_ROOT"
echo "RF_CODE_DIR: $RF_CODE_DIR"
echo "PROJECT_PARENT: $PROJECT_PARENT"
echo "WEIGHTS_DIR: $WEIGHTS_DIR"
echo "ENV_DIR: $ENV_DIR"
echo "DIFFUSION_SIF: $DIFFUSION_SIF"

# Sanity checks

cd "$SCRIPT_DIR"

if [ ! -d "$RF_CODE_DIR/rf_diffusion" ]; then
    echo "Error: rf_diffusion directory not found:"
    echo "$RF_CODE_DIR/rf_diffusion"
    exit 1
fi

if [ ! -f "$DIFFUSION_SIF" ]; then
    echo "Error: diffusion SIF not found:"
    echo "$DIFFUSION_SIF"
    exit 1
fi

if [ ! -f "$motif_pdb" ]; then
    echo "Error: motif PDB not found:"
    echo "$motif_pdb"
    exit 1
fi

for script in "$adj_ss_wrapper_script" "$motif_sampling_script" "$guidepost_settings_script" "$center_script"; do
    if [ ! -f "$script" ]; then
        echo "Error: required script not found:"
        echo "$script"
        exit 1
    fi
done

mkdir -p "$outdir"
> "$output_file"

# Derived parameters

motif_basename="$(basename "$motif_pdb")"
protlen=$((N * nres))

adj_ss_output_dir="${SCRIPT_DIR}/adj_ss_glytwist_pt_files_out_dir/n${n_strands}_nres${residues_per_strand}_glycine_twists_${twist_name}"
mkdir -p "$adj_ss_output_dir"

adj_file="${adj_ss_output_dir}/n${N}_nres${nres}_adj.pt"
ss_file="${adj_ss_output_dir}/n${N}_nres${nres}_ss.pt"

# Center motif

echo "Centering motif PDB:"
echo "$motif_pdb"

python "$center_script" "$motif_pdb"

motif_pdb="${SCRIPT_DIR}/alnZ_${motif_basename}"

if [ ! -f "$motif_pdb" ]; then
    echo "Error: centered motif PDB was not created:"
    echo "$motif_pdb"
    exit 1
fi

echo "Centered motif PDB:"
echo "$motif_pdb"

# Generate adjacency and secondary structure .pt files

if [ ! -f "$adj_file" ] || [ ! -f "$ss_file" ]; then
    echo "Generating adjacency and secondary structure files..."

    python "$adj_ss_wrapper_script" \
        --n "$N" \
        --residues_per_strand "$nres" \
        --output_dir "$adj_ss_output_dir"
else
    echo "Adjacency and secondary structure files already exist. Skipping generation."
fi

# Sample motif orientations

echo "Sampling ${Num_samples} random orientations of motif PDB:"
echo "$motif_pdb"

python "$motif_sampling_script" \
    -m "$motif_pdb" \
    -n "$Num_samples"

# Generate glycine twist residue indices

twist_residues=()

for strand in $gly_twists; do
    start=$(( (strand - 1) * nres + nres / 2 - 2 ))
    for offset in {0..4}; do
        twist_residues+=($((start + offset)))
    done
done

glycine_twists_res=$(IFS=,; echo "${twist_residues[*]}")
twist_name="${gly_twists// /_}"

echo "Glycine twist strands: $gly_twists"
echo "Glycine twist residue indices: $glycine_twists_res"

# Generate inference_commands.txt

shopt -s nullglob
sampled_files=(sampled_pose*.pdb)

if [ ${#sampled_files[@]} -eq 0 ]; then
    echo "Error: no sampled_pose*.pdb files found."
    exit 1
fi

for file in "${sampled_files[@]}"; do
    mot_name="$(basename "$file")"

    outpref="${SCRIPT_DIR}/${outdir}/N${N}_nres${nres}_${mot_name%.pdb}_GP_glytwists_${twist_name}"

    cmd="singularity exec --nv ${DIFFUSION_SIF} \
        env PYTHONPATH=${RF_CODE_DIR}:\$PYTHONPATH \
        bash ${guidepost_settings_script} \
        ${SCRIPT_DIR}/${file} \
        ${GPlen} \
        ${protlen} \
        ${outpref} \
        ${adj_ss_output_dir} \
        ${glycine_twists_res}"

    echo "$cmd" >> "$output_file"
done

echo "File generated: ${SCRIPT_DIR}/${output_file}"
echo "To run diffusion, now run:"
echo "cd ${SCRIPT_DIR}"
echo "bash ${output_file}"