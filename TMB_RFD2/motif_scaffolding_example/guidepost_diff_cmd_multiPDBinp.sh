#!/bin/bash

set -e

inp_pdb=$1
gplen=$2
protlen=$3
outpref=$4
scaffold_dir=$5
gly_twist=$6

ckpt_path="/home/aphilo/TMB_pipeline/weights/TMB_RFD2_motif_scaffolding.pt"

diffprot=$((protlen - gplen))
gp_res="A1-${gplen}"
contig_len="${protlen}-${protlen}"

# Important: this must be a list containing one quoted string
contigmap_all="['${diffprot}-${diffprot},${gp_res}']"

echo "inp_pdb: $inp_pdb"
echo "gplen: $gplen"
echo "protlen: $protlen"
echo "diffprot: $diffprot"
echo "gp_res: $gp_res"
echo "contig_len: $contig_len"
echo "contigmap_all: $contigmap_all"
echo "gly_twist: $gly_twist"
echo "ckpt_path: $ckpt_path"

if [ ! -f "$inp_pdb" ]; then
    echo "Error: input PDB not found:"
    echo "$inp_pdb"
    exit 1
fi

if [ ! -d "$scaffold_dir" ]; then
    echo "Error: scaffold_dir not found:"
    echo "$scaffold_dir"
    exit 1
fi

if [ ! -f "$ckpt_path" ]; then
    echo "Error: checkpoint not found:"
    echo "$ckpt_path"
    exit 1
fi

python -m rf_diffusion.run_inference --config-name="aa_ppi_TMB_GP_kinks" \
      inference.output_prefix="$outpref" \
      inference.input_pdb="$inp_pdb" \
      inference.ckpt_path="$ckpt_path" \
      contigmap.contigs="$contigmap_all" \
      transforms.configs.CenterPostTransform.center_type=is_diffused \
      inference.custom_t_range=[50,49,48,47,46,45,44,43,42,41,40,38,36,34,32,30,27,23,20,17,14,10,7,4,1] \
      inference.num_designs=20 \
      contigmap.has_termini="[True]" \
      inference.contig_as_guidepost=True \
      upstream_inference_transforms.configs.RenumberCroppedInput.enabled=False \
      contigmap.length="$contig_len" \
      scaffoldguided.scaffold_dir="$scaffold_dir" \
      inference.only_guidepost_positions="$gp_res" \
      ++glycine_twists.glycine_twists_res="'$gly_twist'" \
      +inference.start_self_cond_at_t=42 \
      +filters.names='["ChainBreak","GPSatisfaction","GPEarly:GPSatisfaction","GPMid:GPSatisfaction","GlycineKinkCheck"]' \
      +filters.configs.ChainBreak.t=7 \
      +filters.configs.ChainBreak.verbose=True \
      +filters.configs.ChainBreak.C_N_dist_limit=1.8 \
      +filters.configs.GPSatisfaction.t=1 \
      +filters.configs.GPSatisfaction.verbose=True \
      +filters.configs.GPEarly.t=48 \
      +filters.configs.GPEarly.verbose=True \
      +filters.configs.GPEarly.gp_max_error_cut=9 \
      +filters.configs.GPEarly.gp_rmsd_cut=3 \
      +filters.configs.GPMid.t=36 \
      +filters.configs.GPMid.verbose=True \
      +filters.configs.GPMid.gp_max_error_cut=3 \
      +filters.configs.GPMid.gp_rmsd_cut=1 \
      +filters.configs.GlycineKinkCheck.t=20 \
      +filters.configs.GlycineKinkCheck.verbose=True \
      +filters.configs.GlycineKinkCheck.kink_num_cutoff=4 \
      +filters.steps_per_design=50