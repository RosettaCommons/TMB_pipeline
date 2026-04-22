#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=8g
#SBATCH -J TMB_MPNN
#SBATCH -t 05:30:00
#SBATCH -o runmpnn_gpu.out


./make_json_input_longer_hydrophobic_thickness_for_TMB_MPNN.py 40.0 0.25 *.pdb

#first input to make_json_input_longer_hydrophobic_thickness_for_TMB_MPNN.py script is the length in Angstrom of hydrophobic thickness you want to specify, second input is length of interface region. It is calcualte based on length of hydrophibic thickness you specify (in this example interface region on each side will be ~5A on each side of barrel (40x0.25=10A total)

apptainer run --nv ~ENV_DIR/modulus_latest.sif \
python ~/TMB_pipeline/TMB_MPNN/TMB_MPNN_code/v2.1.1/run.py \
    --checkpoint "~WEIGHT_DIR/p_Condition_20250507-dc3e3bb8.pth" \
    --model_type "~/TMB_pipeline/TMB_MPNN/TMB_MPNN_code/v2.1.1/mpnn/configs/big_tmbb_80p_surf_condition.py" \
    --pdb_path_multi "./pdbs_multi_json.jsonl" \
    --out_folder "./TMB_MPNN_sequences" \
    --batch_size 5 \
    --number_of_batches 2 \
    --pack_side_chains 1 \
    --tmbb_surface_condition_multi "./intf_buried.jsonl"
