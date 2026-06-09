# TMB_pipeline

This repository contains code for running a generative pipeline for transmembrane β-barrel (TMB) design. It includes scripts demonstrating how to use TMB_RFD2 for backbone generation and TMB_MPNN for sequence design, as well as notebooks describing the curation and filtering of the distillation set used for training.

---

# Installation and setup

## 1. Clone the repository

git clone https://github.com/baker-laboratory/TMB_pipeline.git

cd TMB_pipeline

---

## 2. Set environment variables and create directories

This section is not strictly required, but the example commands below use these variables.

### Feel free to change directory names

mkdir -p weights environments
export TMB_PIPELINE_DIR="$PWD"
export WEIGHTS_DIR="$TMB_PIPELINE_DIR/weights"
export ENV_DIR="$TMB_PIPELINE_DIR/exec"

---

## 3. Download model weights

Navigate to the weights directory, download checkpoints, and set environment variables.

cd $WEIGHTS_DIR

### TMB_RFD2 checkpoint (standard backbone generation)
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_RFD2/TMB_RFD2.pt

### TMB_RFD2 checkpoint (for motif scaffolding)
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_RFD2/TMB_RFD2_motif_scaffolding.pt

### TMB_MPNN checkpoint (standard sequence design)
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_MPNN/TMB_MPNN.pth

### TMB_MPNN checkpoint (for tuning hydrophobic thickness)
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_MPNN/TMB_MPNN_tuning_hydrophobic_thickness.pth

### Set environment variables
export TMB_RFD2_CKPT_PATH=$WEIGHTS_DIR/TMB_RFD2.pt
export TMB_RFD2_MOTIF_SCAFFOLDING_CKPT_PATH=$WEIGHTS_DIR/TMB_RFD2_motif_scaffolding.pt
export TMB_MPNN_CKPT_PATH=$WEIGHTS_DIR/TMB_MPNN.pth
export TMB_MPNN_COND_CKPT_PATH=$WEIGHTS_DIR/TMB_MPNN_tuning_hydrophobic_thickness.pth

---

## 4. Set up the environment (Singularity / Apptainer)

Download the container and set its path.

cd $ENV_DIR

curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/sif_file/modulus_latest.sif
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/sif_file/TMB_RFD2.sif

### Set container path
export APPTAINER_PATH=$ENV_DIR/modulus_latest.sif

---

## 5. Verify setup (optional but recommended)

echo $RFD2_CKPT_PATH
echo $MPNN_CKPT_PATH
echo $MPNN_COND_CKPT_PATH
echo $APPTAINER_PATH

---

## Notes

- All assets are hosted at:
  https://files.ipd.uw.edu/pub/TMB_pipeline/

- This pipeline assumes access to:
  - a SLURM cluster
  - GPU resources
  - Singularity/Apptainer
