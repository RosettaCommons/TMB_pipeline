# TMB_pipeline

This repository contains tutorials for running a fully deep learning–based pipeline for transmembrane β-barrel (TMB) design. It includes notebooks demonstrating how to use TMB_RFD2 for backbone generation and TMB_MPNN for sequence design, as well as notebooks describing the curation and filtering of the distillation set used for training.

---

# Installation and setup

## 1. Clone the repository

git clone https://github.com//baker-laboratory/TMB_pipeline.git

cd TMB_pipeline

---

## 2. Set environment variables and create directories

This section is not strictly required, but the example commands below use these variables.

### Feel free to change directory names

TMB_PIPELINE_DIR=/path/to/TMB_pipeline
WEIGHTS_DIR=$TMB_PIPELINE_DIR/weights
ENV_DIR=$TMB_PIPELINE_DIR/exec

### Create directories
mkdir -p $TMB_PIPELINE_DIR
mkdir -p $WEIGHTS_DIR
mkdir -p $ENV_DIR

---

## 3. Download model weights

Navigate to the weights directory, download checkpoints, and set environment variables.

cd $WEIGHTS_DIR

### RFD2 checkpoint (backbone generation)
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_RFD2/RFD_TMBB_bulges_twists_masked_withPDB__10.pt

### MPNN checkpoint (standard sequence design)
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_MPNN/nosmoothmask_surfcond.pth

### MPNN checkpoint (conditioned on hydrophobic thickness model)
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_MPNN/p_Condition_20250507-dc3e3bb8.pth

### Set environment variables
export RFD2_CKPT_PATH=$WEIGHTS_DIR/RFD_TMBB_bulges_twists_masked_withPDB__10.pt
export MPNN_CKPT_PATH=$WEIGHTS_DIR/nosmoothmask_surfcond.pth
export MPNN_COND_CKPT_PATH=$WEIGHTS_DIR/p_Condition_20250507-dc3e3bb8.pth

---

## 4. Set up the environment (Singularity / Apptainer)

Download the container and set its path.

cd $ENV_DIR

curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/sif_file/modulus_latest.sif

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
