#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM


exp_label="C3plots_final_final"

exec_file="NCP/examples/symm_GMM.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=4"
hydra_kwargsDRF="--multirun hydra.launcher.n_jobs=2"
opt_params="batch_size=1024 lr=5e-4"
gmm_params="symm_group=C3 gmm.n_kernels=2,3,7 regular_multiplicity=1 gmm.seed=0 gmm.means_std=3.0 gmm.n_total_samples=30000"
model_params="gamma=0.5 embedding.hidden_units=32 embedding.hidden_layers=3 embedding.embedding_dim=32"
exp_params="exp_label=${exp_label} ${gmm_params} ${opt_params} ${model_params}"

python ${exec_file} ${hydra_kwargs}    ${exp_params} device=6 model=NCP  train_samples_ratio=0.1,0.7   seed=0,1,2 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=7 model=NCP  train_samples_ratio=0.25,0.5  seed=0,1,2 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=2 model=ENCP train_samples_ratio=0.1,0.7   seed=0,1,2 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=3 model=ENCP train_samples_ratio=0.25,0.5  seed=0,1,2 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=4 model=DRF  train_samples_ratio=0.1,0.7   seed=0,1,2 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=5 model=DRF  train_samples_ratio=0.25,0.5  seed=0,1,2 &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=0 model=IDRF train_samples_ratio=0.1,0.7   seed=0,1,2 &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=1 model=IDRF train_samples_ratio=0.25,0.5  seed=0,1,2 &

wait
