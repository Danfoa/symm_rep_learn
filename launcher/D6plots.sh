#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM


exp_label="D6MI"

exec_file="NCP/examples/symm_GMM.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=6"
hydra_kwargsDRF="--multirun hydra.launcher.n_jobs=1"
opt_params="batch_size=1024 lr=5e-4"
gmm_params="symm_group=D6 gmm.n_kernels=10 regular_multiplicity=1 gmm.seed=307 gmm.means_max_norm=1.0 gmm.n_total_samples=50000"
model_params="gamma=0.5 embedding.hidden_units=64 embedding.hidden_layers=3 embedding.embedding_dim=32"
exp_params="exp_label=${exp_label} ${gmm_params} ${opt_params} ${model_params}"

python ${exec_file} ${hydra_kwargs}    ${exp_params} device=0 model=NCP  train_samples_ratio=0.05,0.1,0.18,0.25,0.5,0.7  seed=0,1,2 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=1 model=ENCP train_samples_ratio=0.05,0.1,0.18,0.25,0.5,0.7  seed=0,1,2 &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=2 model=DRF  train_samples_ratio=0.05,0.1              seed=0,1,2 &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=3 model=DRF  train_samples_ratio=0.25,0.5              seed=0,1,2 &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=4 model=DRF  train_samples_ratio=0.7,0.18              seed=0,1,2 &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=5 model=IDRF train_samples_ratio=0.05,0.1              seed=0,1,2 &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=6 model=IDRF train_samples_ratio=0.25,0.5              seed=0,1,2 &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=7 model=IDRF train_samples_ratio=0.7,0.18              seed=0,1,2 &

wait
