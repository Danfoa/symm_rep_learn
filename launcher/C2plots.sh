#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM


exp_label="C2plots"

exec_file="NCP/examples/symm_GMM.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=3"
hydra_kwargsDRF="--multirun hydra.launcher.n_jobs=1"
opt_params="batch_size=1024 lr=5e-4 max_epochs=7000 patience=20"
gmm_params="gmm.n_kernels=5,10,50 regular_multiplicity=0 gmm.seed=0 gmm.means_std=3.0"
model_params="gamma=0.1 embedding.hidden_units=32 embedding.hidden_layers=3 embedding.embedding_dim=32"
exp_params="exp_label=${exp_label} ${gmm_params} ${opt_params} ${model_params} seed=0"

python ${exec_file} ${hydra_kwargs}    ${exp_params} device=0 model=NCP  gmm.n_samples=5000,10000    &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=1 model=NCP  gmm.n_samples=20000         &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=2 model=ENCP gmm.n_samples=5000,10000    &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=3 model=ENCP gmm.n_samples=20000         &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=4 model=DRF  gmm.n_samples=5000,10000    &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=5 model=DRF  gmm.n_samples=20000         &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=6 model=IDRF gmm.n_samples=5000,10000    &
python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=7 model=IDRF gmm.n_samples=20000         &

wait
