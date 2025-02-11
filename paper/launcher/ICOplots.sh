#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM


exp_label="IcoPlotsStandard"

exec_file="NCP/examples/symm_GMM.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=4"
hydra_kwargsDRF="--multirun hydra.launcher.n_jobs=1"
opt_params="batch_size=1024 lr=5e-4"
gmm_params="symm_group=ico gmm.n_kernels=5 regular_multiplicity=0 gmm.seed=2 gmm.means_max_norm=1.0"
model_params="gamma=0.5 embedding.hidden_units=64 embedding.hidden_layers=3 embedding.embedding_dim=32"
exp_params="exp_label=${exp_label} ${gmm_params} ${opt_params} ${model_params}"

python ${exec_file} ${hydra_kwargs}    ${exp_params} device=0 model=ENCP train_samples_ratio=0.05,0.1   constraint_out_irreps_dim=False seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=1 model=ENCP train_samples_ratio=0.18,0.25  constraint_out_irreps_dim=False seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=2 model=ENCP train_samples_ratio=0.5        constraint_out_irreps_dim=False seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=3 model=ENCP train_samples_ratio=0.7        constraint_out_irreps_dim=False seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=4 model=ENCP train_samples_ratio=0.05,0.1   constraint_out_irreps_dim=True  seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=5 model=ENCP train_samples_ratio=0.18,0.25  constraint_out_irreps_dim=True  seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=6 model=ENCP train_samples_ratio=0.5        constraint_out_irreps_dim=True  seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs}    ${exp_params} device=7 model=ENCP train_samples_ratio=0.7        constraint_out_irreps_dim=True  seed=0,1,2,3 &

#python ${exec_file} ${hydra_kwargs}    ${exp_params} device=0 model=NCP  train_samples_ratio=0.05,0.1,0.18,0.25,0.5,0.7  seed=0,1,2 &
#python ${exec_file} ${hydra_kwargs}    ${exp_params} device=1 model=ENCP train_samples_ratio=0.05,0.1,0.18,0.25,0.5,0.7  seed=0,1,2 &
#python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=2 model=DRF  train_samples_ratio=0.05,0.1              seed=0,1,2 &
#python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=3 model=DRF  train_samples_ratio=0.25,0.5              seed=0,1,2 &
#python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=4 model=DRF  train_samples_ratio=0.7,0.18              seed=0,1,2 &
#python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=5 model=IDRF train_samples_ratio=0.05,0.1              seed=0,1,2 &
#python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=6 model=IDRF train_samples_ratio=0.25,0.5              seed=0,1,2 &
#python ${exec_file} ${hydra_kwargsDRF} ${exp_params} device=7 model=IDRF train_samples_ratio=0.7,0.18              seed=0,1,2 &

wait
