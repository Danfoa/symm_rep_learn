#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM

robot_name="solo"
exp_name="CoM_learnable_basis_${robot_name}"

exec_file="paper/experiments/CoM_regression.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=2"
opt_params="optim.train_sample_ratio=0.6 optim.lr=1e-3"
model_params="robot_name=${robot_name}"
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 model=NCP  learnable_change_basis=False gamma=1,25  seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 model=NCP  learnable_change_basis=True  gamma=1,25  seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=ENCP learnable_change_basis=False gamma=1,25  seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 model=ENCP learnable_change_basis=False gamma=1,25  seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 model=ENCP learnable_change_basis=True  gamma=1,25  seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=6 model=ENCP learnable_change_basis=True  gamma=1,25  seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 model=EMLP,MLP seed=0,1,2 &
#python ${exec_file} ${hydra_kwargs
wait
