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
exp_name="CoM_sample_efficiency_${robot_name}"

exec_file="paper/experiments/CoM_regression.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=2"
opt_params=""
model_params="robot_name=${robot_name}"
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 optim.lr=5e-4      optim.train_sample_ratio=0.01,0.70  model=NCP  gamma=40 seed=0,1,2 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 optim.lr=5e-4      optim.train_sample_ratio=0.10,0.20  model=NCP  gamma=40 seed=0,1,2 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 optim.lr=5e-4      optim.train_sample_ratio=0.30,0.50  model=NCP  gamma=40 seed=0,1,2 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 optim.lr=5e-4      optim.train_sample_ratio=0.01,0.70  model=ENCP gamma=1  seed=0,1,2 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 optim.lr=5e-4      optim.train_sample_ratio=0.10,0.20  model=ENCP gamma=1  seed=0,1,2 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 optim.lr=5e-4      optim.train_sample_ratio=0.30       model=ENCP gamma=1  seed=0,1,2 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 optim.lr=5e-4      optim.train_sample_ratio=0.50       model=ENCP gamma=1  seed=0,1,2 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 optim.lr=1e-3,5e-4 optim.train_sample_ratio=0.01,0.1,0.2,0.3,0.5,0.7 model=EMLP,MLP seed=0,1,2 &
#python ${exec_file} ${hydra_kwargs
wait
