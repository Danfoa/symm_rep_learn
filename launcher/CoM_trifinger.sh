#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM


exp_name="CoM_sample_eff3"

exec_file="NCP/examples/eNCP_regression.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=2"
opt_params="optim.patience=10"
model_params=""
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 optim.lr=5e-4 optim.train_sample_ratio=0.025,0.05,0.7 model=NCP  gamma=25 seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 optim.lr=5e-4 optim.train_sample_ratio=0.075,0.1,0.6 model=NCP   gamma=25 seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 optim.lr=5e-4 optim.train_sample_ratio=0.18,0.25,0.5 model=NCP   gamma=25 seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 optim.lr=5e-4 optim.train_sample_ratio=0.025,0.05,0.7 model=ENCP gamma=25 seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 optim.lr=5e-4 optim.train_sample_ratio=0.075,0.1,0.6  model=ENCP gamma=25 seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 optim.lr=5e-4 optim.train_sample_ratio=0.18,0.25,0.5  model=ENCP gamma=25 seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=6 optim.lr=5e-4 optim.train_sample_ratio=0.025,0.05,0.075,0.1,0.18,0.25,0.5,0.6,0.7 model=MLP  seed=0,1,2,3 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 optim.lr=5e-4 optim.train_sample_ratio=0.025,0.05,0.075,0.1,0.18,0.25,0.5,0.6,0.7 model=EMLP seed=0,1,2,3 &
#python ${exec_file} ${hydra_kwargs
wait
