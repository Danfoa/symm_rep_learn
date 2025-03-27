#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM

robot_name="cassie"
exp_name="CoM_hp_${robot_name}"

exec_file="paper/experiments/CoM_regression.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=2"
opt_params="optim.train_sample_ratio=0.6 optim.lr=5e-4 optim.max_epochs=50 architecture.residual_encoder=True lstsq=True"
model_params="robot_name=${robot_name}"
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 model=NCP   gamma=1  seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 model=NCP   gamma=10 seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 model=NCP   gamma=25 seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=ENCP  gamma=1  seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 model=ENCP  gamma=10 seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 model=ENCP  gamma=25 seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=6 model=EMLP           seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 model=MLP            seed=0,1 &
#python ${exec_file} ${hydra_kwargs
wait
