#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM

robot_name="atlas_v4"
exp_name="CoM_${robot_name}_hp"

exec_file="NCP/examples/eNCP_regression.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=3"
opt_params="optim.train_sample_ratio=0.5"
model_params=""
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 optim.lr=5e-4 model=NCP architecture.residual_encoder=False,True   gamma=1,5,10     seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 optim.lr=5e-4 model=NCP architecture.residual_encoder=False,True   gamma=25,50      seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 optim.lr=5e-4 model=NCP architecture.residual_encoder=False,True   gamma=100,200    seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 optim.lr=5e-4 model=ENCP gamma=1,5,10     seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 optim.lr=5e-4 model=ENCP gamma=25,50      seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 optim.lr=5e-4 model=ENCP gamma=100,200    seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=6 optim.lr=5e-4 model=ENCP gamma=500        seed=0,1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 optim.lr=5e-4 model=EMLP,MLP  seed=0 &
#python ${exec_file} ${hydra_kwargs
wait
