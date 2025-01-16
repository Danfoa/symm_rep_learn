#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM


exp_label="eNCP_biases"

exec_file="NCP/examples/symm_GMM.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=6"
opt_params="batch_size=1024 lr=5e-4"
exp_params="exp_label=${exp_label} gmm.n_samples=10000 gmm.n_kernels=30 seed=2,4"

python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=0 model=eNCP truncated_op_bias=svals gamma=0.01,0.001,0.0001 embedding.embedding_dim=5        &
python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=1 model=eNCP truncated_op_bias=svals gamma=0.01,0.001,0.0001 embedding.embedding_dim=10       &
python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=2 model=eNCP truncated_op_bias=svals gamma=0.01,0.001,0.0001 embedding.embedding_dim=25       &
python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=3 model=eNCP truncated_op_bias=svals gamma=0.01,0.001,0.0001 embedding.embedding_dim=50       &
python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=4 model=NCP  truncated_op_bias=svals gamma=0.01,0.001,0.0001 embedding.embedding_dim=5        &
python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=5 model=NCP  truncated_op_bias=svals gamma=0.01,0.001,0.0001 embedding.embedding_dim=10       &
python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=6 model=NCP  truncated_op_bias=svals gamma=0.01,0.001,0.0001 embedding.embedding_dim=25       &
python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=7 model=NCP  truncated_op_bias=svals gamma=0.01,0.001,0.0001 embedding.embedding_dim=50       &
wait
