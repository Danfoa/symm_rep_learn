#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM


exp_label="full_rank"

exec_file="NCP/examples/symm_GMM.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=6"
opt_params="batch_size=1024 lr=5e-4,1e-3 max_epochs=5000"
exp_params="exp_label=${exp_label} gmm.n_samples=50000  seed=5,4 embedding.hidden_units=32,64 regular_multiplicitiy=0"
gamma_ncp="gamma=0.01,0.001"
#python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=0 model=NCP truncated_op_bias=Cxy       embedding.embedding_dim=5,10 ${gamma_ncp}  &
#python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=1 model=NCP truncated_op_bias=full_rank embedding.embedding_dim=5,10 ${gamma_ncp}  &
#python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=2 model=NCP truncated_op_bias=svals     embedding.embedding_dim=5,10 ${gamma_ncp}  &
#python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=3 model=NCP truncated_op_bias=diag      embedding.embedding_dim=5,10 ${gamma_ncp}  &
#python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=4 model=NCP truncated_op_bias=    ${gamma_ncp}  &
#python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=6 model=NCP truncated_op_bias=    ${gamma_ncp}  &
python ${exec_file} ${hydra_kwargs} ${opt_params} ${exp_params} device=7 model=DRF ${gamma_ncp}                &
wait
