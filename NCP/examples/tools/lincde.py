import os
import pandas as pd
import subprocess

# simple python wrapper around the R implementation of LinCDE -TODO: find relative path for lincde.R or inject direct code

def lincde(X, Y, verbose='F'):
    subprocess.call(f'Rscript lincde.R {X} {Y} {verbose}')
    Y_pred = pd.read_csv("temp/matrix.csv")
    os.remove("temp/matrix.csv")
    return(Y_pred.to_numpy())