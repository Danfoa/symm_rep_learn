import os
import pandas as pd

# simple python wrapper around the R implementation of LinCDE -TODO: find relative path for lincde.R or inject direct code

# WARNING: only works for code executed from example folder

def lincde(X, Y, verbose='F'):
    print(os.getcwd())
    os.system(f'Rscript tools/lincde.R {X} {Y} {verbose}')
    Y_pred = pd.read_csv("temp/matrix.csv")
    os.remove("temp/matrix.csv")
    return(Y_pred.to_numpy())