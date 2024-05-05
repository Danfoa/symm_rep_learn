import os
import pandas as pd

# simple python wrapper around the R implementation of LinCDE -TODO: find relative path for lincde.R or inject direct code

# WARNING: only works for code executed from example folder

def lincde(X, Y, Xtest, ydiscr, verbose='F'):
    # try:
    print(os.getcwd())
    pd.DataFrame(X).to_csv('temp/xtrain.csv', index=False)
    pd.DataFrame(Y).to_csv('temp/ytrain.csv', index=False)
    pd.DataFrame(Xtest).to_csv('temp/xtest.csv', index=False)
    pd.DataFrame(ydiscr).to_csv('temp/ydiscr.csv', index=False)

    os.system(f'Rscript tools/lincde.R {verbose}')
    Y_pred = pd.read_csv("temp/matrix.csv").to_numpy()
    os.remove("temp/xtrain.csv")
    os.remove("temp/ytrain.csv")
    os.remove("temp/xtest.csv")
    os.remove("temp/ydiscr.csv")
    os.remove("temp/matrix.csv")
    return(Y_pred)
    # except:
    #     print("lincde failed, try installing R package")