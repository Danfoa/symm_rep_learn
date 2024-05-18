import os
import pandas as pd

# simple python wrapper around the R implementation of LinCDE -TODO: find relative path for lincde.R or inject direct code

# WARNING: only works for code executed from example folder

def lincde(X, Y, Xtest, ydiscr, verbose='F', folder_location=''):
    # try:
    print(os.getcwd())
    pd.DataFrame(X).to_csv(folder_location+'temp/xtrain.csv', index=False)
    pd.DataFrame(Y).to_csv(folder_location+'temp/ytrain.csv', index=False)
    pd.DataFrame(Xtest).to_csv(folder_location+'temp/xtest.csv', index=False)
    pd.DataFrame(ydiscr).to_csv(folder_location+'temp/ydiscr.csv', index=False)

    os.system(f'Rscript tools/lincde.R {verbose}')
    Y_pred = pd.read_csv(folder_location+"temp/pdf.csv").to_numpy()
    Y_discr = pd.read_csv(folder_location+"temp/ys.csv").to_numpy()
    os.remove(folder_location+"temp/xtrain.csv")
    os.remove(folder_location+"temp/ytrain.csv")
    os.remove(folder_location+"temp/xtest.csv")
    os.remove(folder_location+"temp/ydiscr.csv")
    os.remove(folder_location+"temp/matrix.csv")
    return(Y_discr, Y_pred)
    # except:
    #     print("lincde failed, try installing R package")