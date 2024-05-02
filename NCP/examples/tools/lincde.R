# install.packages("devtools")
# devtools::install_github("ZijunGao/LinCDE", build_vignettes = TRUE)

library(LinCDE)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(args[1])
Y <- as.matrix(args[2])
verbose <- as.logical(args[3])

Y_pred <- LinCDE.boost(X = X, y = Y, verbose = F)

write.csv(Y_pred, "/temp/matrix.csv")