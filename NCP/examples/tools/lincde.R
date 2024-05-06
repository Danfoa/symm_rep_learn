# comment after first use

# options(repos = list(CRAN="http://cran.rstudio.com/"))
# install.packages("devtools")
# devtools::install_github("ZijunGao/LinCDE", build_vignettes = TRUE)

library(LinCDE)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args <- commandArgs(trailingOnly = TRUE)
X <- read.csv("temp/xtrain.csv")
Y <- as.vector(t(read.csv("temp/ytrain.csv")))
xtest <- read.csv("temp/xtest.csv")
ydiscr <- as.vector(read.csv("temp/ydiscr.csv"))
verbose <- as.logical(args[1])

model = LinCDE.boost(X = X, y = Y, verbose = F)

prediction <- predict(object=model, X=xtest, Y=ydiscr)
estDens <- prediction$cellProb
ys <- prediction$yDiscretized

write.csv(estDens, paste(getwd(), "/temp/pdf.csv", sep=''), row.names = FALSE)
write.csv(ys, paste(getwd(), "/temp/ys.csv", sep=''), row.names = FALSE)