#!/usr/local/bin/Rscript
library(BayesTree)

## Read in directory
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]

# Target FDR level
alpha = 0.1

# Load the data
X = read.csv(file.path(paste(data_dir, "X.csv", sep="/")), header=TRUE, sep=',')
Y = read.csv(file.path(paste(data_dir, "Y.csv", sep="/")), header=TRUE, sep=',')[,1]
T_ = read.csv(file.path(paste(data_dir, "T.csv", sep="/")), header=TRUE, sep=',')[,1]
H = read.csv(file.path(paste(data_dir, "H.csv", sep="/")), header=TRUE, sep=',')[,1]

n.treated = sum(T_)

xt = as.matrix(cbind(X, T_)) # All observations
#yt = as.numeric(unlist(Y["Y"]))
yt = as.numeric(Y)
summary(xt)
summary(yt)

# Counterfactual question, asking for outcomes without
# treatment for treated individuals
xp=xt[xt[,"T_"]==1,]
xp[,ncol(xt)]=0

# Run BART on all the data
bart.tot = bart(x.train=xt, y.train=yt, x.test=xp)

# Get the treatment predictions
mndiffs = bart.tot$yhat.train[,T_==1] - bart.tot$yhat.test

# Perform a 1-sided Bayesian credible interval test (small = reject)
fdr.local = apply(mndiffs < 0, 2, mean)

# FDRs
alphas = seq(0, 1, length.out = 1000)

## Reorder so most significant first
ord = order(fdr.local, decreasing=FALSE)

## Reorder fdr values, take running average
fdr.local = fdr.local[ord]
fdr.local = cumsum(fdr.local)/seq(1,n.treated)

## Subset and reorder hidden variables
H_treated = H[T_==1]
H_treated = H_treated[ord]

## Total positive
n.total.pos = sum(H_treated)

## Step over alpha values, tracking true/false positive rate
num.sel <- integer(length(alphas))
num.pos <- integer(length(alphas))
for (i in 1:length(alphas)){
	alpha = alphas[i]
    mask = fdr.local<=alpha
    num.sel[i] = sum(mask)
    num.pos[i] = sum(H_treated[mask])
}

fdr = (num.sel-num.pos)/num.sel
frac.discoveries = num.pos/n.total.pos

## Package up as a dataframe
df = data.frame("Nominal FDR"=alphas, "# selected"=num.sel, "# positive"=num.pos, "False discovery rate"=fdr, "Fraction of discoveries made"=frac.discoveries)

write.csv(df, paste(data_dir, "bart.csv", sep="/"))


df_raw = data.frame("q_value"= fdr.local, "H"= H_treated)
write.csv(df_raw, paste(data_dir, "bart_raw.csv", sep="/"))


