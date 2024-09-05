#!/usr/local/bin/Rscript
library(grf)

# Target FDR level
alpha = 0.1

## Read in directory
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]

# Load the data
X = read.csv(file.path(paste(data_dir, "X.csv", sep="/")), header=TRUE, sep=',')
Y = read.csv(file.path(paste(data_dir, "Y.csv", sep="/")), header=TRUE, sep=',')[,1]
T_ = read.csv(file.path(paste(data_dir, "T.csv", sep="/")), header=TRUE, sep=',')[,1]
H = read.csv(file.path(paste(data_dir, "H.csv", sep="/")), header=TRUE, sep=',')[,1]

n.treated = sum(T_)

# Run the causal forests with confidence intervals
tau.forest = causal_forest(X, Y, T_, num.trees = 4000, tune.parameters="all")

# Estimate causal effects for the training data using out-of-bag prediction.
tau.hat = predict(tau.forest, estimate.variance = TRUE)
sigma.hat = sqrt(tau.hat$variance.estimates)

## Calculate p.values
p.vals = pnorm(0, mean=tau.hat$predictions, sd=sigma.hat)

## Subset to treated and calculated BH correction
p.vals = p.vals[T_==1]
q.vals = p.adjust(p.vals, method="BH")

# FDRs
alphas = seq(0, 1, length.out = 1000)


## Subset hidden variables to treated
H.treated = H[T_==1]

## Total positive
n.total.pos = sum(H.treated)

## Step over alpha values, tracking true/false positive rate
num.sel <- integer(length(alphas))
num.pos <- integer(length(alphas))
for (i in 1:length(alphas)){
	alpha = alphas[i]
    mask = q.vals<=alpha
    num.sel[i] = sum(mask)
    num.pos[i] = sum(H.treated[mask])
}

fdr = (num.sel-num.pos)/num.sel
frac.discoveries = num.pos/n.total.pos

## Package up as a dataframe
df = data.frame("Nominal FDR"=alphas, "# selected"=num.sel, "# positive"=num.pos, "False discovery rate"=fdr, "Fraction of discoveries made"=frac.discoveries)

write.csv(df, paste(data_dir, "causal_forest.csv", sep="/"))

df_raw = data.frame("q_value"= q.vals, "H"=H.treated)

write.csv(df_raw, paste(data_dir, "causal_forest_raw.csv", sep="/"))
