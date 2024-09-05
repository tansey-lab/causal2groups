library(FDRreg)

## Read in directory
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]

# Load the data
X = read.csv(file.path(paste(data_dir, "X.csv", sep="/")), header=TRUE, sep=',')
Y = read.csv(file.path(paste(data_dir, "Y.csv", sep="/")), header=TRUE, sep=',')[,1]
T_ = read.csv(file.path(paste(data_dir, "T.csv", sep="/")), header=TRUE, sep=',')[,1]
H = read.csv(file.path(paste(data_dir, "H.csv", sep="/")), header=TRUE, sep=',')[,1]

n.treated = sum(T_)

## Subset to treated population
X.treated = X[T_==1,]
H.treated = H[T_==1]
Y.treated = Y[T_==1]

## Total positive
n.total.pos = sum(H.treated)


# calculate Z score on Y
Y_mean = mean(Y.treated)
Y_std = sd(Y.treated)
Z.treated = (Y.treated - Y_mean) / Y_std


# Run FDR regression
fdr_reg = FDRreg(Z.treated, X.treated)
q.vals <- fdr_reg$FDR

# FDRs
alphas = seq(0, 1, length.out = 1000)

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

write.csv(df, paste(data_dir, "FDRreg.csv", sep="/"))

df_raw = data.frame("q_value"= q.vals, "H"=H.treated)

write.csv(df_raw, paste(data_dir, "FDRreg_raw.csv", sep="/"))




