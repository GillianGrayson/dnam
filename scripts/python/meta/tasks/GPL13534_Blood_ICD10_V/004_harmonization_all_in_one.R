rm(list=ls())

install.packages('reticulate')
py_install("pandas")
install_github("https://github.com/regRCPqn/regRCPqn")

library("reticulate")
library(devtools)
library(minfi)
library("regRCPqn")

path <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/GPL13534_Blood_ICD10-V/R/all_in_one"
setwd(path)

pd <- import("pandas")

pheno <- pd$read_pickle("pheno.pkl")

mvals <- pd$read_pickle("mvalsT.pkl")
mvals <- cbind(ID_REF = rownames(mvals), mvals)
rownames(mvals) <- 1:nrow(mvals)

mvals_norm <- regRCPqn(M_data=mvals, ref_path=paste(path, "/", sep=''), data_name="regRCPqn", save_ref=TRUE)

mvals_norm <- cbind(ID_REF = rownames(mvals_norm), mvals_norm)
rownames(mvals_norm) <- 1:nrow(mvals_norm)
write.table(mvals_norm, file=paste(path, "/", "mvals_regRCPqn.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)
