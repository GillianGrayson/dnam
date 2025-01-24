rm(list=ls())

install.packages('reticulate')
py_install("pandas")
install_github("https://github.com/regRCPqn/regRCPqn")

library("reticulate")
library(devtools)
library(minfi)
library("regRCPqn")

path_load <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/to_delete/Parkinson/origin"
path_ref <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/to_delete/Parkinson/harmonized/ref/"
path_save <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/to_delete/Parkinson/harmonized/r"
setwd(path_save)

pd <- import("pandas")

dataset_ref <- "GSE145361"
pheno_ref <- pd$read_pickle(paste(path_load, "/pheno_", dataset_ref, ".pkl", sep=''))
mvals_ref <- pd$read_pickle(paste(path_load, "/mvalsT_", dataset_ref, ".pkl", sep=''))
mvals_ref <- cbind(ID_REF = rownames(mvals_ref), mvals_ref)
rownames(mvals_ref) <- 1:nrow(mvals_ref)

mvals_norm <- regRCPqn(M_data=mvals_ref, ref_path=path_ref, data_name=dataset_ref, save_ref=TRUE)
mvals_norm <- cbind(ID_REF = rownames(mvals_norm), mvals_norm)
rownames(mvals_norm) <- 1:nrow(mvals_norm)
write.table(mvals_norm, file=paste(path_save, "/", "mvalsT_", dataset_ref, "_regRCPqn.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)

datasets <- c("GSE111629", "GSE72774")
for (dataset in datasets){
  pheno <- pd$read_pickle(paste(path_load, "/pheno_", toString(dataset), ".pkl", sep=''))
  mvals <- pd$read_pickle(paste(path_load, "/mvalsT_", toString(dataset), ".pkl", sep=''))
  mvals <- cbind(ID_REF = rownames(mvals), mvals)
  rownames(mvals) <- 1:nrow(mvals)

  mvals_norm <- regRCPqnREF(M_data=mvals, ref_path=path_ref, data_name=dataset_ref)
  mvals_norm <- cbind(ID_REF = rownames(mvals_norm), mvals_norm)
  rownames(mvals_norm) <- 1:nrow(mvals_norm)
  write.table(mvals_norm, file=paste(path_save, "/", "mvalsT_", toString(dataset), "_regRCPqn.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)
}