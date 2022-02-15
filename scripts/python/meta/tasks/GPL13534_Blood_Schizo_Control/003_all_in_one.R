rm(list=ls())

install.packages('reticulate')
py_install("pandas")
install_github("https://github.com/regRCPqn/regRCPqn")

library("reticulate")
library(devtools)
library(minfi)
library("regRCPqn")

path_load <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/GPL13534_Blood_Schizo_Control/origin"
path_ref <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/GPL13534_Blood_Schizo_Control/all_in_one/ref/"
path_save <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/GPL13534_Blood_Schizo_Control/all_in_one"
setwd(path_save)

pd <- import("pandas")

dataset_ref <- "train_val"
pheno_ref <- pd$read_pickle(paste(path_load, "/pheno_", dataset_ref, ".pkl", sep=''))
mvals_ref <- pd$read_pickle(paste(path_load, "/mvalsT_", dataset_ref, ".pkl", sep=''))
mvals_ref <- cbind(ID_REF = rownames(mvals_ref), mvals_ref)
rownames(mvals_ref) <- 1:nrow(mvals_ref)

mvals_norm <- regRCPqn(M_data=mvals_ref, ref_path=path_ref, data_name=dataset_ref, save_ref=TRUE)
mvals_norm <- cbind(ID_REF = rownames(mvals_norm), mvals_norm)
rownames(mvals_norm) <- 1:nrow(mvals_norm)
write.table(mvals_norm, file=paste(path_save, "/", "mvals_", dataset_ref, "_regRCPqn.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)

datasets <- c("GSE116378", "GSE116379", "GSE41169")
for (dataset in datasets){
  pheno <- pd$read_pickle(paste(path_load, "/pheno_", toString(dataset), ".pkl", sep=''))
  mvals <- pd$read_pickle(paste(path_load, "/mvalsT_", toString(dataset), ".pkl", sep=''))
  mvals <- cbind(ID_REF = rownames(mvals), mvals)
  rownames(mvals) <- 1:nrow(mvals)

  mvals_norm <- regRCPqnREF(M_data=mvals, ref_path=path_ref, data_name=dataset_ref)
  mvals_norm <- cbind(ID_REF = rownames(mvals_norm), mvals_norm)
  rownames(mvals_norm) <- 1:nrow(mvals_norm)
  write.table(mvals_norm, file=paste(path_save, "/", "mvals_", toString(dataset), "_regRCPqn.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)
}