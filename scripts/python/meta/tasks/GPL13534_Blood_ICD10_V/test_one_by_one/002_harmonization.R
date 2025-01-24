rm(list=ls())

install.packages('reticulate')
py_install("pandas")
install_github("https://github.com/regRCPqn/regRCPqn")

library("reticulate")
library(devtools)
library(minfi)
library("regRCPqn")

path_ref <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/GPL13534_Blood_ICD10-V/R/one_by_one"
path_wd <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/GPL13534_Blood_ICD10-V/test"
setwd(path_wd)

pd <- import("pandas")

datasets <- c("GSE113725", "GSE116378", "GSE116379", "GSE41169")
for (dataset in datasets){
  pheno <- pd$read_pickle(paste("pheno_", toString(dataset), ".pkl", sep=''))
  mvals <- pd$read_pickle(paste("mvalsT_", toString(dataset), ".pkl", sep=''))
  mvals <- cbind(ID_REF = rownames(mvals), mvals)
  rownames(mvals) <- 1:nrow(mvals)

  mvals_norm <- regRCPqnREF(M_data=mvals, ref_path=paste(path_ref, "/reference/", sep=''), data_name="GSE87571")
  mvals_norm <- cbind(ID_REF = rownames(mvals_norm), mvals_norm)
  rownames(mvals_norm) <- 1:nrow(mvals_norm)
  write.table(mvals_norm, file=paste(path_wd, "/", "mvals_", toString(dataset), "_regRCPqn.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)
}