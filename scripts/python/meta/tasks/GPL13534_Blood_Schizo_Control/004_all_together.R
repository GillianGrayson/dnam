rm(list=ls())

install.packages('reticulate')
py_install("pandas")
install_github("https://github.com/regRCPqn/regRCPqn")

library("reticulate")
library(devtools)
library(minfi)
library("regRCPqn")
library("ChAMP")
library("doParallel")


path_load <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/GPL13534_Blood_Schizo_Control/origin"
path_save <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/GPL13534_Blood_Schizo_Control/all_together"
setwd(path_save)

pd <- import("pandas")

dataset <- "all"
pheno <- pd$read_pickle(paste(path_load, "/pheno_", dataset, ".pkl", sep=''))
mvals <- pd$read_pickle(paste(path_load, "/mvalsT_", dataset, ".pkl", sep=''))
mvals <- cbind(ID_REF = rownames(mvals), mvals)
rownames(mvals) <- 1:nrow(mvals)

champ.SVD(beta = mvals,
          pd = pheno,
          RGEffect = FALSE,
          PDFplot = TRUE,
          Rplot = TRUE,
          resultsDir = "./SVD_before/")

tmpCombat <- champ.runCombat(beta = mvals,
                             pd = pheno,
                             variablename = "Status",
                             batchname = c("Dataset"),
                             logitTrans = FALSE)

champ.SVD(beta = tmpCombat,
          pd = pd,
          RGEffect = FALSE,
          PDFplot = TRUE,
          Rplot = TRUE,
          resultsDir = "./SVD_after/")

combat_df <- data.frame(row.names(tmpCombat), tmpCombat)
colnames(combat_df)[1] <- "ID_REF"
write.table(combat_df, file=paste(path_save, "/", "mvals_combat.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)