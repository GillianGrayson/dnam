rm(list=ls())
options(java.parameters = "-Xmx16g")


if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("DSS")
BiocManager::install(c("minfi","ChAMPdata","Illumina450ProbeVariants.db","sva","IlluminaHumanMethylation450kmanifest","limma","RPMM","DNAcopy","preprocessCore","impute","marray","wateRmelon","goseq","plyr","GenomicRanges","RefFreeEWAS","qvalue","isva","doParallel","bumphunter","quadprog","shiny","shinythemes","plotly","RColorBrewer","DMRcate","dendextend","IlluminaHumanMethylationEPICmanifest","FEM","matrixStats","missMethyl","combinat"))
BiocManager::install("ramwas")
BiocManager::install("ChAMP")
install.packages(c( "foreach", "doParallel"))
install.packages('reticulate')

library("ChAMP")
library("reticulate")

library(devtools)
library(minfi)
library("regRCPqn")
library(sva)
library(minfi)
library("xlsx")
library("doParallel")
detectCores()

pd <- import("pandas")
path_load <- "E:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/040_report_mega_2022/data_for_R"
path_work <- "E:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/040_report_mega_2022/data_for_R"
setwd(path_work)


# Init Data ============================================================================================================
pheno <- pd$read_pickle(paste(path_load, "/pheno.pkl", sep=''))
pheno$Sex <- as.factor(pheno$Sex)
pheno$Sentrix_ID <- as.factor(pheno$Sentrix_ID)
pheno$Sentrix_Position <- as.factor(pheno$Sentrix_Position)

betas <- pd$read_pickle(paste(path_load, "/betas.pkl", sep=''))

# DMP Age ==============================================================================================================
dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Age,
  compare.group = NULL,
  adjPVal = 1,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$NumericVariable, file = "DMP_age.csv")

# DMP Region ===========================================================================================================
dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Sex,
  compare.group = NULL,
  adjPVal = 1,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$F_to_M, file = "DMP_sex.csv")