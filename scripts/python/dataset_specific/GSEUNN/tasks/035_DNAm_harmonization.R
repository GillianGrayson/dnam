rm(list=ls())

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

pd <- import("pandas")
path_load <- "E:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/035_DNAm_harmonization"
path_work <- "E:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/035_DNAm_harmonization"
setwd(path_work)


# Init Data ============================================================================================================
pheno <- pd$read_pickle(paste(path_load, "/pheno.pkl", sep=''))
pheno$Sex <- as.factor(pheno$Sex)
pheno$Status <- as.factor(pheno$Status)
pheno$Region <- as.factor(pheno$Region)
pheno$Sentrix_ID <- as.factor(pheno$Sentrix_ID)
pheno$Sentrix_Position <- as.factor(pheno$Sentrix_Position)

betas <- pd$read_pickle(paste(path_load, "/betas.pkl", sep=''))

# SVD before harmonization with Combat =================================================================================
champ.SVD(
  beta = betas,
  pd = pheno,
  RGEffect = FALSE,
  PDFplot = TRUE,
  Rplot = TRUE,
  resultsDir = "./SVD_before_harm/"
)

# Correction with Combat ===============================================================================================
corrrected <- champ.runCombat(
  beta = betas,
  pd = pheno,
  variablename = c("Age", "Sex", "Status", "Region"),
  batchname = c("Sentrix_ID", "Sentrix_Position"),
  logitTrans = TRUE
)

# SVD after harmonization with Combat ==================================================================================
champ.SVD(
  beta = corrrected,
  pd = pheno,
  RGEffect = FALSE,
  PDFplot = TRUE,
  Rplot = TRUE,
  resultsDir = "./SVD_after_harm/"
)

corrrected_df <- data.frame(row.names(corrrected), corrrected)
colnames(corrrected_df)[1] <- "CpG"
write.table(corrrected_df, file = "corrected.txt", row.names = F, sep = "\t", quote = F)
