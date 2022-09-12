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
path_load <- "E:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/034_central_vs_yakutia/data_for_R"
path_work <- "E:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/034_central_vs_yakutia/data_for_R"
setwd(path_work)


# Init Data ============================================================================================================
pheno <- pd$read_pickle(paste(path_load, "/pheno.pkl", sep=''))
pheno$Region <- as.factor(pheno$Region)
pheno$DNAmPart <- as.factor(pheno$DNAmPart)
pheno$Sentrix_ID <- as.factor(pheno$Sentrix_ID)
pheno$Sentrix_Position <- as.factor(pheno$Sentrix_Position)

betas <- pd$read_pickle(paste(path_load, "/betas.pkl", sep=''))

champ.SVD(
  beta = betas,
  pd = pheno,
  RGEffect = FALSE,
  PDFplot = TRUE,
  Rplot = TRUE,
  resultsDir = "./SVD_0/"
)

# Correction with Combat ===============================================================================================
corrrected <- champ.runCombat(
  beta = betas,
  pd = pheno,
  variablename = "Region",
  batchname = c("Age"),
  logitTrans = TRUE
)

champ.SVD(
  beta = corrrected,
  pd = pheno,
  RGEffect = FALSE,
  PDFplot = TRUE,
  Rplot = TRUE,
  resultsDir = "./SVD_1/"
)

corrrected_df <- data.frame(row.names(tmpCombat), tmpCombat)
colnames(corrrected_df)[1] <- "CpG"
write.table(corrrected_df, file = "corrrected.txt", row.names = F, sep = "\t", quote = F)
save(corrrected, file="corrrected.RData")
load("corrrected.RData")
corrrected <- as.matrix(corrrected)
corrrected <- read.table(
  "regressed.txt",
  header = TRUE,
  sep = "\t",
  dec = ".",
  row.names = "CpG"
)

betas <- corrrected

# DMP ==================================================================================================================
dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  adjPVal = 0.05,
  adjust.method = "BH",
  arraytype = "EPIC"
)
DMP.GUI(
  DMP=dmp$Central_to_Yakutia,
  beta=betas,
  pheno=pheno$Region
)
write.csv(dmp$Central_to_Yakutia, file = "dmp.csv")
write.table(dmp$Central_to_Yakutia, file = "dmp.txt", row.names = F, sep = "\t", quote = F)

# DMR ==================================================================================================================
betas <- data.matrix(betas)
pheno <- data.matrix(pheno)
dmr <- champ.DMR(
  beta = betas1,
  pheno = pheno$Region,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "DMRcate", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = 10,
  adjPvalDmr = 0.05,
  cores = 4,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap",
  ## following parameters are specifically for probe ProbeLasso method.
  meanLassoRadius=375,
  minDmrSep=1000,
  minDmrSize=50,
  adjPvalProbe=0.05,
  Rplot=TRUE,
  PDFplot=TRUE,
  resultsDir="./CHAMP_ProbeLasso/",
  ## following parameters are specifically for DMRcate method.
  rmSNPCH=TRUE,
  fdr=0.05,
  dist=2,
  mafcut=0.05,
  lambda=1000,
  C=2
)
write.csv(dmr, file = "dmr.csv")

DMR.GUI(
  DMR = dmr,
  beta = betas,
  pheno = pheno$Region,
  runDMP = TRUE,
  compare.group = NULL,
  arraytype = "EPIC"
)


myGSEA <- champ.GSEA(beta = tmpCombat,
                     DMP = NULL,
                     DMR = tmpDMR,
                     CpGlist = NULL,
                     Genelist = NULL,
                     pheno = pd$Dataset,
                     method = "fisher",
                     arraytype = "450K",
                     Rplot = TRUE,
                     adjPval = 0.05,
                     cores = 4)