rm(list=ls())
options(java.parameters = "-Xmx16g")

install.packages(c( "foreach", "doParallel"))
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install(c("minfi","ChAMPdata","Illumina450ProbeVariants.db","sva","IlluminaHumanMethylation450kmanifest","limma","RPMM","DNAcopy","preprocessCore","impute","marray","wateRmelon","goseq","plyr","GenomicRanges","RefFreeEWAS","qvalue","isva","doParallel","bumphunter","quadprog","shiny","shinythemes","plotly","RColorBrewer","DMRcate","dendextend","IlluminaHumanMethylationEPICmanifest","FEM","matrixStats","missMethyl","combinat"))
BiocManager::install("ramwas")

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

path_load <- "E:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN"


# Check ChAMP with preprocessed files ==================================================================================
pheno <- pd$read_pickle(paste(path_load, "/pheno_xtd.pkl", sep=''))
betas <- pd$read_pickle(paste(path_load, "/betas.pkl", sep=''))

df <- cbind(pheno, betas)

cpgs <- colnames(betas)
features <- c("Age", "Region", "DNAmPart", "Sentrix_ID", "Sentrix_Position")

df <- df[df$Status == "Control" &  df$Sample_Chronology < 2]

pd <- read.table("E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/unn_dataset_specific/007_prepare_combined_data_for_R/GSE87571/pheno.csv",
                 header = TRUE,
                 sep = ",",
                 dec = ".",
                 row.names = "subject_id")

champ.SVD(beta = tmpNorm,
          pd = pd,
          RGEffect = FALSE,
          PDFplot = TRUE,
          Rplot = TRUE,
          resultsDir = "./SVD_tmp/")

tmpCombat <- champ.runCombat(beta = tmpNorm,
                             pd = pd,
                             variablename = "Age",
                             batchname = c("Dataset"),
                             logitTrans = TRUE)

champ.SVD(beta = tmpCombat,
          pd = pd,
          RGEffect = FALSE,
          PDFplot = TRUE,
          Rplot = TRUE,
          resultsDir = "./SVD_tmp/")

tmpCombat_df <- data.frame(row.names(tmpCombat), tmpCombat)
colnames(tmpCombat_df)[1] <- "CpG"
write.table(tmpCombat_df, file = "tmpCombat.txt", row.names = F, sep = "\t", quote = F)
save(tmpCombat, file="tmpCombat.RData")
load("tmpCombat.RData")
tmpCombat <- as.matrix(tmpCombat)
tmpCombat <- read.table("tmpCombat.txt",
                      header = TRUE,
                      sep = "\t",
                      dec = ".",
                      row.names = "CpG")

tmpDMR <- champ.DMR(beta = tmpCombat,
                   pheno = pd$Dataset,
                   compare.group = NULL,
                   arraytype = "450K",
                   method = "Bumphunter",
                   minProbes = 5,
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
                   nullMethod = "bootstrap")
write.csv(tmpDMR, file = "tmpDMR_Bumphunter.csv")
save(tmpDMR, file="tmpDMR_Bumphunter.RData")
load("tmpDMR_Bumphunter.RData")

myDMR <- champ.DMR(beta = tmpCombat,
                   pheno = pd$Dataset,
                   compare.group = NULL,
                   arraytype = "450K",
                   method = "ProbeLasso",
                   minProbes = 5,
                   adjPvalDmr = 0.05,
                   cores = 4,
                   ## following parameters are specifically for probe ProbeLasso method.
                   meanLassoRadius = 375,
                   minDmrSep = 1000,
                   minDmrSize = 50,
                   adjPvalProbe = 0.001,
                   Rplot = FALSE,
                   PDFplot = FALSE,
                   resultsDir = "./ProbeLasso/")
write.csv(myDMR, file = "myDMR_ProbeLasso.csv")
save(myDMR, file="myDMR_ProbeLasso.RData")

myDMR <- champ.DMR(beta = tmpCombat,
                   pheno = pd$Dataset,
                   compare.group = NULL,
                   arraytype = "450K",
                   method = "DMRcate",
                   minProbes = 5,
                   adjPvalDmr = 0.05,
                   cores = 4,
                   ## following parameters are specifically for DMRcate method.
                   rmSNPCH = T,
                   fdr = 0.05,
                   dist = 2,
                   mafcut = 0.05,
                   lambda = 1000,
                   C = 2)

DMR.GUI(DMR = tmpDMR,
        beta = tmpCombat,
        pheno = pd$Dataset,
        runDMP = TRUE,
        compare.group = NULL,
        arraytype = "450K")

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