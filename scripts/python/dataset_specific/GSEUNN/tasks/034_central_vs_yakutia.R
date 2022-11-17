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
# pheno$DNAmPart <- as.factor(pheno$DNAmPart)
pheno$Sentrix_ID <- as.factor(pheno$Sentrix_ID)
pheno$Sentrix_Position <- as.factor(pheno$Sentrix_Position)

betas <- pd$read_pickle(paste(path_load, "/betas.pkl", sep=''))

champ.SVD(
  beta = betas,
  pd = pheno,
  RGEffect = FALSE,
  PDFplot = TRUE,
  Rplot = TRUE,
  resultsDir = "./SVD/"
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

betas <- corrrected


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

# DMP ==================================================================================================================
dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  adjPVal = 1,
  adjust.method = "BH",
  arraytype = "EPIC"
)
DMP.GUI(
  DMP=dmp$Central_to_Yakutia,
  beta=betas,
  pheno=pheno$Region
)
write.csv(dmp$Yakutia_to_Central, file = "DMP_region.csv")

# DMR ==================================================================================================================
betas <- data.matrix(betas)
dmr <- champ.DMR(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = 10,
  adjPvalDmr = 0.01,
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
  minDmrSize=20,
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

RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "dmr_genes.csv", row.names=FALSE)

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "fisher",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = 0.05,
  cores = 4
)
write.csv(data.frame(gsea$DMR), file = "dmr_gsea.csv", row.names=FALSE)

# DMB ==================================================================================================================
dmb <- champ.Block(
  beta = betas,
  pheno = pheno$Region,
  arraytype = "EPIC",
  maxClusterGap = 250000,
  B = 500,
  bpSpan = 250000,
  minNum = 10,
)

Block.GUI(
  Block = dmb,
  beta = betas,
  pheno = pheno$Region,
  runDMP = TRUE,
  compare.group = NULL,
  arraytype = "EPIC"
)