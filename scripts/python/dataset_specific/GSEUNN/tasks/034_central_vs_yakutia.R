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
Sys.setenv(RETICULATE_PYTHON = "C:/Users/user/anaconda3/envs/py39/python.exe")
library("reticulate")
py_config()
Sys.which('python')
use_condaenv('py39')
py_run_string('print(1+1)')

library(devtools)
library(minfi)
library("regRCPqn")
library(sva)
library(minfi)
library("xlsx")
library("doParallel")
detectCores()

pd <- import("pandas")
path_load <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/034_central_vs_yakutia/tmp/dnam/data_for_R"
path_work <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/034_central_vs_yakutia/tmp/dnam/data_for_R"
setwd(path_work)


# Init Data ============================================================================================================
pheno <- pd$read_pickle(paste(path_load, "/pheno.pkl", sep=''))
pheno$Region <- as.factor(pheno$Region)
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
write.csv(dmp$NumericVariable, file = "DMP_Age.csv")

# DMP Region ===========================================================================================================
dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  adjPVal = 1,
  adjust.method = "BH",
  arraytype = "EPIC"
)
DMP.GUI(
  DMP=dmp$Yakutia_to_Central,
  beta=betas,
  pheno=pheno$Region
)
write.csv(dmp$Central_to_Yakutia, file = "DMP_Region.csv")

# DMR ==================================================================================================================
dmr_Bumphunter <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Region,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = 7,
  adjPvalDmr = 0.05,
  cores = 12,
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
write.csv(dmr_Bumphunter$BumphunterDMR, file = "DMR_Regiom_Bumphunter.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_Regiom_Bumphunter_genes.csv", row.names=FALSE)

DMR.GUI(
  DMR = dmr_Bumphunter,
  beta = betas,
  pheno = pheno$Region,
  runDMP = TRUE,
  compare.group = NULL,
  arraytype = "EPIC"
)

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

# Residence ============================================================================================================
pheno_yakutia <- pd$read_pickle(paste(path_load, "/pheno_yakutia.pkl", sep=''))
pheno_yakutia$Residence <- as.factor(pheno_yakutia$Residence)
betas_yakutia <- pd$read_pickle(paste(path_load, "/betas_yakutia.pkl", sep=''))
dmp_yakutia <- champ.DMP(
  beta = betas_yakutia,
  pheno = pheno_yakutia$Residence,
  compare.group = NULL,
  adjPVal = 1,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp_yakutia$Village_to_City, file = "DMP_yakutia_residence.csv")