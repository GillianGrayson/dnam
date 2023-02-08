rm(list=ls())

if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("DSS")
BiocManager::install(c("minfi","ChAMPdata","Illumina450ProbeVariants.db","sva","IlluminaHumanMethylation450kmanifest","limma","RPMM","DNAcopy","preprocessCore","impute","marray","wateRmelon","goseq","plyr","GenomicRanges","RefFreeEWAS","qvalue","isva","doParallel","bumphunter","quadprog","shiny","shinythemes","plotly","RColorBrewer","DMRcate","dendextend","IlluminaHumanMethylationEPICmanifest","FEM","matrixStats","missMethyl","combinat"))
BiocManager::install("ramwas")
BiocManager::install("ChAMP")
BiocManager::install("methylGSA")
install.packages(c( "foreach", "doParallel"))
install.packages('reticulate')


library("ChAMP")
library("methylGSA")
library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
Sys.setenv(RETICULATE_PYTHON = "C:/Users/user/anaconda3/envs/py39/python.exe")
library("reticulate")
py_config()
Sys.which('python')
use_condaenv('py39')
py_run_string('print(1+1)')

# all_region ===========================================================================================================
rm(list=ls())
pd <- import("pandas")

dmp_pval <- 1
dmr_pval <- 0.05
dmr_min_probes <- 10
gsea_pval <- 0.05

path_load <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/data_for_R"
path_work <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/00_all_region/data_from_R"
setwd(path_work)

pheno <- pd$read_pickle(paste(path_load, "/pheno_R_all_region.pkl", sep=''))
pheno$Region <- as.factor(pheno$Region)
betas <- pd$read_pickle(paste(path_load, "/betas_R_all_region.pkl", sep=''))

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = NULL,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "ebayes",
  arraytype = "EPIC",
  Rplot = FALSE,
  adjPval = gsea_pval,
  cores = 8
)
gtResult <- data.frame(row.names(gsea[[3]]), gsea[[3]])
colnames(gtResult)[1] <- "ID"
write.csv(gtResult, file = "GSEA(ebayes)_gtResult.csv", row.names=TRUE)
write.csv(gsea$GSEA[[1]], file = "GSEA(ebayes)_Rank(P).csv", row.names=TRUE)

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$Central_to_Yakutia, file = "DMP.csv")
dmp_df <- data.frame(row.names(dmp$Central_to_Yakutia), dmp$Central_to_Yakutia)
colnames(dmp_df)[1] <- "CpG"
cpg_pval <- setNames(dmp_df$adj.P.Val, dmp_df$CpG)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "GO",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_GO.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "KEGG",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_KEGG.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "Reactome",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_Reactome.csv", row.names=FALSE)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Region,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 8,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_genes.csv", row.names=FALSE)

GSEA_gometh <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "gometh",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = dmr_pval,
  cores = 8
)
write.csv(data.frame(GSEA_gometh$DMR), file = "DMR_GSEA_gometh.csv", row.names=FALSE)

# central_sex ==========================================================================================================
rm(list=ls())
pd <- import("pandas")

dmp_pval <- 1
dmr_pval <- 0.05
dmr_min_probes <- 10
gsea_pval <- 0.05

path_load <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/data_for_R"
path_work <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/01_central_sex/data_from_R"
setwd(path_work)

pheno <- pd$read_pickle(paste(path_load, "/pheno_R_central_sex.pkl", sep=''))
pheno$Sex <- as.factor(pheno$Sex)
betas <- pd$read_pickle(paste(path_load, "/betas_R_central_sex.pkl", sep=''))

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = NULL,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Sex,
  method = "ebayes",
  arraytype = "EPIC",
  Rplot = FALSE,
  adjPval = gsea_pval,
  cores = 8
)
gtResult <- data.frame(row.names(gsea[[3]]), gsea[[3]])
colnames(gtResult)[1] <- "ID"
write.csv(gtResult, file = "GSEA(ebayes)_gtResult.csv", row.names=TRUE)
write.csv(gsea$GSEA[[1]], file = "GSEA(ebayes)_Rank(P).csv", row.names=TRUE)

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Sex,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$F_to_M, file = "DMP.csv")
dmp_df <- data.frame(row.names(dmp$F_to_M), dmp$F_to_M)
colnames(dmp_df)[1] <- "CpG"
cpg_pval <- setNames(dmp_df$adj.P.Val, dmp_df$CpG)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "GO",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_GO.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "KEGG",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_KEGG.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "Reactome",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_Reactome.csv", row.names=FALSE)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Sex,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 8,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_genes.csv", row.names=FALSE)

GSEA_gometh <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Sex,
  method = "gometh",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = dmr_pval,
  cores = 8
)
write.csv(data.frame(GSEA_gometh$DMR), file = "DMR_GSEA_gometh.csv", row.names=FALSE)

# yakutia_sex ==========================================================================================================
rm(list=ls())
pd <- import("pandas")

dmp_pval <- 1
dmr_pval <- 0.05
dmr_min_probes <- 10
gsea_pval <- 0.05

path_load <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/data_for_R"
path_work <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/02_yakutia_sex/data_from_R"
setwd(path_work)

pheno <- pd$read_pickle(paste(path_load, "/pheno_R_yakutia_sex.pkl", sep=''))
pheno$Sex <- as.factor(pheno$Sex)
betas <- pd$read_pickle(paste(path_load, "/betas_R_yakutia_sex.pkl", sep=''))

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = NULL,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Sex,
  method = "ebayes",
  arraytype = "EPIC",
  Rplot = FALSE,
  adjPval = gsea_pval,
  cores = 8
)
gtResult <- data.frame(row.names(gsea[[3]]), gsea[[3]])
colnames(gtResult)[1] <- "ID"
write.csv(gtResult, file = "GSEA(ebayes)_gtResult.csv", row.names=TRUE)
write.csv(gsea$GSEA[[1]], file = "GSEA(ebayes)_Rank(P).csv", row.names=TRUE)

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Sex,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$F_to_M, file = "DMP.csv")
dmp_df <- data.frame(row.names(dmp$F_to_M), dmp$F_to_M)
colnames(dmp_df)[1] <- "CpG"
cpg_pval <- setNames(dmp_df$adj.P.Val, dmp_df$CpG)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "GO",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_GO.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "KEGG",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_KEGG.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "Reactome",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_Reactome.csv", row.names=FALSE)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Sex,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 8,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_genes.csv", row.names=FALSE)

GSEA_gometh <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Sex,
  method = "gometh",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = dmr_pval,
  cores = 8
)
write.csv(data.frame(GSEA_gometh$DMR), file = "DMR_GSEA_gometh.csv", row.names=FALSE)

# females_region =======================================================================================================
rm(list=ls())
pd <- import("pandas")

dmp_pval <- 1
dmr_pval <- 0.05
dmr_min_probes <- 10
gsea_pval <- 0.05

path_load <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/data_for_R"
path_work <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/03_females_region/data_from_R"
setwd(path_work)

pheno <- pd$read_pickle(paste(path_load, "/pheno_R_females_region.pkl", sep=''))
pheno$Region <- as.factor(pheno$Region)
betas <- pd$read_pickle(paste(path_load, "/betas_R_females_region.pkl", sep=''))

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = NULL,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "ebayes",
  arraytype = "EPIC",
  Rplot = FALSE,
  adjPval = gsea_pval,
  cores = 8
)
gtResult <- data.frame(row.names(gsea[[3]]), gsea[[3]])
colnames(gtResult)[1] <- "ID"
write.csv(gtResult, file = "GSEA(ebayes)_gtResult.csv", row.names=TRUE)
write.csv(gsea$GSEA[[1]], file = "GSEA(ebayes)_Rank(P).csv", row.names=TRUE)

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$Central_to_Yakutia, file = "DMP.csv")
dmp_df <- data.frame(row.names(dmp$Central_to_Yakutia), dmp$Central_to_Yakutia)
colnames(dmp_df)[1] <- "CpG"
cpg_pval <- setNames(dmp_df$adj.P.Val, dmp_df$CpG)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "GO",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_GO.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "KEGG",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_KEGG.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "Reactome",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_Reactome.csv", row.names=FALSE)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Region,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 8,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_genes.csv", row.names=FALSE)

GSEA_gometh <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "gometh",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = dmr_pval,
  cores = 8
)
write.csv(data.frame(GSEA_gometh$DMR), file = "DMR_GSEA_gometh.csv", row.names=FALSE)

# males_region =========================================================================================================
rm(list=ls())
pd <- import("pandas")

dmp_pval <- 1
dmr_pval <- 0.05
dmr_min_probes <- 10
gsea_pval <- 0.05

path_load <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/data_for_R"
path_work <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/04_males_region/data_from_R"
setwd(path_work)

pheno <- pd$read_pickle(paste(path_load, "/pheno_R_males_region.pkl", sep=''))
pheno$Region <- as.factor(pheno$Region)
betas <- pd$read_pickle(paste(path_load, "/betas_R_males_region.pkl", sep=''))

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = NULL,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "ebayes",
  arraytype = "EPIC",
  Rplot = FALSE,
  adjPval = gsea_pval,
  cores = 8
)
gtResult <- data.frame(row.names(gsea[[3]]), gsea[[3]])
colnames(gtResult)[1] <- "ID"
write.csv(gtResult, file = "GSEA(ebayes)_gtResult.csv", row.names=TRUE)
write.csv(gsea$GSEA[[1]], file = "GSEA(ebayes)_Rank(P).csv", row.names=TRUE)

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$Central_to_Yakutia, file = "DMP.csv")
dmp_df <- data.frame(row.names(dmp$Central_to_Yakutia), dmp$Central_to_Yakutia)
colnames(dmp_df)[1] <- "CpG"
cpg_pval <- setNames(dmp_df$adj.P.Val, dmp_df$CpG)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "GO",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_GO.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "KEGG",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_KEGG.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "Reactome",
  minsize = 100,
  maxsize = 500
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_Reactome.csv", row.names=FALSE)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Region,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 8,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_genes.csv", row.names=FALSE)

GSEA_gometh <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "gometh",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = dmr_pval,
  cores = 8
)
write.csv(data.frame(GSEA_gometh$DMR), file = "DMR_GSEA_gometh.csv", row.names=FALSE)
