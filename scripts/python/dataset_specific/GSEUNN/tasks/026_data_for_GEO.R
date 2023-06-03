rm(list=ls())


if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("DMRcate")
BiocManager::install("methylumi")
BiocManager::install("ChAMP")
BiocManager::install("minfi")
BiocManager::install("minfiData")
BiocManager::install("wateRmelon")
BiocManager::install("shinyMethyl")
BiocManager::install("FlowSorted.Blood.EPIC")
BiocManager::install("FlowSorted.Blood.450k")
BiocManager::install("FlowSorted.DLPFC.450k")

library(ChAMP)
library("xlsx")
library(openxlsx)
library(minfi)
library(minfiData)
library(shinyMethyl)
library(wateRmelon)
library(FlowSorted.Blood.EPIC)
library(FlowSorted.Blood.450k)
library(FlowSorted.DLPFC.450k)
library(sva)

myDir <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/raw/idat"
targets <- read.metharray.sheet(myDir)
rgSet <- read.metharray.exp(targets = targets, extended = FALSE, force = force)
mset <- preprocessRaw(rgSet)
meth_table <- getMeth(mset)
unmeth_table <- getUnmeth(mset)
pval_table <- detectionP(rgSet)
beta_table <- getBeta(mset)

meth_table <- cbind(ID_REF = rownames(meth_table), meth_table)
rownames(meth_table) <- 1:nrow(meth_table)
write.table(meth_table, file=paste(myDir, "/", "meth_table.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)

unmeth_table <- cbind(ID_REF = rownames(unmeth_table), unmeth_table)
rownames(unmeth_table) <- 1:nrow(unmeth_table)
write.table(unmeth_table, file=paste(myDir, "/", "unmeth_table.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)

pval_table <- cbind(ID_REF = rownames(pval_table), pval_table)
rownames(pval_table) <- 1:nrow(pval_table)
write.table(pval_table, file=paste(myDir, "/", "pval_table.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)

beta_table <- cbind(ID_REF = rownames(beta_table), beta_table)
rownames(beta_table) <- 1:nrow(beta_table)
write.table(beta_table, file=paste(myDir, "/", "beta_table.txt", sep=''), col.name=TRUE, row.names=FALSE, sep="\t", quote=F)

