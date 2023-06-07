rm(list=ls())
path <- "D:/YandexDisk/Work/pydnameth/datasets/GEO"
setwd(path)

.libPaths()
install.packages("vctrs")

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("GEOmetadb")
BiocManager::install("GEOquery")

library(GEOquery)
library(GEOmetadb)

getSQLiteFile(destdir = getwd(), destfile = "GEOmetadb.sqlite.gz")

if(!file.exists("GEOmetadb.sqlite")) getSQLiteFile() #command to download GEO SQLite file
file.info('GEOmetadb.sqlite')
con <- dbConnect(SQLite(),'GEOmetadb.sqlite') # con is an RSQLite connection object
geo_tables <- dbListTables(con) #function dbListTables lists all the tables in the SQLite database handled by the connection object con
geo_tables
dbListFields(con,'gsm') #dbListFields function that can list database fields associated with a table. 
dbListFields(con,'gse')
dbListFields(con,'gpl')

#Searching for EPICK or other beadchip methylation data

###########################
#to create the search term#
###########################

sql <- paste("SELECT DISTINCT",
             "gsm.ID,",
             "gsm.title,",
             "gsm.gsm,",
             "gsm.series_id,",
             "gsm.gpl,",
             "gsm.status,",
             "gsm.last_update_date,",
             "gsm.type,", 
             "gsm.source_name_ch1,",
             "gsm.organism_ch1,",
             "gsm.characteristics_ch1,",
             "gsm.molecule_ch1,",
             "gsm.label_ch1,",
             "gsm.treatment_protocol_ch1,",
             "gsm.extract_protocol_ch1,",
             "gsm.label_protocol_ch1,",
             "gsm.source_name_ch2,",
             "gsm.organism_ch2,",
             "gsm.characteristics_ch2,",
             "gsm.molecule_ch2,",
             "gsm.label_ch2,",
             "gsm.treatment_protocol_ch2,",
             "gsm.extract_protocol_ch2,",
             "gsm.label_protocol_ch2,",
             "gsm.hyb_protocol,",
             "gsm.description,",
             "gsm.data_processing,",   
             "gsm.supplementary_file,",
             "gsm.data_row_count,",
             "gsm.channel_count",
             "FROM",
             "gsm",
             "WHERE",
             "gsm.organism_ch1 LIKE '%Homo sapiens%' AND",
             "gsm.gpl LIKE '%GPL13534%'",sep=" ") # there the GPL of interest should be added. For example GPL8490 contain Human Methylation 27k Beadchip data (HM27k); but we can use GPL13534 (HM450k) or GPL21145 (EPIC Beadchip) to access other Illumina DNA methylation data platforms.

#############################
#Creating the GEO base table#
#############################

rs <- dbGetQuery(con, sql) #this is the GEO base table
# This next substitution is to guarantee that saving this table as a txt file, the previous tab operator do not interfer to reopen the GEO table.
rs$characteristics_ch1 <- gsub(";\t", ";", rs$characteristics_ch1, perl = TRUE)
write.csv(rs, file = "gsm_table.csv", row.names = FALSE)
dbDisconnect(con) #It is a good idea to close the connection
