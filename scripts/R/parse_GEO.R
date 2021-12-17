rm(list=ls())
path <- "E:/YandexDisk/Work/pydnameth/datasets/GEO"
setwd(path)

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("GEOmetadb")

library(GEOmetadb)
library("xlsx")

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
             #"gsm.characteristics_ch1 LIKE '%age%' AND",
             #"gsm.characteristics_ch1 NOT LIKE '%blood%' AND",#commenting or not this four lines we at the end will include or exclude blood samples on the final table.
             #"gsm.source_name_ch1 NOT LIKE '%blood%' AND",
             #"gsm.characteristics_ch1 NOT LIKE '%leukocyte%' AND",
             #"gsm.source_name_ch1 NOT LIKE '%leukocyte%' AND",
             #"(gsm.characteristics_ch1 LIKE '%control%' OR gse.title LIKE '%control%' OR gsm.title LIKE '%control%' OR gsm.source_name_ch1 LIKE '%control%' OR gsm.characteristics_ch1 LIKE '%normal%' OR gse.title LIKE '%normal%' OR gsm.title LIKE '%normal%' OR gsm.source_name_ch1 LIKE '%normal%' OR gsm.characteristics_ch1 LIKE '%health%' OR gse.title LIKE '%health%' OR gsm.title LIKE '%health%' OR gsm.source_name_ch1 LIKE '%health%' OR  gsm.characteristics_ch1 LIKE '%non-tumor%' OR gsm.source_name_ch1 LIKE '%non-tumor%') AND",
             "gsm.gpl LIKE '%GPL21145%'",sep=" ") #here the GPL of interest should be added. For example GPL8490 contain Human Methylation 27k Beadchip data (HM27k); but we can use GPL13534 (HM450k) or GPL21145 (EPIC Beadchip) to access other Illumina DNA methylation data platforms.


sql <- paste("SELECT DISTINCT gsm.ID, gsm.title, gsm.gsm, gsm.series_id, gsm.gpl, gsm.status, gsm.last_update_date, gsm.type, gsm.source_name_ch1, gsm.organism_ch1, gsm.characteristics_ch1, gsm.source_name_ch2, gsm.organism_ch2, gsm.characteristics_ch2, gsm.data_row_count, gsm.supplementary_file",
             "FROM",
             "gsm JOIN gse_gsm ON gsm.gsm=gse_gsm.gsm",
             "JOIN gse ON gse_gsm.gse=gse.gse",
             "JOIN gse_gpl ON gse_gpl.gse=gse.gse",
             "JOIN gpl ON gse_gpl.gpl=gpl.gpl",
             "WHERE",
             "gsm.characteristics_ch1 LIKE '%age%' AND",
             #"gsm.characteristics_ch1 NOT LIKE '%blood%' AND",#commenting or not this four lines we at the end will include or exclude blood samples on the final table.
             #"gsm.source_name_ch1 NOT LIKE '%blood%' AND",
             #"gsm.characteristics_ch1 NOT LIKE '%leukocyte%' AND",
             #"gsm.source_name_ch1 NOT LIKE '%leukocyte%' AND",
             #"(gsm.characteristics_ch1 LIKE '%control%' OR gse.title LIKE '%control%' OR gsm.title LIKE '%control%' OR gsm.source_name_ch1 LIKE '%control%' OR gsm.characteristics_ch1 LIKE '%normal%' OR gse.title LIKE '%normal%' OR gsm.title LIKE '%normal%' OR gsm.source_name_ch1 LIKE '%normal%' OR gsm.characteristics_ch1 LIKE '%health%' OR gse.title LIKE '%health%' OR gsm.title LIKE '%health%' OR gsm.source_name_ch1 LIKE '%health%' OR  gsm.characteristics_ch1 LIKE '%non-tumor%' OR gsm.source_name_ch1 LIKE '%non-tumor%') AND",
             "gsm.gpl LIKE '%GPL21145%'",sep=" ") #here the GPL of interest should be added. For example GPL8490 contain Human Methylation 27k Beadchip data (HM27k); but we can use GPL13534 (HM450k) or GPL21145 (EPIC Beadchip) to access other Illumina DNA methylation data platforms.

#Note that on the last line  were used gsm.gpl "GPL8490" because that is the point that defines in what platforms the previous key-words will be searched.

#############################
#Creating the GEO base table#
#############################

rs <- dbGetQuery(con,sql) #this is the GEO base table 
#This next substitution is to guarantee that saving this table as a txt file, the previous tab operator do not interfer to reopen the GEO table.
rs$characteristics_ch1 <- gsub(";\t", ";", rs$characteristics_ch1, perl = TRUE)
write.table(rs,"GPL21145_gsm_table.txt",sep="\t",row.names=F,quote = F) 
write.xlsx(rs, "gsm_table.xlsx", sheetName = "Sheet1", col.names = TRUE, row.names = FALSE, append = FALSE)
#OR
#write.table(rs,"~/GPL13534/GPL13534_gsm_table.txt",sep="\t",row.names=F,quote = F)

gpl <- rs
#######################################################
#Disconecting and removing GEOmetadb parsed connection#
#######################################################

dbDisconnect(con) #It is a good idea to close the connection
file.remove('GEOmetadb.sqlite') #to remove old GEOmetadb.sqlite file before retrieve a new updated version from the server, at the beggining of this script.

##########
#        #
# PART 2 #
#        #                                            
##########

######################################################
#Uploading GSE table - In case it was already created#
######################################################

#rm(list=ls())
#setwd("~/Desktop/GEO_update_DEC2019/")
#gpl <- read.csv("GPL8490",sep="\t",header=T,quote="") # could be GPL13534, depending on the platform table generated on the part 1.
#gpl <- read.csv("GPL13534_gsm_table.txt",sep="\t",header=T,quote="")
#gpl <- read.csv("GPL21145_gsm_table.txt",sep="\t",header=T,quote="")
#dim(gpl)

#ATTENTION: from now This GEO table will be, filtred and changed. And will be refered as GPL table.

#####################
#creating Age column#
#####################

gpl$characteristics_ch1=as.character(gpl$characteristics_ch1)
gpl[8] <- NA
c=1
terms <- c()
for(i in 1:dim(gpl)[1]){
  var <- unlist(strsplit(as.character(gpl[i,7]),";"))
  for(j in 1:length(var)){
    # \\b matches the empty string at either edge of a word
    if (( grepl('\\bage', var[j],ignore.case = T) == T) & (grepl("[[:digit:]]", var[j])==T)) {
      if (( grepl('\\bStage\\b', var[j],ignore.case = T) == F) & (grepl('\\wgrp|\\bgrp', var[j],ignore.case = T) == F) & 
          (grepl('\\wbin|\\bbin', var[j],ignore.case = T) == F) & (grepl('tanner', var[j],ignore.case = T) == F) & 
          (grepl('\\Bage\\B', var[j],ignore.case = T) == F) & (grepl('acceleration', var[j],ignore.case = T) == F) & (grepl('dna', var[j],ignore.case = T) == F) & 
          (grepl('death', var[j],ignore.case = T) == F)) {
        if ( grepl('^cell description:', var[j]) == T){
          ## get only age when dealing with a cell description field
          string_age = sub(".*?(age [0-9]+).*", "\\1", var[j],perl=TRUE)
          gpl[i,7] = gsub('age', 'age:',string_age)
          terms[c] = 'age'
        }
        else {
          gpl[i,8] <- as.character(var[j])
          terms[c] = strsplit(var[j],':')[[1]][1] #to verify if there is a term not involved with the key words search
        }
      }
    }
  }
  c=c+1
}

unique(terms)
colnames(gpl)[8] <- "Age"
gpl$Age


sum(!is.na(gpl$Age))

##########################
#creating Age column Unit#
##########################

#creating a column with age unit (years,weeks,months)

gpl[9] <- 'NA'
for(i in 1:dim(gpl)[1]){
  if(!is.na(gpl[i,7])==TRUE){
    if (grepl("years",gpl[i,8],ignore.case=T)==T | grepl("yrs",gpl[i,8],ignore.case=T)==T | grepl("yr",gpl[i,8],ignore.case=T)==T | grepl("y",gpl[i,8],ignore.case=T)==T){
      gpl[i,9] <- "years"
    }
    if (grepl(">", gpl[i,8])==T){
      gpl[i,9] <- "more than in years"
    }
    if (grepl("<", gpl[i,8])==T){
      gpl[i,9] <- "less than in years"
    }
    if (grepl("months", gpl[i,8])==T){
      gpl[i,9] <- "months"
    }
    if (grepl("weeks", gpl[i,8])==T){
      gpl[i,9] <- "weeks"
    }
    if(grepl("years",gpl[i,8],ignore.case=T)==F & grepl("yrs",gpl[i,8],ignore.case=T)==F & grepl("yr",gpl[i,8],ignore.case=T)==F & grepl("y",gpl[i,8],ignore.case=T)==F & grepl("months",gpl[i,8],ignore.case=T)==F & grepl("weeks",gpl[i,8],ignore.case=T)==F){
      gpl[i,9] <- "years"
    }
  }
}
colnames(gpl)[9] <- "Age_unit"
unique(gpl$Age_unit)

#cleaning Age columns (after use it to obtain Age_unit colum) the intention here is remaining only with numbers
gpl[,8] <- gsub("^.*?:","",gpl[,8]) #clean everything before the colon
gpl[,8] <- gsub("[ <>-]","",gpl[,8]) #clean space, special character as > and < signals. Pay attention if minus operator is not a number's part.
gpl[,8] <- gsub("[[:alpha:]]","",gpl[,8]) # clean text characters
#gpl[,8] <- as.numeric(gpl[,8])

##############################################
#Removing samples which contain adjacent term#
##############################################

#The terms search followed after the generation of GEO base table have a clear goal: have as result an another table
#containing HEALH samples with clear information of AGE and TISSUE.
#When we remove the "adjacent" term we are removing undesiderable samples, because this term accompanies tumor associated
#tissues.

Adjacent_ch=c()
Adjacent_s=c()
Adjacent_t=c()
c=1
for(i in 1:dim(gpl)[1]){
  var <- unlist(strsplit(as.character(gpl[i,7]),";"))
  for(j in 1:length(var)){
    if (grepl("Adjacent",var[j],ignore.case=T)==T){
      Adjacent_ch[c] <- as.character(gpl[i,2])
    }
  }
  if (grepl("Adjacent",gpl[i,6],ignore.case=T)==T){
    Adjacent_s[c]  <- as.character(gpl[i,2])
  }
  if (grepl("Adjacent",gpl[i,1],ignore.case=T)==T){
    Adjacent_t[c] <- as.character(gpl[i,2])
  }
  c=c+1
}

gsm_adjacent_ch <- as.character(Adjacent_ch)
gsm_adjacent_s <- as.character(Adjacent_s)
gsm_adjacent_t <- as.character(Adjacent_t)
gsm_adjacent <- unique(c(gsm_adjacent_ch,gsm_adjacent_s,gsm_adjacent_t))
length(gsm_adjacent)
gpl <- gpl[!gpl$gsm %in% gsm_adjacent,] #gpl without adjacent data
dim(gpl)

######################################################
#Extracting control status using specific query terms#
######################################################

# Between the grep terms to exclude from the outcomes, -n was added to exclude p-normal and -q-normal. The operator ^ limitates the search only 
#grepping terms that started with normal; "\b" (which in R environment is \\b) or "\b " operator also reduces the search, so I decided exclude -n.

#controls from characteristics column [10]; source column [11] and title column [12]
gpl[10:12] <- NA
c=1
terms_ch <- c()
for(i in 1:dim(gpl)[1]){
  var <- unlist(strsplit(as.character(gpl[i,7]),";"))
  for(j in 1:length(var)){
    if (grepl("\\bControl",var[j],ignore.case=T)==T | grepl("\\bNormal",var[j],ignore.case=T)==T | grepl("\\bHealth",var[j],ignore.case=T)==T){
      if (grepl("FFPE",var[j],ignore.case=T)==F & grepl("miscarriage",var[j],ignore.case=T)==F & grepl("incubation",var[j],ignore.case=T)==F & grepl("cultured",var[j],ignore.case=T)==F & grepl("Cell line",var[j],ignore.case=T)==F & grepl("Carcinoma",var[j],ignore.case=T)==F & grepl("Lesion",var[j],ignore.case=T)==F & grepl("quality",var[j],ignore.case=T)==F & grepl("general",var[j])==F & grepl("group",var[j])==F & grepl("birth",var[j])==F & grepl("age",var[j])==F & grepl("negative",var[j])==F & grepl("braf",var[j])==F & grepl("hiv",var[j])==F & grepl("-n",var[j])==F & grepl("\\BNormal|Normal\\B",var[j],ignore.case=T)==F){
        gpl[i,10] <- as.character(var[j])
        #terms[c] = strsplit(var[j],':')[[1]][1]
        terms_ch[c] = var[j]
      }
    }
  }
  if (grepl("\\bControl",gpl[i,6],ignore.case=T)==T | grepl("\\bNormal",gpl[i,6],ignore.case=T)==T | grepl("\\bHealth",gpl[i,6],ignore.case=T)==T){
    if (grepl("FFPE",gpl[i,6],ignore.case=T)==F & grepl("miscarriage",gpl[i,6],ignore.case=T)==F & grepl("incubation",gpl[i,6],ignore.case=T)==F & grepl("cultured",gpl[i,6],ignore.case=T)==F & grepl("Cell line",gpl[i,6],ignore.case=T)==F & grepl("Carcinoma",gpl[i,6],ignore.case=T)==F & grepl("Lesion",gpl[i,6],ignore.case=T)==F & grepl("quality",gpl[i,6],ignore.case=T)==F & grepl("general",gpl[i,6])==F & grepl("group",gpl[i,6])==F & grepl("birth",gpl[i,6])==F & grepl("age",gpl[i,6])==F & grepl("negative",gpl[i,6])==F & grepl("braf",gpl[i,6])==F & grepl("hiv",gpl[i,6])==F & grepl("-n",gpl[i,6])==F  & grepl("\\BNormal|Normal\\B",gpl[i,6],ignore.case=T)==F){
      gpl[i,11] <- as.character(gpl[i,6])
    }
  }
  if (grepl("\\bControl",gpl[i,1],ignore.case=T)==T | grepl("\\bNormal",gpl[i,1],ignore.case=T)==T | grepl("\\bHealth",gpl[i,1],ignore.case=T)==T){
    if (grepl("FFPE",gpl[i,1],ignore.case=T)==F & grepl("miscarriage",gpl[i,1],ignore.case=T)==F & grepl("incubation",gpl[i,1],ignore.case=T)==F & grepl("cultured",gpl[i,1],ignore.case=T)==F & grepl("Cell line",gpl[i,1],ignore.case=T)==F & grepl("Carcinoma",gpl[i,1],ignore.case=T)==F & grepl("Lesion",gpl[i,1],ignore.case=T)==F & grepl("quality",gpl[i,1],ignore.case=T)==F & grepl("general",gpl[i,1])==F & grepl("group",gpl[i,1])==F & grepl("birth",gpl[i,1])==F & grepl("age",gpl[i,1])==F & grepl("negative",gpl[i,1])==F & grepl("braf",gpl[i,1])==F & grepl("hiv",gpl[i,1])==F & grepl("-n",gpl[i,1])==F  & grepl("\\BNormal|Normal\\B",gpl[i,1],ignore.case=T)==F){
      gpl[i,12] <- as.character(gpl[i,1])
    }
  }
  c=c+1
}

colnames(gpl)[10:12] <- c("control_ch_col","control_source_col","control_title_col")

#With these information bellow we will write the substitutions...
unique(gpl$control_ch_col)
unique(gpl$control_source_col)
unique(gpl$control_title_col)

#This point we will be usefull for two things: (1) choose what terms we will substitute bellow to not create false positive controls for example: 
#healthstate:Diseaseddonor should have the term health removed so it were subtituted by Diseaseddonor; and (2) to see if there is some therms that
#have nothing to do with contoll/normal/healthy terms, if there is something completely unrelated with these words, this term could be added as a term to
#be excluded from the search - put inside the inner if together with "quality, general, group,..." current terms.
gpl[,10] <- gsub("[ ()=/,]","",gpl[,10]) 
unique(gpl$control_ch_col)
#unique(gpl$control_source_col)
#unique(gpl$control_title_col)

#################################
#Cleaning control status columns#      
#################################

#the idea of this step is identify what control terms are not from control samples. So doing this
#substitutions bellow we will guarantee that control/normal/healthy key words being associated with 
#real control healthy samples.

#...here to clean control status columns, result from a regular expression search on the GEO base table.
# Note that mainly in the characteristics_ch comum there is irregular quantity of information, in a irregular
# description, i.e., data type with the same nature can have very differente explanatory prefixes.

#This point we will be usefull for two things: (1) choose what terms we will substitute bellow to not create false positive controls for example: 
#healthstate:Diseaseddonor should have the term health removed so it were subtituted by Diseaseddonor; and (2) to see if there is some terms that
#have nothing to do with control/normal/healthy terms, if there is something completely unrelated with these words, this term could be added as a term to
#be excluded from the search - to this we have add them inside the inner if loop, together with "quality, general, group,..." current terms.

gpl[,10] <- gsub("[ ()=/,]","",gpl[,10]) #At this point all special characters/puntuaction inside the brackets are removed.
#This was necessary to facilitate the use of the function gsub.
unique(gpl$control_ch_col)
unique(gpl$control_source_col)
unique(gpl$control_title_col)

gpl[,10] <- gsub("status0normal1cancer:0","normal",gpl[,10])
gpl[,10] <- gsub("status0normal1cancer:1","cancer",gpl[,10])
gpl[,10] <- gsub("casecontrol:1","unknown",gpl[,10]) #case_breast_cancer within a 5-year follow-up period_GSE58119
gpl[,10] <- gsub("casecontrol:0","unknown",gpl[,10]) #control_remained_cancer-free within a 5-year follow-up period_GSE58119
gpl[,10] <- gsub("casecontrol:Control","control",gpl[,10])
gpl[,10] <- gsub("casecontrol:Case","case",gpl[,10])
gpl[,10] <- gsub("diseasestatus:control-HPVpos","affected",gpl[,10])
gpl[,10] <- gsub("diseasestatus:control-HPVneg","normal",gpl[,10])
gpl[,10] <- gsub("status0healthy1cancer:0","healthy",gpl[,10])
gpl[,10] <- gsub("status0healthy1cancer:1","cancer",gpl[,10]) 
gpl[,10] <- gsub("status0wthealthy1wtcancer2muthealthy3mutcancer:0","healthy",gpl[,10]) 
gpl[,10] <- gsub("status0wthealthy1wtcancer2muthealthy3mutcancer:1","wt/cancer",gpl[,10])
gpl[,10] <- gsub("status0wthealthy1wtcancer2muthealthy3mutcancer:2","mutated",gpl[,10])
gpl[,10] <- gsub("status0wthealthy1wtcancer2muthealthy3mutcancer:3","mut/cancer",gpl[,10])
gpl[,10] <- gsub("disease status1control2scz patient:1","control",gpl[,10])
gpl[,10] <- gsub("disease status1control2scz patient:2","scz patient",gpl[,10])
gpl[,10] <- gsub("healthstate:Diseaseddonor","Diseaseddonor",gpl[,10])
gpl[,10] <- gsub("^.*?:","",gpl[,10]) # to remove everything before ":" (including it)
gpl[,10] <- gsub("[[:digit:]]","",gpl[,10]) #clean excess of digits, to see short and clear unique categories.
unique(gpl$control_ch_col)
gpl[,11] <- gsub("[[:digit:]]","",gpl[,11]) #clean excess of digits, to see short and clear unique categories.
unique(gpl$control_source_col)
gpl[,12] <- gsub("[[:digit:]]","",gpl[,12]) #clean excess of digits, to see short and clear unique categories.
unique(gpl$control_title_col)

#################################
#Joining, control status columns#
#################################

gpl[13] <- NA
for(i in 1:dim(gpl)[1]){
  if (grepl("\\bControl",gpl[i,10],ignore.case=T)==T | grepl("\\bNormal",gpl[i,10],ignore.case=T)==T | grepl("\\bHealth",gpl[i,10],ignore.case=T)==T | grepl("\\bControl",gpl[i,11],ignore.case=T)==T | grepl("\\bNormal",gpl[i,11],ignore.case=T)==T | grepl("\\bHealth",gpl[i,11],ignore.case=T)==T | grepl("\\bControl",gpl[i,12],ignore.case=T)==T | grepl("\\bNormal",gpl[i,12],ignore.case=T)==T | grepl("\\bHealth",gpl[i,12],ignore.case=T)==T){
    gpl[i,13] <- 1
  }
  else{
    gpl[i,13] <- "NA"
  }
}

colnames(gpl)[13] <- c("Merged_codifyed_control_cols")
dim(gpl)

####################
#reducing GPL table#
####################

#reducing this current gpl table, we will follow with the tissue search. 
#The tissue search are the time limiting step of this analysis.
#So from now, it is interesting that we follow only with the samples that will be object of the work:
#One more time: DNA methylation data from healthy(control) samples, containing AGE and TISSUE information.

dim(gpl) # (27k)3018 rows and 17 columns
gpl_Age=gpl[!is.na(gpl$Age),]
dim(gpl_Age) # (27k) 2606/3018 (450k)6461/7011
gpl_Age_Control=gpl_Age[gpl_Age$Merged_codifyed_control_cols==1,]
dim(gpl_Age_Control) # (27k) 1341/2606 (450k)3131/6461
gpl <- gpl_Age_Control

####################################
#Opening MESH tissue ontology table#
####################################

#As tissue information metadata is a kind of messy, because are not linked in a standard retractable term
#we used this table above, which refers to a tissue ontology database. So matching the tissue
#terms of these table with the current GPL columns (characteristics_ch, title, and source_ch - that are suposed
#to contain samples' tissue data) to recover that desirable information.

#opening the list of tissues. 
tissues <- read.table("Desktop/GEO_update_DEC2019/MeshTreeHierarchyWithScopeNotes.txt",sep="\t",head=T,quote="",fill = T)
#this subcategories bellow are those related to human anatomic terms
to_keep <- c("A01","A02","A03","A04","A05","A06","A07","A08","A09","A010","A11","A12","A14","A015","A16","A17")
tissues <- as.data.frame(subset(tissues,MeSH.subcategories %in% to_keep))
tissues <- subset(tissues,nchar(as.character(Tree.Number))>3) # to take off categories head information
dim(tissues)
colnames(tissues)
#head(tissues)

tissues_red=tissues[c(3,5)]
tissues_red <- tissues_red[!duplicated(tissues_red),]
dim(tissues_red)

##################################################
#Creating a reordered MESH tissue ontology column#
##################################################

#That were done to reorder words inside the Mesh column "Terms". For exemple "Cranial Fossa, Anterior" became "Anterior
#Cranial Fossa" which is the correct term to be searched.

tissue_reorder <- function(var){
  var <- unlist(strsplit(var,", "))
  var_re <- rev(var)
  reordered_var <- paste0(var_re,collapse=" ")
  return(reordered_var)
}

#######
tissues_red$Term=as.character(tissues_red$Term)
tissues_red[3]="NA"
for (i in 1:dim(tissues_red)[1]){
  tissues_red[i,3]= tissue_reorder(tissues_red[i,2])
}
colnames(tissues_red)[3]="Reordered_Term"
dim(tissues_red)
tissues_red # Contains 1652 terms, that will be used to fich tissues from the GPL table.

####################################################################################
#Functions to grep tissue information from characteristics_ch, source, title column#
####################################################################################

#TO SAVE TISSUE TERM PRESENT ON GPL table, TO SAVE TISSUE TERM PRESENT ON MESH table, To save CODE term present on MeSH table.
#Function used to search for original terms called VAR - present on GPL table, and save it to further analysis.
test_mesh = function(var,tissue){ # this function permit that inside string chars will be the name of the original variable matched with tissues strings (tissues from MeSH tissue list) 
  meshterm = c()
  meshcode = c()
  original <- c()
  orig = 1
  c = 1
  for (j in 1:length(var)){
    for (k in 1:dim(tissue)[1]){
      separated_words = unlist(strsplit(as.character(tissue[k,2])," ")) ## separate words
      regex_string = paste("(?=.*\\b",paste(separated_words,collapse=")(?=.*\\b"),")",sep="") ## prepare for regex (case, space, etc)
      if (grepl(regex_string,var[j],ignore.case=T, perl=T)==T) { #paste("\\b",tissue[k,3],"\\b",sep="")
        meshterm[c] <- as.character(tissue[k,2])
        meshcode[c] = as.character(tissue[k,1])
        c = c + 1
        if (orig==1){
          original = var[j]
          orig = orig + 1
        }
      }
    }
  }
  
  
  
  meshterm <- paste(meshterm, collapse = ';')
  meshcode <- paste(meshcode, collapse=';')
  original <- paste(original, collapse=';')
  res = c(original, meshterm, meshcode)
  return (res)
}

###########################################################################
#Using functions to obtain VAR, TISSES an COD variables to each GPL column#
###########################################################################

ptm <- proc.time()

gpl[14:22] <- "NA"
for(i in 1:dim(gpl)[1]){ 
  var <- unlist(strsplit(as.character(gpl[i,7]),";")) # sample "characteristics_ch" column;
  query_mesh = test_mesh(var, tissues_red)
  gpl[i,14] = query_mesh[1]
  gpl[i,15] = query_mesh[2]
  gpl[i,16] = query_mesh[3]
  var = as.vector(gpl[i,6]) #sample "source_ch" column;
  query_mesh = test_mesh(var, tissues_red)
  gpl[i,17] = query_mesh[1]
  gpl[i,18] = query_mesh[2]
  gpl[i,19] = query_mesh[3]
  var = as.vector(gpl[i,1]) #sample "title" column.
  query_mesh = test_mesh(var, tissues_red)
  gpl[i,20] = query_mesh[1]
  gpl[i,21] = query_mesh[2]
  gpl[i,22] = query_mesh[3]
}

final <- proc.time() - ptm #to know how many time this process took.
#27k  user    system   elapsed 
#54436.574   202.869 68524.132 ~19 hours


#450k   user    system   elapsed 
#63485.582   234.043 73288.562  ~20 hours

#EPIC user  system elapsed 
#553.841   1.552 557.064 ~10 min

colnames(gpl)[14:22] <- c("MeSH_list_ch_var_col","MeSH_list_ch_tissue_col","MeSH_list_ch_cod_col",
                          "MeSH_list_source_var_col", "MeSH_list_source_tissue_col", "MeSH_list_source_cod_col",
                          "MeSH_list_title_var_col","MeSH_list_title_tissue_col","MeSH_list_title_cod_col")


##################################
#To merge var and tissues columns#
##################################

for(i in 1:dim(gpl)[1]){
  gpl[i,23]= paste(unique(unlist(strsplit(as.character(c(gpl[i,14],gpl[i,17],gpl[i,20])),";"))),collapse=";") #union of all var terms present in the GPL table
  gpl[i,24]= paste(unique(unlist(strsplit(as.character(c(gpl[i,15],gpl[i,18],gpl[i,21])),";"))),collapse=";") #union of all tissues terms present in the MeSH table
  gpl[i,25]= paste(unique(unlist(strsplit(as.character(c(gpl[i,16],gpl[i,19],gpl[i,22])),";"))),collapse=";") #union of all cod (D...) terms present in the MeSH table
}
colnames(gpl)[23:25] <- c("Merged_MeSH_list_ch_var_col","Merged_MeSH_list_ch_tissue_col","Merged_MeSH_list_ch_cod_col")
gpl[,23] <- gsub("^.*?:","",gpl[,23]) #clean everything before the colon
gpl=subset(gpl,!Merged_MeSH_list_ch_tissue_col=="") #to exclude GSM samples without tissue

###########################
# Fetal/Placental samples #
###########################

gpl[26] <- 'NA'
for(i in 1:dim(gpl)[1]){
    if (grepl("weeks",gpl[i,9],ignore.case=T)==T | grepl("months",gpl[i,9],ignore.case=T)==T | grepl("less than in years",gpl[i,9],ignore.case=T)==T){
    if (grepl("Placental",gpl[i,1],ignore.case=T)==T | grepl("Fetal",gpl[i,1],ignore.case=T)==T | grepl("Foetal",gpl[i,1],ignore.case=T)==T | grepl("Vill",gpl[i,1],ignore.case=T)==T |grepl("Placental",gpl[i,6],ignore.case=T)==T | grepl("Fetal",gpl[i,6],ignore.case=T)==T | grepl("Foetal",gpl[i,6],ignore.case=T)==T | grepl("Vill",gpl[i,6],ignore.case=T)==T){
      gpl[i,26] <- "Fetal sample"
    }
  }
}
colnames(gpl)[26] <- "Fetal_status"
unique(gpl$Fetal_status)
table(gpl$Fetal_status)

###############################
#Adding sex information column#
###############################
gpl[27] <- NA
c=1
terms <- c()
for(i in 1:dim(gpl)[1]){
  var <- unlist(strsplit(as.character(gpl[i,7]),";"))
  for(j in 1:length(var)){
    # \\b matches the empty string at either edge of a word
    if ( grepl('Gender', var[j],ignore.case = T) == T | grepl("Sex", var[j],ignore.case = T)==T) {
      if (grepl("dnamage",var[j],ignore.case=T)==F){
      gpl[i,27] <- as.character(var[j])
      terms[c] = strsplit(var[j],':')[[1]][1] #to verify if there is a term not involved with the key words search
      }
    }
  }
  c=c+1
  }

colnames(gpl)[27] <- c("Gender_status")

gpl$Gender_status <- as.character(gpl$Gender_status)
gpl[,27] <- gsub("^.*?: ","",gpl[,27]) #clean everything before the colon
gpl[,27] <- gsub("Female|female|FEMALE|f","F",gpl[,27]) #substitutes female terms by "F".
gpl[,27] <- gsub("Male|male|MALE|m","M",gpl[,27]) #substitutes male terms by "M".
gpl[,27] <- gsub("^\\s+|\\s+$","",gpl[,27]) #To trim leading and trailing whitespaces
gpl[,27] <- gsub("UNKNOWN|pool_na",NA,gpl[,27]) #substitutes male terms by "M".

########################
#Saving GPL final table#
########################
#write.table(gpl,"GPL8490_age_control_and_tissue_withSex_final.txt",sep="\t",row.names=F,quote=F)
#write.table(gpl,"GPL13534_age_control_and_tissue_withSex_final.txt",sep="\t",row.names=F,quote=F)
write.table(gpl,"Desktop/GEO_update_DEC2019/GPL21145_age_control_and_tissue_withSex_final.txt",sep="\t",row.names=F,quote=F)
