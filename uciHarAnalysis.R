#' # Analysis of UCI HAR Dataset
#' 
#' Syrus Nemat-Nasser (R@syrus.us)
#' 
#' 2013-11

#+ setup, include=FALSE
library(knitr)
opts_chunk$set(fig.path='figure/uciHar-', dev='svg', fig.width=8, fig.height=4, dpi=128)

#' We will analyze a human activity recognition data set available from the UCI 
#' Machine Learning Repository. The data is downloaded and uncompressed locally.
#' 
#' ## Load required packages

library(ggplot2)

#+ data-ingress, echo=FALSE
ProjectDirectory = getwd()
DataDirectory = "UCI HAR Dataset/"
if (!file.exists(DataDirectory)) {
    download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
                  , "data.zip", "curl", quiet=TRUE, mode="wb")
    unzip("data.zip")
    file.remove("data.zip")
}
stopifnot(file.exists(DataDirectory))
setwd(DataDirectory)
temp = read.table("activity_labels.txt", sep="")
activityLabels = as.character(temp$V2)
temp = read.table("features.txt", sep="")
attributeNames = temp$V2
Xtrain = read.table("train/X_train.txt", sep="")
names(Xtrain) = attributeNames
Ytrain = read.table("train/y_train.txt", sep="")
names(Ytrain) = "Activity"
Ytrain$Activity = as.factor(Ytrain$Activity)
levels(Ytrain$Activity) = activityLabels
trainSubjects = read.table("train/subject_train.txt", sep="")
names(trainSubjects) = "subject"
trainSubjects$subject = as.factor(trainSubjects$subject)
train = cbind(Xtrain, trainSubjects, Ytrain)
setwd(ProjectDirectory)
