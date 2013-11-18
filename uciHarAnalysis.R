#' # Analysis of UCI HAR Dataset
#' 
#' Syrus Nemat-Nasser (R [at] syrus [dot] us)
#' 
#' 2013-11

#+ setup, include=FALSE
library(knitr)
opts_chunk$set(fig.path='figure/uciHar-', dev='png', fig.width=8.5, fig.height=4, dpi=96, comment=NA)

#' We will analyze a human activity recognition data set available from the UCI 
#' Machine Learning Repository. The data is downloaded and uncompressed locally.
#' 
#' ## Data Ingress

#+ data-ingress, echo=TRUE
ProjectDirectory = getwd()
DataDirectory = "UCI HAR Dataset/"
dataFile = "dataset.RData"
if (!file.exists(DataDirectory)) {
    download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
                  , "data.zip", "curl", quiet=TRUE, mode="wb")
    unzip("data.zip")
    file.remove("data.zip")
}
stopifnot(file.exists(DataDirectory))
setwd(DataDirectory)
if (!file.exists(dataFile)) {
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
    
    Xtest = read.table("test/X_test.txt", sep="")
    names(Xtest) = attributeNames
    Ytest = read.table("test/y_test.txt", sep="")
    names(Ytest) = "Activity"
    Ytest$Activity = as.factor(Ytest$Activity)
    levels(Ytest$Activity) = activityLabels
    testSubjects = read.table("test/subject_test.txt", sep="")
    names(testSubjects) = "subject"
    testSubjects$subject = as.factor(testSubjects$subject)
    test = cbind(Xtest, testSubjects, Ytest)
    
    save(train, test, file=dataFile)
    rm(train,test,temp,Ytrain,Ytest,Xtrain,Xtest,trainSubjects,testSubjects,activityLabels,attributeNames)
}
load(dataFile)
setwd(ProjectDirectory)
numPredictors = ncol(train) - 2

#' ## Data Summary
#' The data comes to us partitioned by human subject with 9 subjects held out in *test*. We
#' will respect this partition of the data and use these subjects as our strict hold out sample.
#' 
#+ local options, include=FALSE
options(width=90) # prevent longer lines from wrapping
library(ggplot2)
library(plyr)
library(reshape2)
library(caret)
#+ data summary, echo=TRUE
summary(train$subject)
summary(test$subject)
train$Partition="Train"
test$Partition = "Test"
all = rbind(train,test) # combine sets for visualization
all$Partition = as.factor(all$Partition)
qplot(data=all, x=subject, fill=Partition)
qplot(data=all, x=subject, fill=Activity)
rm(all) # recover memory
#' We observe that the distribution of examples is fairly evently distributed
#' accross experimental subjects and activity types.
#' 
#' The README file indicates that the predictor variables are normalized and scaled
#' to the range [-1,1]. We can perform a basic check to compare the distribution of
#' each predictor to a normal distribution.
trainSd = colwise(sd)(train[,1:numPredictors])
trainSd$stat = "Predictor Variable Standard Deviation"
trainMean = colwise(mean)(train[,1:numPredictors])
trainMean$stat = "Predictor Variable Mean"
temp = melt(rbind(trainMean, trainSd), c("stat"))
qplot(data=temp, x=value, binwidth = 0.025) + facet_wrap(~ stat, ncol=1)
rm(temp,trainMean,trainSd)
#' If each variable was z-scaled, the mean would be approximately zero and the standard deviation
#' would be 1. These variables may be *normalized*, but they are not z-scaled. If we intend to use
#' modeling methods that are sensitive to feature scaling, we might want to do some preprocessing.

#' ## Preprocessing
#' Caret offers several options for preprocessing continuous variables such as the predictors in the UCI HAR
#' dataset. We will prepare several different versions of the predictor matrix to compare how these perform
#' when we build a predictive model.
#' 
#' ### Z-scaling
zScaleTrain = preProcess(train[,1:numPredictors])
scaledX = predict(zScaleTrain, train[,1:numPredictors])
head(names(scaledX))
#' ### Near Zero Variance Predictor Detection
nzv = nearZeroVar(scaledX, saveMetrics=TRUE)
summary(nzv)
head(nzv[order(nzv$percentUnique, decreasing=FALSE),], n=20)
head(nzv[order(nzv$freqRatio, decreasing=TRUE),], n=20)
#' ### Find and Remove Highly Correlated Predictors
correlatedPredictors = findCorrelation(cor(scaledX), cutoff=0.95)
#' There are `r length(correlatedPredictors)` correlated predictors to remove.
reducedCorrelationX = scaledX[,-correlatedPredictors]
head(names(reducedCorrelationX))
#' The reduced correlation predictor set retains `r ncol(reducedCorrelationX)` variables.
#' ### PCA Transformed Predictors
pcaTrain = preProcess(scaledX, method="pca", thresh=0.99)
#' The PCA transformed data retains `r pcaTrain$numComp` components to capture `r 100*pcaTrain$thresh`% of the variance.
pcaX = predict(pcaTrain, scaledX)
head(names(pcaX))
#' After PCA, the original predictor names are no longer available in a straightforward manner.

#' ## Data Splitting
#' An important aspect of predictive modeling is ensuring that we can accurately predict model
#' performance on unseen data; when we deploy our model, our reputation and that of our employer
#' are often impacted by how our model performs. For the UCI HAR data, our *test* set is a strict
#' hold-out sample. We will not use this set for model development or model selection.
#' 
#' Our training data set is not tiny, but neither is it large. We have data examples from a limited
#' number of experimental subjects. My technical approach will be to use cross-validation by experimental
#' subject for model selection. There are `r length(levels(train$subject))` training subjects. If we
#' wish to cross-validate over every experimental subject, we can generate sets of training sample indices
#' like this:
leaveOneSubjectOutIndices = lapply(levels(train$subject), function(X) {which(!X==train$subject)})
#' If instead we want to control computation time, we can create a different partition.
cvBreaks = 7
temp = sample(levels(train$subject), length(levels(train$subject))) # randomize subjects
temp = split(temp, cut(1:length(temp), breaks=cvBreaks, labels=FALSE)) # split into CV groups
cvGroupIndices = lapply(temp, function(X) {which(!train$subject %in% X)})

#' ## Model Training
#' ### Set up parallel processing
library(parallel)
cl = parallel::makeForkCluster(nnodes=detectCores()/2)
setDefaultCluster(cl)
library(doParallel)
registerDoParallel(cl)
getDoParWorkers()
#' ### Train a Random Forest model using method='rf'
saveFile = paste(DataDirectory, "modelRF.RData", sep='')
if (!file.exists(saveFile)) {
    rfCtrl = trainControl(method="cv", number=length(cvGroupIndices), index=cvGroupIndices, classProbs=TRUE)
    modelRF = train(reducedCorrelationX, train$Activity, method="rf", trControl=rfCtrl
                    , tuneGrid = data.frame(.mtry = c(2,5,10,15,20)), importance=TRUE)
    save(rfCtrl, modelRF, correlatedPredictors, zScaleTrain, file=saveFile)
}
if (!exists("modelRF")) { load(saveFile) }
print(modelRF)
plot(modelRF)
print(confusionMatrix(modelRF))
m = as.data.frame(modelRF$finalModel$importance)
m = m[order(m$MeanDecreaseAccuracy, decreasing=TRUE),]
head(m, n=20)
#' ### Train a Random Forest model using method='parRF'
saveFile = paste(DataDirectory, "modelParRF.RData", sep='')
if (!file.exists(saveFile)) {
    parRfCtrl = trainControl(method="cv", number=length(cvGroupIndices), index=cvGroupIndices, classProbs=TRUE)
    modelParRF = train(reducedCorrelationX, train$Activity, method="parRF", trControl=parRfCtrl
                       , tuneGrid = data.frame(.mtry = c(2,5,10,15,20)), importance=TRUE)
    save(parRfCtrl, modelParRF, correlatedPredictors, zScaleTrain, file=saveFile)
}
if (!exists("modelParRF")) { load(saveFile) }
print(modelParRF)
plot(modelParRF)
print(confusionMatrix(modelParRF))
#' ### Train using a simpler model
saveFile = paste(DataDirectory, "modelKnn.RData", sep='')
if (!file.exists(saveFile)) {
    knnCtrl = trainControl(method="cv", number=length(cvGroupIndices), index=cvGroupIndices, classProbs=TRUE)
    modelKnn = train(reducedCorrelationX, train$Activity, method="knn", trControl=knnCtrl
                     , tuneGrid = data.frame(.k = c(5,10,15,20)))
    save(knnCtrl, modelKnn, correlatedPredictors, zScaleTrain, file=saveFile)
}
if (!exists("modelKnn")) { load(saveFile) }
print(modelKnn)
confusionMatrix(modelKnn)
#' ### Selecting the "best" model
#' For simplicity, we will choose the *best* model based on overall cross-validation accuracy
#' which leaves us with one of the random forest models.
bestModel = modelRF
#' ### Predicting generalization performance
holdoutX = predict(zScaleTrain, test[,1: numPredictors])[,-correlatedPredictors]
holdoutLabels = test$Activity
holdoutPrediction = predict(bestModel, holdoutX)
head(holdoutPrediction)
holdoutConfusionMatrix = confusionMatrix(holdoutPrediction, holdoutLabels)
print(holdoutConfusionMatrix)
#' #### Comparison of holdout predictions and cross-validation predictions
print(confusionMatrix(bestModel), digits=2)
print(100 * (holdoutConfusionMatrix$table / sum(holdoutConfusionMatrix$table)), digits=1)
#' Finally, we will save an image of all workspace objects so we can use them in the presentation.
save.image(file="workspaceImage.RData")
