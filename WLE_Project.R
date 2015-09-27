library(caret)
library(rpart)
library(randomForest)
library(parallel)
library(doParallel)
library(rattle)

TrainData<-read.csv('pml-training.csv',header=T,na.strings=c("NA","#DIV/0!",""))
namesTrain<-colnames(TrainData)
TestData<-read.csv('pml-testing.csv',header=T,na.strings=c("NA","#DIV/0!",""))
namesTest<-colnames(TestData)

#Cleaning Data
noNA<-as.vector(apply(TrainData, 2, function(x) length(which(!is.na(x)))))
coldrop<-c()
for(i in 1:length(noNA))
{
   if(noNA[i]<nrow(TrainData))
   {
      coldrop<-c(coldrop,namesTrain[i])
   }
}

TrainData<-TrainData[,!(namesTrain %in% coldrop)]
TrainData<-TrainData[,8:length(names(TrainData))]

TestData<-TestData[,!(namesTest %in% coldrop)]
TestData<-TestData[,8:length(names(TestData))]

nearZeroVar(TrainData,saveMetrics=T)

#Data Partition
set.seed(201509)
inTrain<-createDataPartition(TrainData$classe,p=0.7,list=FALSE)
Training<-TrainData[inTrain,]
Testing<-TrainData[-inTrain,]

qplot(classe,data=Training,geom="bar",fill=Training$classe)
registerDoParallel(makeCluster(4))

#Tree Model
set.seed(201509)
model1<-train(classe~.,data=Training,method='rpart')
model1
pred1<-predict(model1,newdata=Testing)
plot(model1$finalModel)
text(model1$finalModel,cex=.7)
confusionMatrix(pred1,Testing$classe)

#Random Forest Model
set.seed(201509)
model2<-train(classe~.,data=Training,method='rf',trControl=trainControl(method='cv',number=3))
model2
pred2<-predict(model2,newdata=Testing)
plot(model2)
plot(model2$finalModel,main='Random Forest Model')
varImpPlot(model2$finalModel,cex=.7,main='Random Forest Model',pch=19)
confusionMatrix(pred2,Testing$classe)

#Out Sample Error
list(OutSampleError_Tree=1-confusionMatrix(pred1,Testing$classe)$overall[[1]],
     Predict_Tree=predict(model1,newdata=TestData),
     OutSampleError_RForest=1-confusionMatrix(pred2,Testing$classe)$overall[[1]],
     Predict_RForest=predict(model2,newdata=TestData))

#Create .txt files with predictions
prediction=predict(model2,newdata=TestData)
for (i in seq(20)){
   fileName<-paste("problem",i,".txt",sep="_")
   write.table(prediction[i],file=fileName,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
