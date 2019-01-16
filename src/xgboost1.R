library(xgboost)
library(data.table)
library(pROC)


data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test

bst <- xgboost(data = train$data, label = train$label, max.depth = 2, eta = 1,nround = 2, objective = "binary:logistic")

pred <- predict(bst, test$data)
modelrocTrain <- roc(test$label,pred)
plot(modelrocTrain, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

cv.res <- xgb.cv(data = train$data, label = train$label, max.depth = 2, eta = 1, nround = 2, 
                 objective = "binary:logistic", nfold = 5)

cv.res


#--------------------------
trainSet <- read.csv("trainSet2.csv")
trainSet <- data.matrix(trainSet)

mapminmax <- function(x){
  minx <- min(x)
  maxx <- max(x)
  x <- (x-minx)/(maxx-minx)
}
#3,7,9:12
trainSet[,3:12] <- apply(trainSet[,3:12],2,mapminmax)

#trainSet[,7:10] <- scale(trainSet[,7:10])
pca <- princomp(trainSet[,3:12])
summary(pca)
trainSet2 <- pca$scores[,1:3]
trainSet <- cbind(trainSet2,trainSet[,13])

train <- trainSet[1:60000,]
test <- trainSet[60001:65535,]

#trainX <- train[,3:12];trainY <- train[,13] 
#testX <- test[,3:12]; testY <- test[,13]

trainX <- train[,1:3];trainY <- train[,4] 
testX <- test[,1:3]; testY <- test[,4]


bst <- xgboost(data = trainX, label = trainY, max.depth = 2, eta = 1,nround = 500, objective = "binary:logistic")

predTrain <- predict(bst,trainX)
modelrocTrain <- roc(trainY,predTrain)
plot(modelrocTrain, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

predTest <- predict(bst,testX)
modelrocTest <- roc(testY,predTest)
plot(modelrocTest, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)





cv.res <- xgb.cv(data = trainX, label = trainY, max.depth = 4, eta = 1,nround = 1000, objective = "binary:logistic",nfold = 5)
 cv.res



