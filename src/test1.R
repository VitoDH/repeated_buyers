library(data.table)
library(plyr)
library(e1071)
library(randomForest)
setwd("~/Documents/Study/ComplexDataAnalysis/Homework/大作业/IJCAI15 Data/data_format1")
user_log <- fread("user_log_format1.csv",header = T)
user_info <- fread("user_info_format1.csv",header = T)
user_info <- na.omit(user_info)
trainSet <- fread("train_format1.csv",header = T);colnames(trainSet)[2]<-"seller_id"
trainSet <- trainSet[1:10000,]
testSet <- fread("test_format1.csv",header = T);colnames(testSet)[2]<-"seller_id"
testSet <- testSet[1:1000,]
trainLength <- nrow(trainSet)
testLength <- nrow(testSet)


#-----------------------------------------------------Direct Connection
#将用户行为与训练集，测试集合并Ó
train <- merge(user_log,trainSet,by=c("user_id","seller_id"))
train <- merge(train,user_info,by=c("user_id"))
#test <- merge(user_log,testSet,by=c("user_id","seller_id"))#test <- user_log[(user_log$user_id %in% testSet$user_id) & (user_log$seller_id %in% testSet$merchant_id),]


sgbTrain <- split(train,by=c("user_id","seller_id"))          #split group by train
#sgbTest <- split(test,by=c("user_id","seller_id")) 


classifyActionType <- function(tempTable){      #导入要对action type进行分类统计的
  act_type <- as.data.frame(table(tempTable$action_type))
  rownum <- nrow(act_type)
  act_type <- as.numeric(as.matrix(act_type))   
  act_type <- matrix(act_type,nrow = rownum)
  tmp1 <- setdiff(0:3,act_type[,1])      #act_type第一列是action_type种类，第二列是对应的频数
  if(length(tmp1)==0){tmp3 <- act_type   #若包含了所有action，则不用修改，直接赋值
  }else{                                 #若不然，补全数据框，使之包含四种action
    tmp2 <- cbind(tmp1,0)
    tmp3 <- rbind(act_type,tmp2) 
  }
  sortVec <- tmp3[order(tmp3[,1]),][,2]   #重新对一列排序，使之按照点击、加入购物车、购买、喜欢的顺序
  return(sortVec)
}



direct_link <-function(tableTrain){
  directActionVec <- classifyActionType(tableTrain)
  label <- as.numeric(tableTrain[1,8])      #提取标签
  age_range <- as.numeric(tableTrain[1,9])
  gender <-as.numeric(tableTrain[1,10])
  directVec <- c(directActionVec,age_range,gender,label)
  return(directVec)
}
pp<-direct_link(sgbTrain[[5]])

direct_linkVec <- lapply(sgbTrain, direct_link)     #得到用户-商家的直接汇总记录
list2Vec <- unlist(direct_linkVec)
Vec2Mat <- matrix(list2Vec,ncol = 7,byrow = T)
colnames(Vec2Mat) <- c("click","cart","buy","favorite","age_range","gender","label")

#direct_linkVecTest <- lapply(sgbTest, direct_link)
#list2VecTest <- unlist(direct_linkVecTest)
#Vec2MatTest <- matrix(list2VecTest,ncol = 5,byrow = T)
#colnames(Vec2MatTest) <- c("click","cart","buy","favorite","label")


trainMat <- as.data.frame(Vec2Mat)
trainMat$label <- as.factor(trainMat$label)
trainMat1 <-trainMat[1:8000,]      #切分成训练和测试看效果
testMat1 <- trainMat[8001:9854,]   #删掉了有缺失值（年龄，性别）的用户，所以不够10000个


#testMat <- as.data.frame(Vec2MatTest)
#testMat$label <- as.factor(testMat$label)

#--------------svm
x <- subset(trainMat1, select = -label)
y <- trainMat1$label
model <- svm(x, y)
print(model)
summary(model)
x <- subset(testMat1, select = -label)
y <- testMat1$label
pred <- predict(model, x)

# Check accuracy:
table(pred, y)

#--------------randomForest
model.forest <- randomForest(label~.,data = trainMat1)
pre.forest <- predict(model.forest,testMat1)
table(pre.forest,testMat1$label)




#--------------------------------------------------Indirect Connection
clusterSet <- na.omit(user_log[1:5100000,])[1:5000000,]
clusterSetUser <- merge(clusterSet,user_info,by=c("user_id"))  
clusterSetSeller <- clusterSet

catName <- as.character(sort(unique(clusterSet$cat_id)))       #提取所有商品种类名
brandName <- as.character(sort(unique(clusterSet$brand_id)))   #提取所有品牌名


clusterUser <- split(clusterSetUser,by="user_id")          #为用户建立数据库
clusterSeller <- split(clusterSetSeller,by="seller_id")    #为商家建立数据库


weighted.mean(c(1,2),c(0.4,0.6))


BuildUserfeatVec <- function(userList){        #------------用于对clusterUser的lapply,对整个List loop一遍非常慢！
  
  #-----------------obtain the the vector of catagories 
  userList <- userList[order(cat_id)]       #对商品种类按数字进行排序
  catSplit <- split(userList,by = "cat_id")      #对某用户根据其浏览过的商品种类来分类
  catAction <- lapply(catSplit,classifyActionType)  #返回某个用户的关于商品种类的基础特征向量(未加权)，形式为列表
  catActionWeightedMean <- unlist(lapply(catAction,weighted.mean,w= c(0.1,0.2,0.3,0.4))) #用户-种类的加权特征向量
  names(catActionWeightedMean) <- paste0(names(catActionWeightedMean),"-cat")
  
  #-----------------obtain the vector of brands
  userList <- userList[order(brand_id)]       #对商品种类按数字进行排序
  brandSplit <- split(userList,by = "brand_id")
  brandAction <- lapply(brandSplit,classifyActionType)  #返回某个用户的关于商品种类的基础特征向量(未加权)，形式为列表
  brandActionWeightedMean <- unlist(lapply(brandAction,weighted.mean,w= c(0.1,0.2,0.3,0.4))) #用户-种类的加权特征向量
  names(brandActionWeightedMean) <- paste0(names(brandActionWeightedMean),"-brand")
  
  UserVec <- c(catActionWeightedMean,brandActionWeightedMean)
  
}


BuildSellerfearVec <- function(sellList){
  
}

