library(ggplot2)
library(data.table)
library(reshape2)

#Multiple plot function  
#  
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)  
# - cols:   Number of columns in layout  
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.  
#  
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),  
# then plot 1 will go in the upper left, 2 will go in the upper right, and  
# 3 will go all the way across the bottom.  
#  
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {  
  library(grid)  
  
  # Make a list from the ... arguments and plotlist  
  plots <- c(list(...), plotlist)  
  
  numPlots = length(plots)  
  
  # If layout is NULL, then use 'cols' to determine layout  
  if (is.null(layout)) {  
    # Make the panel  
    # ncol: Number of columns of plots  
    # nrow: Number of rows needed, calculated from # of cols  
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),  
                     ncol = cols, nrow = ceiling(numPlots/cols))  
  }  
  
  if (numPlots==1) {  
    print(plots[[1]])  
    
  } else {  
    # Set up the page  
    grid.newpage()  
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))  
    
    # Make each plot, in the correct location  
    for (i in 1:numPlots) {  
      # Get the i,j matrix positions of the regions that contain this subplot  
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))  
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,  
                                      layout.pos.col = matchidx$col))  
    }  
  }  
}  




train <- fread('trainSet3.csv') 
index <- 1:nrow(train)
tmp <- data.frame(index,train$click,train$cart,train$buy,train$favorite)
colnames(tmp) <- c("index","click","cart","buy","favorite")


p1 <- ggplot(data = tmp, mapping = aes(x = index, y = click, colour = click)) + geom_point(size = 3) + scale_colour_gradient(low = 'lightblue', high = 'darkblue')
p2 <- ggplot(data = tmp, mapping = aes(x = index, y = cart, colour = cart)) + geom_point(size = 3) + scale_colour_gradient(low = 'lightblue', high = 'darkblue')
p3 <- ggplot(data = tmp, mapping = aes(x = index, y = buy, colour = buy)) + geom_point(size = 3) + scale_colour_gradient(low = 'lightblue', high = 'darkblue')
p4 <- ggplot(data = tmp, mapping = aes(x = index, y = favorite, colour = favorite)) + geom_point(size = 3) + scale_colour_gradient(low = 'lightblue', high = 'darkblue')

layout <- matrix(rep(1:4,each=2), nrow = 2, byrow = TRUE)  
multiplot(plotlist = list(p1, p2, p3,p4), layout = layout)  




trainSet2 <- train[train$click<1000,]
trainSet2 <- trainSet2[trainSet2$favorite<40,]

boxData <- melt(tmp,id=1)
p11 <-  ggplot(data = boxData, mapping = aes(x =variable , y = value)) + geom_boxplot(aes(fill=variable)) 
print(p11)

logitfit <- glm(label~.,family=binomial(link='logit'),data=data_balanced_under)
pred.test <- predict(logitfit,type='response',newdata = test)

modelrocTest <- roc(testY,pred.test)
plot(modelrocTest, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
accuracy.meas(testY, pred.test,threshold = 0.501)


