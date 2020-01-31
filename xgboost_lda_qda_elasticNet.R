library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

# I. Load Data

train_rm <- read.csv("train_rm.csv")
test <- read.csv("test.csv")

# Response Variable
table(train_rm$HTWins)
## Code Response Variable into 0/1 values
train_rm$HTwins_01 <- numeric(nrow(train_rm))
train_rm$HTwins_01[train_rm$HTWins == 'Yes'] <- 1
train_rm$HTwins_01[train_rm$HTWins == 'No'] <- 0


# Feature Engineering -- Construct totals
train_rm$VT.OS.plmin.total <- train_rm$VT.OS1.plmin + train_rm$VT.OS2.plmin + train_rm$VT.OS3.plmin + train_rm$VT.OS4.plmin + train_rm$VT.OS5.plmin
test$VT.OS.plmin.total <- test$VT.OS1.plmin + test$VT.OS2.plmin + test$VT.OS3.plmin + test$VT.OS4.plmin + test$VT.OS5.plmin

train_rm$VT.S.plmin.total <- train_rm$VT.S1.plmin + train_rm$VT.S2.plmin + train_rm$VT.S3.plmin + train_rm$VT.S4.plmin + train_rm$VT.S5.plmin
test$VT.S.plmin.total <- test$VT.S1.plmin + test$VT.S2.plmin + test$VT.S3.plmin + test$VT.S4.plmin + test$VT.S5.plmin

train_rm$HT.total.pts <- train_rm$HT.S1.pts + train_rm$HT.S2.pts + train_rm$HT.S3.pts + train_rm$HT.S4.pts + train_rm$HT.S5.pts
test$HT.total.pts <- test$HT.S1.pts + test$HT.S2.pts + test$HT.S3.pts + test$HT.S4.pts + test$HT.S5.pts
 
 
features_selected <- c('VT.OS.plmin.total', 'VT.S.plmin.total', 'VT.OTA.ast', 'VT.TA.ast', 'VT.OTA.pts',  'VT.OTS.pts', 'VT.OTA.fgm', 'VT.OTS.ast', 'HT.total.pts')
train_rm_combined <- train_rm[,c('HTwins_01',features_selected)] 

test_combined <- test[,c(features_selected)] 


# Xgboost
 
library(xgboost)
a = rep(0, 50)
for (i in 1:50){
  dev.ind <- sample(1:9520, 1000)
  devset <- train_rm_combined[dev.ind, ]
  trainset <- train_rm_combined[-dev.ind, ]
  
  xgb.fit <- xgboost(data = as.matrix(trainset[,-1]), 
                     label = trainset[,1], 
                     eta = 0.1,
                     max_depth = 1, 
                     nround=20, 
                     base_score = 0.6,
                     subsample = 0.5,
                     num_parallel_tree = 5,
                     colsample_bytree = 1,
                     round = 1,
                     eval_metric = "rmse",
                     objective = "binary:logistic",
                     nthread = 5,
                     verbose = 0,
                     lambda = 0.01
  )
  train_pred <- predict(xgb.fit, data.matrix(devset[,-1]))
  train_pred_01 <- ifelse(train_pred > 0.5, 1, 0)
  
  a[i] = sum(train_pred_01 == devset$HTwins_01) / 1000
}

mean(a)
 

# predict values in test set
train_pred <- predict(xgb.fit, data.matrix(devset[,-1]))
train_pred_01 <- ifelse(train_pred > 0.5, 1, 0)
confusionMatrix(as.factor(train_pred_01), as.factor(devset$HTwins_01))
 
test_pred <- predict(xgb.fit, data.matrix(test_combined))
table(test_pred)

test12_YN <- as.factor(ifelse(test_pred == 1, 'Yes', 'No'))
test12_output <- data.frame('id' = test$id, 'HTWins' = test12_YN)
write.csv(test12_output, "test12.csv", row.names = FALSE)
 



 
data = as.matrix(train_rm_combined[,-1])
dim(data)

 
library(glmnet)
library(MASS)

dev.ind <- sample(1:9520, 1000, replace = F)
devset <- train_rm_combined[dev.ind, ]
trainset <- train_rm_combined[-dev.ind, ]

train.dat.mat <- as.matrix(trainset[,-1])
dev.dat.mat <- data.matrix(devset[,-1])
test.mat <- data.matrix(test_combined)

ridge_cv_fit = cv.glmnet(train.dat.mat, trainset[,1], alpha = 0)

print("ridge")
bestlambda.ridge = ridge_cv_fit$lambda.min
dev.ridge.result = predict(ridge_cv_fit, s = bestlambda.ridge, newx = dev.dat.mat) 
dev_pred_01 <- ifelse(dev.ridge.result > 0.5, 1, 0)
dev_acc <- sum(dev_pred_01 == devset$HTwins_01) / 1000
dev_acc

train.ridge.result = predict(ridge_cv_fit, s = bestlambda.ridge, newx = train.dat.mat) 
train_pred_01 <- ifelse(train.ridge.result > 0.5, 1, 0)
train_acc <- sum(train_pred_01 == trainset$HTwins_01) / nrow(trainset)
train_acc

print("lda")
lda.fit=lda(HTwins_01~.,data=trainset)
train.lda.result = predict(lda.fit, type = 'response')$class
train_acc <- sum(train.lda.result == trainset$HTwins_01) / nrow(trainset)
train_acc

dev.lda.result = predict(lda.fit, newdata = devset[,-1])$class
dev_acc <- sum(dev.lda.result == devset$HTwins_01) / nrow(devset)
dev_acc

print('qda')
qda.fit=qda(HTwins_01~.,data=trainset)
train.qda.result = predict(qda.fit, type = 'response')$class
train_acc <- sum(train.qda.result == trainset$HTwins_01) / nrow(trainset)
train_acc

dev.qda.result = predict(qda.fit, newdata = devset[,-1])$class
dev_acc <- sum(dev.qda.result == devset$HTwins_01) / nrow(devset)
dev_acc
 





 
# Numeric predictors & the response variable
train_rm_double <- train_rm[, sapply(train, typeof) == 'double']
cor_all <- cor(train_rm_double)
corrplot.mixed(cor_all,lower = "number", upper = "pie")
 

# Elastic Net
 
train_rm.samples <- train_rm$HTwins_01 %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- train_rm[train_rm.samples, ]
test.data <- train_rm[-train_rm.samples, ]

x <- model.matrix(HTwins_01~ VT.OS.plmin.total + VT.S.plmin.total + VT.OTA.ast + VT.TA.ast + VT.OTA.pts + VT.OTS.pts + VT.OTA.fgm  + VT.OTS.ast + HT.total.pts, train.data)[,-1]
y <- train.data$HTwins_01

set.seed(123)
model <- train(
  HTwins_01~ VT.OS.plmin.total + VT.S.plmin.total + VT.OTA.ast + VT.TA.ast + VT.OTA.pts + VT.OTS.pts + VT.OTA.fgm  + VT.OTS.ast + HT.total.pts, data = train.data, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
# Best tuning parameter
model$bestTune
 

 
glm4 <- glmnet(x, y, alpha = 0.1, lambda = 0.04161097)
coef(glm4)
train.x <- model.matrix(HTwins_01~ VT.OS.plmin.total + VT.S.plmin.total + VT.OTA.ast + VT.TA.ast + VT.OTA.pts + VT.OTS.pts + VT.OTA.fgm  + VT.OTS.ast + HT.total.pts, train_rm)[,-1]
glm4_predict <- glm4 %>% predict(train.x)
glm4_01 <- ifelse(glm4_predict > 0.5, 1, 0)
confusionMatrix(as.factor(glm4_01), as.factor(train_rm$HTwins_01))
 

# Output
 
testx = as.matrix(test[,c('VT.OS.plmin.total', 'VT.S.plmin.total', 'VT.OTA.ast', 'VT.TA.ast', 'VT.OTA.pts',  'VT.OTS.pts', 'VT.OTA.fgm', 'VT.OTS.ast', 'HT.total.pts')])
test9 <- predict(glm4, s =  0.04161097, newx= testx, type = 'response')
test9_01 <- ifelse(test9 > 0.5, 1, 0)
test9_YN <- as.factor(ifelse(test9_01 == 1, 'Yes', 'No'))
test9_output <- data.frame('id' = test$id, 'HTWins' = test9_YN)
write.csv(test9_output, "test9.csv", row.names = FALSE)
 


