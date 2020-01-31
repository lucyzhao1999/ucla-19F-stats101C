library(corrplot)
library(caret)
library(ggplot2)
setwd("~/Desktop/101cFinalProject")
train <- read.csv("train.csv")
train$HTwins_01 <- numeric(nrow(train))
train$HTwins_01[train$HTWins == 'Yes'] <- 1
train$HTwins_01[train$HTWins == 'No'] <- 0

# get difference data from original train data: take difference between VT.***** and HT.*****
names(train)[9] <- "VT.cumRest"
names(train)[10] <- "HT.cumRest"

VTallvars <- sort(names(train)[grepl("VT",  names(train))])
HTallvars <- sort(names(train)[grepl("HT",  names(train))])

diffData <- train[, c(3,4,5,6,7)]

for (i in 2:(length(VTallvars)-1)){
  newColName <- substring(VTallvars[i], 4)
  diffData[newColName] <- train[HTallvars[i]] - train[VTallvars[i]]
}

diffData$HTWins = ifelse(train$HTWins == 'Yes', 1, 0)
diffData_double <- diffData[, sapply(diffData, typeof) == 'double']

cor_all <- cor(diffData_double)
sort(cor_all['HTWins',])[1:10]

topbottom10 <- diffData_double[,c("HTWins",names(sort(cor_all['HTWins',])[1:10]), names(sort(cor_all['HTWins',])[(ncol(cor_all) - 10): (ncol(cor_all) - 1)]))]

glm1 <- glm(HTWins ~ ., data = topbottom10, family = 'binomial')
summary(glm1)
glm1_predict <- predict(glm1,type = "response")
glm1_01 <- ifelse(glm1_predict > 0.5, 1, 0)
confusionMatrix(as.factor(glm1_01), as.factor(train$HTwins_01))


# match format of test data (get difference scores)

test <- read.csv("test.csv")
names(test)[8] <- "VT.cumRest"
names(test)[9] <- "HT.cumRest"

VTallvars_test <- sort(names(test)[grepl("VT",  names(test))])
HTallvars_test <- sort(names(test)[grepl("HT",  names(test))])


diffData_test <- test[, c(3, 4,5,6)]
for (i in 2:(length(VTallvars_test)-1)){
  newColName <- substring(VTallvars_test[i], 4)
  diffData_test[newColName] <- test[HTallvars_test[i]] - test[VTallvars_test[i]]
}
diffData_test_double <- diffData_test[, sapply(diffData_test, typeof) == 'double']


test1 <- predict(glm1, diffData_test_double, type = "response")
test1_01 <- ifelse(test1 > 0.5, 1, 0)
test1_YN <- as.factor(ifelse(test1_01 == 1, 'Yes', 'No'))
test1_output <- data.frame('id' = test$id, 'HTWins' = test1_YN)
write.csv(test1_output, "test4.csv", row.names = FALSE)

# problem: some of the predictors are same except positive/negative (sum = 0), thus model 1 is not actually
# using all 20 predictors



################11.30
diffData_double['OS.plmin'] = diffData_double$OS1.plmin + diffData_double$OS2.plmin + diffData_double$OS3.plmin + diffData_double$OS4.plmin + diffData_double$OS5.plmin

sort(cor_all['HTWins',])[1:20]
sort(cor_all['HTWins',])[(ncol(cor_all) - 20): (ncol(cor_all) - 1)]

# pick predictors without duplicates (based on correlation)
# deleted predictors that are highly insignificant
glm2 <- glm(HTWins ~ OS.plmin + TA.ast + TA.pts + OTS.pts + TA.fgm + OTS.ast + TA.dreb + OTS.tpm  + OTS.blk + OTS.fgm + TA.fta+ OTS.tpa + OS1.fgm 
            +S1.pts +  S3.pts +  S2.pts + S4.pts + OS3.dreb , data = diffData_double, family = 'binomial')

summary(glm2)
glm2_predict <- predict(glm2,type = "response")
glm2_01 <- ifelse(glm2_predict > 0.5, 1, 0)
confusionMatrix(as.factor(glm2_01), as.factor(train$HTwins_01))

# test
diffData_test_double['OS.plmin'] = diffData_test_double$OS1.plmin + diffData_test_double$OS2.plmin + diffData_test_double$OS3.plmin + diffData_test_double$OS4.plmin + diffData_test_double$OS5.plmin

test2 <- predict(glm2, diffData_test_double, type = "response")
test2_01 <- ifelse(test2 > 0.5, 1, 0)
test2_YN <- as.factor(ifelse(test2_01 == 1, 'Yes', 'No'))
test2_output <- data.frame('id' = test$id, 'HTWins' = test2_YN)
write.csv(test2_output, "test7.csv", row.names = FALSE)

# Accuracy Result: 65.533

################################################################################################################
# try scaling predictors to be normally distributed

glm2_dat = diffData_double[, c("OS.plmin", "TA.ast", "TA.pts","OTS.pts", "TA.fgm", "OTS.ast", "TA.dreb", "OTS.tpm", "OTS.blk",
                               "OTS.fgm", "TA.fta", "OTS.tpa",  "OS1.fgm", "S1.pts", "S3.pts","S2.ast", "S4.pts", "OS3.dreb" )]
glm2_dat_scaled = as.data.frame(scale(glm2_dat))
glm2_dat_scaled['HTWins'] = train$HTwins_01
glm3 <- glm(HTWins ~ ., data = glm2_dat_scaled, family = 'binomial')

glm3_predict <- predict(glm3,type = "response")
glm3_01 <- ifelse(glm3_predict > 0.5, 1, 0)
confusionMatrix(as.factor(glm3_01), as.factor(train$HTwins_01))
summary(glm3)

diffData_test_double_norm <- as.data.frame(scale(diffData_test_double))
test3 <- predict(glm3, diffData_test_double_norm, type = "response")
test3_01 <- ifelse(test3 > 0.5, 1, 0)
test3_YN <- as.factor(ifelse(test3_01 == 1, 'Yes', 'No'))
test3_output <- data.frame('id' = test$id, 'HTWins' = test3_YN)
write.csv(test3_output, "test8.csv", row.names = FALSE)

# Accuracy Result: 65.898

###############12.2
nn0.00001epochs5000Minibatch16Result<- read.csv("nn0.00001epochs5000Minibatch16.csv")
testNN1_YN <- as.factor(ifelse(nn0.00001epochs5000Minibatch16Result == 1, 'Yes', 'No'))
test4_output <- data.frame('id' = test$id, 'HTWins' = testNN1_YN)
write.csv(test4_output, "test11.csv", row.names = FALSE)
# Accuracy Result: 64.077


