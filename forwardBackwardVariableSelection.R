train_rm <- read.csv("train_rm.csv")
test <- read.csv("test.csv")
table(train_rm$HTWins)
train_rm$HTwins_01 <- numeric(nrow(train_rm))
train_rm$HTwins_01[train_rm$HTWins == 'Yes'] <- 1
train_rm$HTwins_01[train_rm$HTWins == 'No'] <- 0

train_rm_num <- train_rm[, sapply(train_rm, class) == 'numeric']
glm.full <- glm(HTwins_01 ~., data = train_rm_num, family = 'binomial')


backBIC <- step(glm.full,direction="backward", data=train_rm_num, k = log(2500))

train_rm_num_scale <- data.frame(scale(train_rm_num))
train_rm_num_scale$HTwins_01 = train_rm_num$HTwins_01

glm.backBIC <- glm(HTwins_01 ~ 
                     VT.TS.fga + VT.TS.tpa + VT.TS.fta + VT.TS.pts + HT.OS5.fgm + 
                     VT.TA.fgm + VT.TA.tpa + VT.TA.fta + VT.OTS.fgm + VT.OTS.pts + 
                     VT.OTA.fgm + VT.OTA.fta + VT.OTA.ast + VT.OTA.blk + VT.S5.pts + 
                     VT.OS1.plmin + VT.OS3.plmin + VT.OS3.fgm + VT.OS4.dreb + 
                     HT.S1.pts + HT.S3.pts + 
                     HT.OS1.dreb + HT.OS3.dreb + HT.OS2.oreb + HT.OS4.oreb + HT.OS5.oreb, 
                   data = train_rm_num_scale, family = 'binomial')

glm.backBIC.pred <- predict(glm.backBIC, type = 'response')
glm.backBIC.pred.01 <- ifelse(glm.backBIC.pred > 0.5, 1, 0)
sum(glm.backBIC.pred.01 == train_rm_num_scale$HTwins_01)/ nrow(train_rm_num_scale)
summary(glm.backBIC)



test.double <- test[, sapply(test, class) == 'numeric']
test.scaled = data.frame(scale(test.double))

glm.backBIC.test.pred <- predict(glm.backBIC, newdata = test.scaled, type = 'response')
glm.backBIC.test.pred.01 <- ifelse(glm.backBIC.test.pred > 0.5, 1, 0)
glm.backBIC.test_YN <- as.factor(ifelse(glm.backBIC.test.pred.01 == 1, 'Yes', 'No'))
glm.backBIC.test_output <- data.frame('id' = test$id, 'HTWins' = glm.backBIC.test_YN)
write.csv(glm.backBIC.test_output, "test24.csv", row.names = FALSE)
#submission score = 0.6614, Dec 7th








## Forward Selection

null<-glm(HTwins_01~ 1, data=train_rm_num_scale, family = 'binomial') # 1 here means the intercept
full <- glm(HTwins_01 ~. , data = train_rm_num_scale, family = 'binomial')

forwardBIC <- step(null,
                   scope= list(lower=null,upper=full),
                   direction="forward", data=train_rm_num_scale,
                   k = log(2500)) # log 1500 gives the same output

glm.forwardBIC <- glm(HTwins_01 ~ VT.OS1.plmin + VT.S1.plmin + VT.OS3.plmin + VT.S3.plmin + 
                        VT.OTA.ast + VT.OTS.fgm + VT.OS4.dreb + VT.S5.stl + VT.OTA.blk + 
                        HT.S1.pts + VT.OS2.plmin + VT.TA.ast + VT.S2.pts + VT.OTA.dreb + 
                        VT.OTS.stl + HT.OS3.dreb + HT.OS3.oreb + HT.S3.pts + VT.OS3.fgm + 
                        VT.S1.pts,data = train_rm_num_scale, family = 'binomial')

glm.forwardBIC.pred <- predict(glm.forwardBIC, type = 'response')
glm.forwardBIC.pred.01 <- ifelse(glm.forwardBIC.pred > 0.5, 1, 0)
sum(glm.forwardBIC.pred.01 == train_rm_num_scale$HTwins_01)/ nrow(train_rm_num_scale)
summary(glm.forwardBIC)











