#Stock Analysis Project ISM 4930
library(quantmod)
library(xts)

#Import data and get various types of data formats
GSPC <- read.csv("C:/Users/brand/Downloads/INTC-5Y.csv")
GSPX <- xts(GSPC[,-1], order.by = as.Date(GSPC$Date))
GSPT <- ts(GSPC, frequency = 5)

#Initial EDA of INTC
#Graph of candles and volume
chartSeries(GSPX, theme="black")
zoomChart('2018::')
#Graph of just price and MACD and BBands 
chartSeries(GSPX[,5], theme = "black")
addMACD()
addBBands()
#daily return and graph
drGSPX <- dailyReturn(GSPX[,5])
plot(drGSPX)
#Distribution plot for daily returns
hist(drGSPX, breaks = 40, col = "Red", main = "Distribution of daily returns")
#########
#Metrics#
#########
HLC <- matrix(c(GSPC$High,GSPC$Low, GSPC$Close),nrow = length(GSPC$High))
GSPX.p <- GSPX[,5]
ret <- Delt(GSPX.p)
macd <- MACD(GSPC$Close, nFast=12, nSlow=26, nSig=9)  
sto <- stoch(HLC, nFastK = 14) *100
wpr <-WPR(HLC, n=14) * (-100)
roc <- ROC(GSPC$Close, n=14) *100
obv <- OBV(GSPC$Close, GSPC$Volume)

direction <- NULL
direction[GSPX.p > Lag(GSPX.p,20)] <- 1
direction[GSPX.p < Lag(GSPX.p,20)] <- 0
direction[is.na(direction)] <- 1

GSPX.p<-cbind(GSPX.p,ret,macd,sto,wpr,roc,obv,direction)
GSPX.p <- GSPX.p[-(1:33),]
dm <- dim(GSPX.p)
dm

summary(GSPX.p)
library(corrplot)
corrGSPX <- cor(GSPX.p[,1:11])
corrplot(corrGSPX,methods="circle")
corrGSPX[11,]

issd <- "2014-04-01"
ised <- "2018-02-20"
ossd <- "2018-02-21"
osed <- "2019-02-21"

isrow <- which(index(GSPX.p) >= issd& index(GSPX.p) <= ised)
osrow <- which(index(GSPX.p) >= ossd& index(GSPX.p) <= osed)
gTrain <- GSPX.p[isrow,]
gTest <- GSPX.p[osrow,]

library(rpart)
library(rpart.plot)
library(rpart.utils)
library(party)
library(tree)
library(randomForest)
library(e1071)
library(caret)
library(kernlab)
library(nnet)

fit.log <- glm(direction ~., data = gTrain, family = binomial)
fit.rpart <- rpart(direction ~ ., data = gTrain)
fit.Repart <- rpart(direction ~ .,method = "class", data = gTrain)
fit.ctree <- ctree(direction ~., data = gTrain)
fit.tree <- tree(direction ~., data = gTrain)
set.seed(150)
fit.rf <- randomForest(direction ~., data = gTrain)
dfGSPX <- as.data.frame(GSPX.p)
dfGSPX$direction <- as.factor(dfGSPX$direction)
class(dfGSPX$direction)
g2Train <- dfGSPX[isrow,]
g2Test <- dfGSPX[osrow,]
fit.rfc <- randomForest(direction ~., data = g2Train)

fit.svm <- svm(direction ~., data = gTrain, kernel = "radial", cost = 4, gamma = 0.5, epsilon = 0.1, type = "eps-regression")

rpart.plot(fit.rpart, cex = 0.6)
rpart.plot(fit.Repart, cex = 0.6)
plotcp(fit.Repart)
plotcp(fit.rpart)
plot(fit.ctree,cex = 0.6)
plot(fit.tree, cex=1)
text(fit.tree)
varImpPlot(fit.rf)
plot(fit.rf)
varImpPlot(fit.rfc)
plot(fit.rfc)

logPredict <- predict(fit.log, newdata = gTest, type = "response")
rpartPre <- rpart.predict(fit.rpart, newdata = gTest)
ctreePre <- predict(fit.ctree, newdata = gTest)
treePre <- predict(fit.tree, newdata = gTest)
rfPre <- predict(fit.rf, newdata = gTest)
svmPre <- predict(fit.svm, newdata = gTest)
ctPre <- predict(fit.Repart, newdata = gTest)
rfcPre <- predict(fit.rfc, newdata = g2Test, type = "prob")
library(pROC)
auc(response = gTest[,11], predictor = rpartPre)    #0.8842
auc(response = gTest[,11], predictor = ctreePre)    #0.8903
auc(response = gTest[,11], predictor = treePre)     #0.919
auc(response = gTest[,11], predictor = rfPre)       #0.9367
auc(response = gTest[,11], predictor = svmPre)      #0.7355
auc(response = gTest[,11],predictor = logPredict)   #0.9633
auc(response = gTest[,11], predictor = ctPre[,1])   #0.9388   
auc(response = as.numeric(g2Test[,11]), predictor = as.numeric(rfcPre[,1])) #0.93
library(ROCR)
pred_rp <- prediction(labels = gTest[,11], predictions = rpartPre)
perf_rp <- performance(pred_rp, "tpr", "fpr")
pred_cf <- prediction(labels = gTest[,11], predictions = ctreePre)
perf_cf <- performance(pred_cf, "tpr", "fpr")
pred_log <- prediction(labels = gTest[,11], predictions = logPredict)
perf_log <- performance(pred_log, "tpr", "fpr")
pred_t <- prediction(labels = gTest[,11], predictions = treePre)
perf_tree <- performance(pred_t, "tpr", "fpr")
pred_rf <- prediction(labels = gTest[,11], predictions = rfPre)
perf_rf <- performance(pred_rf, "tpr", "fpr")
pred_svm <- prediction(labels = gTest[,11], predictions = svmPre)
perf_svm <- performance(pred_svm, "tpr", "fpr")
pred_crt <- prediction(labels = gTest[,11], predictions = ctPre[,2])
perf_crt <- performance(pred_crt, "tpr", "fpr")
pred_rfc <- prediction(labels = as.numeric(g2Test[,11]), predictions = as.numeric(rfcPre[,2]))
perf_rfc <- performance(pred_rfc, "tpr", "fpr")


plot(perf_cf, col =1)
plot(perf_log,add = TRUE, col = 2)
plot(perf_tree, add = TRUE, col = 3)
plot(perf_rf, add = TRUE, col = 4)
plot(perf_svm, add = TRUE, col = 5)
plot(perf_rp, add = TRUE, col = 6)
plot(perf_crt, add = TRUE, col = 7)
plot(perf_rfc, add = TRUE, col = 8)
legend("bottomright", legend=c("ctree", "log", "tree","random forest","svm","rpart", "class tree","class rf"),
       col=c(1,2,3,4,5,6,7,8), lty=1, cex=0.6)
#Confusion Matrix
#We are more concerned about Specifictiy

logP <- ifelse(logPredict > 0.5,1,0)
rpP <- ifelse(rpartPre > 0.5,1,0)
ctPre <- ifelse(ctreePre > 0.5,1,0)
tPre <- ifelse(logPredict > 0.5,1,0)
rPre <- ifelse(rfPre > 0.5,1,0)
sPre <- ifelse(svmPre > 0.5, 1, 0)
cPre <- ifelse(ctPre > 0.5, 1, 0)
rcPre <- ifelse(rfcPre > 0.5, 1,0)
confusionMatrix(factor(logP),factor(gTest[,11]))     #0.7658/0.9574     A:0.873
confusionMatrix(factor(rpP),factor(gTest[,11]))      #0.08108/0.93617   A:0.5596
confusionMatrix(factor(ctPre),factor(gTest[,11]))    #0.11712/0.92908   A:0.5714
confusionMatrix(factor(tPre), factor(gTest[,11]))    #0.7658/0.9574     A:0.873
confusionMatrix(factor(rPre), factor(gTest[,11]))    #0.8559/0.9078     A:0.8849
confusionMatrix(factor(sPre),factor(gTest[,11]))     #0.7387/0.9291     A.0.8452
confusionMatrix(factor(cPre[,2]),factor(gTest[,11])) #0.7387/0.9362     A:0.8492
confusionMatrix(factor(rcPre[,2]), factor(g2Test[,11])) #0.8559/0.9149  A:0.8889

##SVM not predicting as it used to. 
############
#NOT DONE!!#
############
#Finding best models for some models

svmH <- tune.svm(direction~., data = gTrain, gamma = 2^(-1:1), cost = 2^(2:4))
summary(svmH)
plot(svmH)
#I couldn't recreate the random forest tune but I know it was 500 :)
rpartTunemd <- tune.rpart(direction~., data = gTrain,maxdepth = 1:5,cp = c(0.002,0.005,0.01,0.015,0.02,0.03))
summary(rpartTunemd)
plot(rpartTunemd)

###########
#SVM Plots#
###########
GSPdf <- as.data.frame(GSPX.p)
plot(GSPdf, col = GSPdf$direction)
GSPdf$direction <- as.factor(GSPdf$direction)
plot(GSPdf$roc,GSPdf$macd, col = GSPdf$direction)
svmTrain <- GSPdf[isrow,c(3,10,11)]
svmTest <- GSPdf[osrow,]
svmplotFit <- svm(direction~., data = svmTrain,  kernel = "radial", cost = 100, gamma=4)
print(svmplotFit)
plot(svmplotFit, svmTrain)
svmplottune <- tune.svm(direction~., data = svmTrain, kernel = "radial", cost=c(5,10,100,500,1000), gamma = c(0.1,1,2,3,4,5,10))
summary(svmplottune)
plot(svmplottune)

svmPlotPre <- predict(svmplotFit, newdata = svmTest, type="response")
auc(response = svmTest[,3], predictor = as.numeric(svmPlotPre))      #0.8953

predsvm2 <- prediction(labels = as.numeric(svmTest[,3]), predictions = as.numeric(svmPlotPre))
perfsvm2 <- performance(predsvm2, "tpr", "fpr")
plot(perfsvm2)

confusionMatrix(svmPlotPre,svmTest[,3])

##################
#END OF SVM PLOTS#
##################
#########
#Returns#
#########
library(PerformanceAnalytics)
signal <- ifelse(logP == 1,1, ifelse(logP == 0, -1,0))
ret <- ret[osrow]
cost <-  0
cumm_ret <- Return.cumulative(ret)
annual_ret <- Return.annualized(ret)
charts.PerformanceSummary(ret) #Passive return is 0.027 but our method is -0.092/ full 5 years: passive: 0.4823/Active: 0.439

#####
#SMA#
#####

ret.INTX <- Delt(GSPX[,4])

sma20 <- SMA(GSPX[,4], n=20)
sma120 <- SMA(GSPX[,4],n = 120)
plot(GSPX[,4], col = "Blue")
lines(sma20, col = "Red")
lines(sma120, col = "Black")

issd2 <- "2014-04-01"
ised2 <- "2018-02-20"
ossd2 <- "2018-02-21"
osed2 <- "2019-02-21"

pssd <- "2019-01-21"
psed <- "2019-02-21"

pssd2 <- "2018-12-21"
psed2 <- "2019-02-21"

isrow2 <- GSPX[(index(GSPX) >= issd2& index(GSPX) <= ised2),]
inret2 <- ret.INTX[(index(ret.INTX) >= issd2 & index(ret.INTX) <= ised2),]
osrow2 <- GSPX[(index(GSPX) >= ossd2& index(GSPX) <= osed2),]
osret2 <- ret.INTX[(index(ret.INTX) >= ossd2 & index(ret.INTX) <= osed2),]

psrow <- GSPX[(index(GSPX) >= pssd& index(GSPX) <= psed),]
psret <- ret.INTX[(index(ret.INTX) >= pssd & index(ret.INTX) <= psed),]

psrow2 <- GSPX[(index(GSPX) >= pssd2& index(GSPX) <= psed2),]
psret2 <- ret.INTX[(index(ret.INTX) >= pssd2 & index(ret.INTX) <= psed2),]

bss <- NULL
bss[sma20 > sma120] <- 1
bss[sma20 < sma120] <- 0

library(PerformanceAnalytics)
#Performance metrics using Direction - Full year
signal <- ifelse(logP == 1,1, ifelse(logP == 0, -1,0))
ret <- ret[osrow]
cost <-  0
cumm_ret <- Return.cumulative(psret2)
annual_ret <- Return.annualized(psret2)
charts.PerformanceSummary(psret2)
#2 months 
#Performance metrics using Momentium trading. 
signal <- NULL
signal <- ifelse(bss==1,1,ifelse(bss==0,-1,0))
cost <-  0
cumm_ret <- Return.cumulative(psret)
annual_ret <- Return.annualized(psret)
charts.PerformanceSummary(psret) #Passive Return: 0.0650, Active Return: 0.0451

cumm_ret2 <- Return.cumulative(psret2)
annual_ret2 <- Return.annualized(psret2)
charts.PerformanceSummary(psret2) #Passive Return: 0.1465, Active: 0.1289

cumm_ret2 <- Return.cumulative(osret2)
annual_ret2 <- Return.annualized(osret2)
charts.PerformanceSummary(osret2)