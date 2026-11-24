################################################################################
################################################################################
## Taller: Explicando Predicciones y Modelos de Machine Learning
##
## Script: Construcción de los Modelos Predictivos. Statlog Heart
##
## AUTOR: Jesús Herranz
##
## V Congreso y XVI Jornadas de Usuarios de R 2026
## Noviembre 2026 Universidad de La Laguna
##
################################################################################
################################################################################
 
rm(list=ls(all=TRUE))  
 
################################################################################
## Librerías
library(tidymodels)
library(tidyverse)

library(corrplot)
library(GGally)
library(infotheo)   ## NMI

library(rpart)
library(rpart.plot)
library(partykit)
library(randomForest)
library(nnet)
library(e1071)
library(caret)
library(pROC)

################################################################################
## Lectura de Datos. Heart
################################################################################

## Lectura del fichero
path = "C://Taller Explicando Predicciones y Modelos de Machine Learning/Datos/"
xx_hd_all <- read.delim( paste0(path, "heart.dat"), header = FALSE, sep= " " )
dim(xx_hd_all)

## Nombres
names(xx_hd_all) <- c( "age", "sex", "chest_pain", "blood_press", "chol",
                       "sugar", "ecg", "max_heart_rate", "angina", "oldpeak",
                       "slope", "major_vessels", "thal", "diagnosis" )
head(xx_hd_all)

## Factores
xx_hd_all <- xx_hd_all %>%
  mutate(
    sex = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    chest_pain = factor(chest_pain, levels = c(1, 2, 3, 4),
                                    labels = c("Typical", "Atypical", 
                                               "NonAnginal", "Asymptomatic")),
    sugar = factor(sugar, levels = c(0, 1), labels = c("No", "Yes")),
    ecg = factor(ecg, levels = c(0, 1, 2), 
                      labels = c("Normal", "ST-T", "LVH")),
    angina = factor(angina, levels = c(0, 1), labels = c("No", "Yes")),
    slope = factor(slope, levels = c(1, 2, 3), labels = c("Up", "Flat", "Down")),
    major_vessels = factor(major_vessels, levels = c(0, 1, 2, 3),
                                labels = c("None", "One", "Two", "Three")),
    thal = factor(thal, levels = c(3, 6, 7),
                        labels = c("Normal", "Fixed", "Reversible")),
    diagnosis = factor(diagnosis,  levels = c(1, 2), labels = c("No", "Yes"))
  )

# Columnas y nombres de las variables
col_resp <- which(names(xx_hd_all) == "diagnosis")  ## Columna diagnosis
col_pred <- names(xx_hd_all)[ - col_resp ]          ## Columnas Predictores
name_pred_cont  <- c( "age", "blood_press", "chol","max_heart_rate", "oldpeak" )
name_pred_categ <- c( "sex", "chest_pain", "sugar", "ecg", "angina",
                       "slope", "major_vessels", "thal" )

## Explorar
table(xx_hd_all$diagnosis)
glimpse(xx_hd_all)
str(xx_hd_all)


################################################################################
## Definición de la Muestra de Training y Muestra de Testing
################################################################################

set.seed(101)   ## Se fija una semilla, para reproducir los datos
cred_split <- initial_split(xx_hd_all, prop = 0.70, strata = diagnosis)
cred_split
xx_hd_train <- training(cred_split)
xx_hd_test <- testing(cred_split)
dim(xx_hd_train)
dim(xx_hd_test)


################################################################################
################################################################################
## Análisis Exploratorio
################################################################################
################################################################################

################################################################################
## Variables continuas

# Pasamos a formato largo para usar facet_wrap
df_long <- xx_hd_train %>%
  pivot_longer(cols = all_of(name_pred_cont),
               names_to = "Variable",
               values_to = "Valor")

## Histograms 
dev.new()
ggplot(df_long, aes(x = Valor)) +
  geom_histogram(bins = 20, color = "black", fill = "steelblue", alpha = 0.7) +
  facet_wrap(~ Variable, scales = "free") +
  labs(title = "Distribución de las Variables Continuas",
       x = "",
       y = "Frecuencia") +
  theme_bw()


## Boxplots
dev.new()
ggplot(df_long, aes(x = diagnosis, y = Valor, fill = diagnosis)) +
  geom_boxplot() +
  facet_wrap(~ Variable, scales = "free_y") +
  labs(title = "Boxplots por Estado de Enfermedad Cardiaca",
       x = "Heart Disease",
       y = "") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = rev(scales::hue_pal()(2)))



## Medias y Test de Mann-Whitney
wilcox.test(age ~ diagnosis, data=xx_hd_train)$p.value
tapply(xx_hd_train$age, xx_hd_train$diagnosis, mean)
wilcox.test(blood_press ~ diagnosis, data=xx_hd_train)$p.value
tapply(xx_hd_train$blood_press, xx_hd_train$diagnosis, mean)
wilcox.test(chol ~ diagnosis, data=xx_hd_train)$p.value
tapply(xx_hd_train$chol, xx_hd_train$diagnosis, mean)
wilcox.test(max_heart_rate ~ diagnosis, data=xx_hd_train)$p.value
tapply(xx_hd_train$max_heart_rate, xx_hd_train$diagnosis, mean)
wilcox.test(oldpeak ~ diagnosis, data=xx_hd_train)$p.value
tapply(xx_hd_train$oldpeak, xx_hd_train$diagnosis, mean)
    

################################################################################
## Variables Categóricas

name_resp <- "diagnosis"
round(100*prop.table(table(xx_hd_train$diagnosis)),1)

# Pasar a formato largo
df_long_cat <- xx_hd_train %>%
  pivot_longer(cols = all_of(c(name_pred_categ, name_resp)),
               names_to = "Variable",
               values_to = "Categoria")

# Gráfico de barras de conteo
dev.new()
ggplot(df_long_cat, aes(x = Categoria, fill = Categoria)) +
  geom_bar() +
  facet_wrap(~ Variable, scales = "free_x") +
  labs(title = "Conteo de categorías de variables predictoras",
       x = "Categoría",
       y = "Frecuencia") +
  theme_bw() +
  theme(legend.position = "none")

# Gráfico de barras agrupado por HeartDisease
dev.new()
ggplot(df_long_cat, aes(x = Categoria, fill = diagnosis)) +
  geom_bar(position = "dodge") +
  facet_wrap(~ Variable, scales = "free_x") +
  labs(title = "Distribución de categorías según Diagnóstico",
       x = "Categoría",
       y = "Cantidad") +
  theme_bw() +
  scale_fill_manual(values = rev(scales::hue_pal()(2)))



################################################################################
## Correlaciones
################################################################################

################################################################################
## Explorando las asociaciones entre las variables
#dev.new()
#ggpairs( xx_hd_train [ , col_pred] )
         

################################################################################
## Gráfico de la matriz de correlaciones

w_cor <- cor( xx_hd_train[ , name_pred_cont ])
dev.new()
corrplot( w_cor, order = "hclust", diag = FALSE, addCoef.col = "black")


################################################################################
## NMI. Normalized Mutual Information
################################################################################

## se copia el dataset, se separan las numéricas y las categórica
xx_nmi <- xx_hd_train
xx_nmi_num <- xx_nmi[ , name_pred_cont ]
xx_nmi_cat <- xx_nmi[ , name_pred_categ ]

## Se discretizan las numéricas y se vuelven a unir todas
xx_nmi_num_disc <- infotheo::discretize(xx_nmi_num, disc = "equalfreq")
xx_nmi <- bind_cols( xx_nmi_num_disc, xx_nmi_cat)

## Cálculo del NMI
n <- ncol(xx_nmi)

w_nmi <- matrix(NA_real_, n, n,
                dimnames = list(names(xx_nmi), names(xx_nmi)))

for (i in seq_len(n)) {
  for (j in seq_len(n)) {
    
    mi <- infotheo::mutinformation(xx_nmi[[i]], xx_nmi[[j]])
    hx <- infotheo::entropy(xx_nmi[[i]])
    hy <- infotheo::entropy(xx_nmi[[j]])
    
    w_nmi[i, j] <- 2 * mi / (hx + hy)
  }
}

w_nmi[ w_nmi > 0.15 & w_nmi < 0.9 ]


diag(w_nmi) <- NA  ## Cambia 1 por NA en la diagonal

dev.new()
corrplot( w_nmi, order = "hclust", diag = FALSE, addCoef.col = "black",
          tl.cex = 0.8, number.cex = 0.8,  
          col.lim = c(0, 0.5), col = colorRampPalette(c("white", "lightblue", "blue"))(200))



################################################################################
################################################################################
## Modelo. Regresión Logística
################################################################################
################################################################################

mod_rlog <- glm( diagnosis ~ ., data = xx_hd_train, family = binomial )
summary(mod_rlog)


## Predicción de probabilidades
pred_rlog_prob = predict( mod_rlog, newdata = xx_hd_test, type = "response" )

## Predicción de clase
pred_rlog_class <- ifelse(pred_rlog_prob >= 0.5, "Yes", "No")
pred_rlog_class <- factor(pred_rlog_class, levels = levels(xx_hd_test$diagnosis))


## Medidas de capacidad predictiva
cm_rlog = confusionMatrix( pred_rlog_class, xx_hd_test$diagnosis )
cm_rlog$overall["Accuracy"]
cm_rlog$overall["Kappa"]

roc_rlog = roc( xx_hd_test$diagnosis, pred_rlog_prob )
auc(roc_rlog) 


################################################################################
################################################################################
## Modelo: Árbol de Clasificación
################################################################################
################################################################################

mod_cart <- rpart( diagnosis ~ . , method="class", data=xx_hd_train )
dev.new()
prp(mod_cart ,extra=1 )
#dev.new()
#plot(as.party(mod_cart))

## Predicción de clase
pred_cart_class = predict( mod_cart, newdata = xx_hd_test, type = "class" )

## Predicción de probabilidades
pred_cart_prob = predict( mod_cart, newdata = xx_hd_test, type = "prob" )

## Medidas de capacidad predictiva
cm_cart = confusionMatrix( pred_cart_class, xx_hd_test$diagnosis )
cm_cart$overall["Accuracy"]
cm_cart$overall["Kappa"]

roc_cart = roc( xx_hd_test$diagnosis, pred_cart_prob[ ,"Yes"] )
auc(roc_cart) 


  
################################################################################
################################################################################
## Modelo: Random Forest
################################################################################
################################################################################

mod_rf <- randomForest( diagnosis ~ . , data=xx_hd_train, 
                        importance = F, , ntree = 10000 )
mod_rf
dev.new(); varImpPlot( mod_rf, main="")

## Predicción de clase
pred_rf_class = predict( mod_rf, newdata = xx_hd_test, type = "class" )

## Predicción de probabilidades
pred_rf_prob = predict( mod_rf, newdata = xx_hd_test, type = "prob" )

## Medidas de capacidad predictiva
cm_rf = confusionMatrix( pred_rf_class, xx_hd_test$diagnosis )
cm_rf$overall["Accuracy"]
cm_rf$overall["Kappa"]

roc_rf = roc( xx_hd_test$diagnosis, pred_rf_prob[ ,"Yes"] )
auc(roc_rf) 



################################################################################
################################################################################
## Modelo: Red Neuronal
################################################################################
################################################################################

## Control de la Técnica de Remuestreo. 3 times 10-fold CV
cv_ctrl = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
                                                
## Parámetros a explorar en NN                             
nnGrid = expand.grid( .decay = c( 0.001, 0.01, 0.1, 0.2, 0.3, 0.4 ), 
                      .size  = c( 2, 4, 6, 8 ) )     


## Construcción del Modelo Predictivo
mod_nn = train(
  xx_hd_train[, - col_resp],
  xx_hd_train$diagnosis,
  method = "nnet",
  tuneGrid = nnGrid,
  trControl = cv_ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  trace = F
)

mod_nn

## Gráfico
dev.new()
plot(mod_nn)

## Predicción de clase
pred_nn_class = predict( mod_nn , xx_hd_test)
head(pred_nn_class)                
## Predicción de probabilidades
pred_nn_prob  = predict( mod_nn , xx_hd_test, type = "prob" )[ , "Yes"]
head(pred_nn_prob)


## Medidas de capacidad predictiva
cm_nn = confusionMatrix( pred_nn_class, xx_hd_test$diagnosis )
cm_nn$overall["Accuracy"]
cm_nn$overall["Kappa"]

roc_nn = roc( xx_hd_test$diagnosis, pred_nn_prob )
auc(roc_nn) 



################################################################################
################################################################################
## Modelo: Support Vector Machine
################################################################################
################################################################################

## Optimización de parámetros, con kernel "radial"
tune_rad <- tune.svm( diagnosis ~ . , data = xx_hd_train, kernel="radial",
                      scale = TRUE,   
                      gamma = c( 0.1, 0.01, 0.001, 0.0001 ), 
                      cost = c( 10, 50, 100, 500, 1000) )
tune_rad
summary(tune_rad)

## Modelo final con los parámetros óptimos, con "probability"
tune_rad$best.parameters
best_rad_G <- tune_rad$best.parameters$gamma 
best_rad_C <- tune_rad$best.parameters$cost

mod_svm <- svm( diagnosis ~ . , data = xx_hd_train, kernel="radial", scale = TRUE,
                gamma = best_rad_G, cost = best_rad_C, probability=TRUE )

## Predicciones en la muestra de testing
pred_svm <- predict(mod_svm, xx_hd_test, decision.values=TRUE, probability=TRUE)

## Predicción de clase
pred_svm_class <- as.vector(pred_svm)
pred_svm_class <- as.factor(pred_svm_class)
head(pred_svm_class)  

## Predicción de probabilidades
head( attr(pred_svm, "probabilities" ) )
pred_svm_prob <- attr(pred_svm, "probabilities" )[ , "Yes" ] 
head(pred_svm_prob)


## Medivas de capacidad predictiva
cm_svm = confusionMatrix( pred_svm_class, xx_hd_test$diagnosis )
cm_svm$overall["Accuracy"]
cm_svm$overall["Kappa"]

roc_svm = roc( xx_hd_test$diagnosis, pred_svm_prob )
auc(roc_svm) 


################################################################################
################################################################################
## Salva los data frame y los modelos en un RData
################################################################################
################################################################################

data.frame( method = c("RLog","CART","RF","NN","SVM"),
               accuracy = c(cm_rlog$overall["Accuracy"], 
                         cm_cart$overall["Accuracy"], cm_rf$overall["Accuracy"], 
                         cm_nn$overall["Accuracy"], cm_svm$overall["Accuracy"]),
               auc = c(auc(roc_rlog), auc(roc_cart), auc(roc_rf), auc(roc_nn),
                       auc(roc_svm)) )             
                          

#save( xx_hd_all, xx_hd_train, xx_hd_test, 
#      mod_rlog, mod_cart, mod_rf, mod_nn, mod_svm, 
#      file = paste0( path, "Statlog Heart Data and Models.RData") )


                 