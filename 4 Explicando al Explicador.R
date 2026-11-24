################################################################################
################################################################################
## Taller: Explicando Predicciones y Modelos de Machine Learning
##                                                               
## Script: Explicando las Técnicas de Explicación. Statlog Heart
##
## OBSERVACIÓN: Este script está preparado para explicar un modelo de random
##              forest. Si se desea otra técnica, hay que revisar todas las
##              funciones predict() que aparecen en el script
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
## Libraries

library(tidyverse)
library(randomForest)
library(pROC)
library(DALEX)


################################################################################
################################################################################
## Functions
################################################################################
################################################################################

################################################################################
## Calcula las contribuciones individuales de varias variables, y sus valores 
## - "set_var" contiene los nombres de las variables ordenadas
## - "obs_explain" observación a explicar, contiene los valores de las variables  
## - "model_explore" modelo construido 
## - "dataset_exp" el dataset en el que se basa la explicación

calc_contrib_indiv <- function(model_explore, obs_explain, dataset_exp, set_var)
{ 
  ## Prediction of the observation
  pred_obs <- predict(model_explore, obs_explain, type="prob")[,2]
  pred_obs 
  
  ## Baseline prediction, called intercept. Mean of the prediction in the dataset 
  pred_data <- predict(model_explore, dataset_exp, type = "prob")[,2]
  intercept <- mean(pred_data)
  intercept 

  ## Contribution
  contrib <- rep(NA, length(set_var))   ## To save the contributions
  names(contrib) <- set_var
  pred_temp <- rep(NA, length(set_var)) ## Predicion of the temporal data frame
  
  ## Loop to variables
  for( i in 1:length(set_var))
  { ## Data frame temporal, changing all the variable to explain for the value of
    ## the observatio in this variable until the index. Controlling the Sequence 
    data_temp <- dataset_exp
    ## 1:i is the sequence, from 1 until i
    data_temp[ , set_var[1:i]] <- obs_explain[ , set_var[1:i] ] 
    
    ## Mean of the predictions in the temporal dataframe
    pred_temp[i] <- mean(predict(model_explore, data_temp, type = "prob" )[,2])   
    
    ## The Contribution is the difference between this prediction and the previous
    if(i==1) contrib[i] <- pred_temp[i] - intercept
    if(i>1)  contrib[i] <- pred_temp[i] - pred_temp[i-1]  
    
  }

  ## Ouput
  list( contrib_ind = contrib, pred_dataset_temp = c(intercept,pred_temp) )
}


################################################################################
## Calcula las contribuciones por parejas de varias variables, y sus valores
## Solo cálcula las contribuciones como primeras variables de todas las parejas
## NO analiza una secuencia 

calc_contrib_pair_first <- function(model_explore, obs_explain, dataset_exp, set_var)
{ 
  ## Prediction of the observation
  pred_obs <- predict(model_explore, obs_explain, type="prob")[,2]
  pred_obs 
  
  ## Baseline prediction, called intercept. Mean of the prediction in the dataset 
  pred_data <- predict(model_explore, dataset_exp, type = "prob")[,2]
  intercept <- mean(pred_data)
  intercept 

  ## Contribution
  w_long = ( length(set_var) * (length(set_var) - 1)) / 2
  contrib <- rep(NA, w_long)   ## To save the contributions
  ## Nombres de las parejas
  names(contrib) <- combn(names(obs_explain), 2, 
                          FUN = \(x) paste(x, collapse = " : "))
  pred_temp <- rep(NA, w_long) ## Predicion of the temporal data frame
  
  ind_pred <- 0
  
  ## Loop to variables
  for( i in 1:(length(set_var)-1) )
  for( j in (i+1):length(set_var) )
  { ## Solo interesa la contribución de la pareja como primera variable
    data_temp <- dataset_exp
    data_temp[ , set_var[i]] <- obs_explain[ , set_var[i] ] 
    data_temp[ , set_var[j]] <- obs_explain[ , set_var[j] ] 
  
    ## indice 
    ind_pred <- ind_pred + 1   
    ## Mean of the predictions in the temporal dataframe
    pred_temp[ind_pred] <- mean(predict(model_explore, data_temp, type = "prob" )[,2])   

    ## The Contribution is the difference between this prediction and the previous
    contrib[ind_pred] <- pred_temp[ind_pred] - intercept    
  }

  ## Ouput
  list( contrib_pair = contrib )
}


################################################################################
## Calcula las contribuciones de varias variables, que pueden ser variable 
## individuales o pares de variables, es decir, interacciones
## - "set_var" contiene los nombres de las variables ordenadas, que pueden ser
##   indoviduales o interacciones ( " : " ) 
## - "obs_explain" observación a explicar, contiene los valores de las variables  
## - "model_explore" modelo construido 
## - "dataset_exp" el dataset en el que se basa la explicación

calc_contrib_indiv_or_pair <- function(model_explore, obs_explain, dataset_exp, set_var)
{ 
  ## Preparación de las variables según sea efecto principal o interacción
  ## - En las interacciones se separan los nombres y se pone un mismo valor en 
  ##   "ind_step" para saber que se deben tratar juntas
  ind_step <- NULL
  var_step <- NULL
  for( j in 1:length(set_var))
  {
    w_pos <- regexpr(" : ", set_var[j] )[1]
    if ( w_pos > 1 )
    { ## Interacción
      name_var1 <- substr( set_var[j] , 1 , w_pos - 1 )
      name_var2 <- substr( set_var[j] , w_pos + 3 , nchar(set_var[j]))
      ind_step = c( ind_step, j, j ) ## se asigna el mismo step a las 2 variables
      var_step = c( var_step, name_var1, name_var2 ) ## nombres de las 2 variables
    }
    else 
    { ## Individual
      ind_step = c( ind_step, j ) 
      var_step = c( var_step, set_var[j] ) }
  }

  ## Prediction of the observation
  pred_obs <- predict(model_explore, obs_explain, type="prob")[,2]
  pred_obs 
  
  ## Baseline prediction, called intercept. Mean of the prediction in the dataset 
  pred_data <- predict(model_explore, dataset_exp, type = "prob")[,2]
  intercept <- mean(pred_data)
  intercept 

  ## Contribution
  contrib <- rep(NA, length(set_var))   ## To save the contributions
  names(contrib) <- set_var
  pred_temp <- rep(NA, length(set_var)) ## Predicion of the temporal data frame
  
  
  ## Loop por valores distintos de "ind_step" que puede tener valores iguales,
  ## cuando hay interacciones

  for( i in unique(ind_step))
  { ## Data frame temporal, changing all the variable to explain for the value of
    ## the observatio in this variable until the index. Controlling the Sequence 
    data_temp <- dataset_exp
    
    ## Las variables de las interacciones se tratan juntas, ya que están consecutivas
    ## con el mismo valor en "ind_step"
    data_temp[ ,var_step[ind_step <= i]] <- obs_explain[ ,var_step[ind_step <= i]]
    
    ## Mean of the predictions in the temporal dataframe
    pred_temp[i] <- mean(predict(model_explore, data_temp, type = "prob" )[,2])   
    
    ## The Contribution is the difference between this prediction and the previous
    if(i==1) contrib[i] <- pred_temp[i] - intercept
    if(i>1)  contrib[i] <- pred_temp[i] - pred_temp[i-1]  
    
  }

  ## Ouput
  list( contrib = contrib )
}



################################################################################
################################################################################
################################################################################
## Lectura de Datos. Heart Disease
################################################################################
################################################################################
################################################################################

path = "C://Taller Explicando Predicciones y Modelos de Machine Learning/Datos/"
load(paste0(path, "Statlog Heart Data and Models.RData"))
ls()
dim(xx_hd_all)
dim(xx_hd_train)
dim(xx_hd_test)

head(xx_hd_all)
summary(xx_hd_all)


# Columnas y nombres de las variables
col_resp <- which(names(xx_hd_all) == "diagnosis")  ## Columna diagnosis
name_pred <- names(xx_hd_all)[ - col_resp ]         ## Columnas Predictores
name_pred_cont  <- c( "age", "blood_press", "chol","max_heart_rate", "oldpeak" )
name_pred_categ <- c( "sex", "chest_pain", "sugar", "ecg", "angina",
                       "slope", "major_vessels", "thal" )

################################################################################
## Explainer sobre la muestra completa
################################################################################
 
exp_rf   <- DALEX::explain( mod_rf, data = xx_hd_all[ , - col_resp], 
                            y = xx_hd_all$diagnosis == "Yes", label = "RF")

################################################################################
## Observaciones a explicar
################################################################################

## Observación 1
(pepito <- as.data.frame(xx_hd_all[36, - col_resp] ))
predict(mod_rf, pepito, type="prob")[ , "Yes"]

## Observación 2
(juanita <- as.data.frame(xx_hd_all[15, - col_resp] ))
predict(mod_rf, juanita, type="prob")[ , "Yes"]    

## Probabilidad media
mean(predict(mod_rf, xx_hd_all, type="prob")[ , "Yes"])


################################################################################
################################################################################
################################################################################
## 1.- Técnicas de Explicación Local
################################################################################
################################################################################
################################################################################


################################################################################
################################################################################
## 1.1.- Break-down Plots for Additive Attributions
################################################################################
################################################################################

## Break-down plot con el orden por defecto
bd_rf_pep <- predict_parts(explainer = exp_rf, new_observation = pepito,
                           type = "break_down" )
bd_rf_pep


################################################################################
## Explaining Break-down Plot

###########################
## Step 1: Contributions of all variables and values of the observation, trying 
## as first variable to explain 

contrib_first <- NULL
## Loop for all the variables
for ( name_var in name_pred)
{ ## Contribution of the each variable
 contrib_first <- c(contrib_first,
           calc_contrib_indiv(mod_rf, pepito, xx_hd_all, name_var)$contrib_ind ) 
}  
contrib_first   


## Las variables se ordenan por el valor absoluto, en orden decreciente, de su 
## contribución como primeras variables a explorar
ordered_contrib <- contrib_first[order(abs(contrib_first), decreasing = TRUE)]
ordered_contrib
ordered_var <- names(ordered_contrib)
ordered_var


###########################
## Step 2: Contributions of all variables and values of the observation 

#contrib <- rep(NA, length(ordered_var))   ## To save the contributions
#names(contrib) <- ordered_var
#pred_temp <- rep(NA, length(ordered_var)) ## Predicion of the temporal data frame

calc_contrib_indiv(mod_rf, pepito, xx_hd_all, ordered_var)
bd_rf_pep


################################################################################
## Break-down results. Contributions of all variables and values of the observation 
## Alternative Explications of the 3 first contributions
################################################################################

## Initial Situation
head(predict(mod_rf, xx_hd_all, type="prob")[,2])
tail(predict(mod_rf, xx_hd_all, type="prob")[,2])

intercept <- mean(predict(mod_rf, xx_hd_all, type = "prob" )[,2]) 
intercept
obs_explain <- pepito

## Start with the data set
data_temp <- xx_hd_all

## Contribution 1 
ordered_var[1]
data_temp[ , ordered_var[1]] <- obs_explain[ , ordered_var[1]]
head(predict(mod_rf, data_temp, type="prob")[,2])
tail(predict(mod_rf, data_temp, type="prob")[,2])
      
pred_temp_1 <- mean(predict(mod_rf, data_temp, type = "prob" )[,2])   
pred_temp_1
contrib_1 <- pred_temp_1 - intercept
contrib_1
    

## Contribution 2 
ordered_var[2]
data_temp[ , ordered_var[2]] <- obs_explain[ , ordered_var[2]]
head(predict(mod_rf, data_temp, type="prob")[,2])
tail(predict(mod_rf, data_temp, type="prob")[,2])
    
pred_temp_2 <- mean(predict(mod_rf, data_temp, type = "prob" )[,2])       
pred_temp_2
contrib_2 <- pred_temp_2 - pred_temp_1
contrib_2
    
    
## Contribution 3
ordered_var[3]
data_temp[ , ordered_var[3]] <- obs_explain[ , ordered_var[3]]
   
pred_temp_3 <- mean(predict(mod_rf, data_temp, type = "prob" )[,2])   
pred_temp_3
contrib_3 <- pred_temp_3 - pred_temp_2
contrib_3



################################################################################
################################################################################
## 1.2.- Break-down Plot for Interactions
################################################################################
################################################################################

################################################################################
## Break-down Plot for Interactions - Random Forest

bd_rf_int <- predict_parts(explainer = exp_rf, new_observation = pepito,
                        type = "break_down_interactions" )
bd_rf_int
dev.new()
plot(bd_rf_int)

################################################################################
## Explaining Break-down Plot  for Interactions
################################################################################

###########################
## Step 1: Contributions of all variables and values of the observation, trying as first
## variable to explain 

contrib_first <- NULL
## Loop for all the variables
for ( name_var in names(pepito))
{ ## Contribution of the each variable
 contrib_first <- c(contrib_first,
             calc_contrib_indiv(mod_rf, pepito, 
                                xx_hd_all, name_var)$contrib_ind ) 
}  
contrib_first   
round(contrib_first[ order(abs(contrib_first), decreasing=T) ] ,3)


###########################
## Step 2.1: Contributions of all the pair variables and values of the observation, 
## trying as first pair of variables to explain 

contrib_first_inter <- calc_contrib_pair_first(mod_rf, pepito, 
                                               xx_hd_all, names(pepito))
contrib_first_inter 


###########################
## Step 2.2: calculate the Net Effect of the Interaction

net_effect_inter <- rep(NA, length(contrib_first_inter$contrib_pair))
names(net_effect_inter) <- names(contrib_first_inter$contrib_pair)

for( ind_pair in 1:length(net_effect_inter))
{ ## Nombre de las 2 variables por separado, localizando " : "
  w_pos <- regexpr(" : ", names(net_effect_inter)[ind_pair] )[1]
  name_var1 <- substr( names(net_effect_inter)[ind_pair] , 1 , w_pos - 1 )
  name_var2 <- substr( names(net_effect_inter)[ind_pair] , w_pos + 3 , 
                       nchar(names(net_effect_inter)[ind_pair]) )

  ## Diferencia entre el efecto de la interacción menos los 2 efectos individuales
  net_effect_inter[ind_pair] <- contrib_first_inter$contrib_pair[ind_pair] -
                                contrib_first [name_var1] - 
                                contrib_first [name_var2]

}

###########################
## Step 3: Se decide el orden de entrada, como efecto individual o interaccción

## Unimos los efectos principales y los de la interacción
effect_all <- data.frame( effect     = c( net_effect_inter, contrib_first ),
                          effect_abs = abs(c( net_effect_inter, contrib_first)))


## Determinamos el orden de entrada para el cálculo final de las contribuciones
## Primero se ordenan
effect_ord <- effect_all[order(effect_all$effect_abs, decreasing = TRUE), ]

vars <- rownames(effect_ord)
eliminar <- rep(FALSE, length(vars))

## - Si es un efecto individual, se borran todas las interacciones donde participa
## - Si es una interacción se borran los 2 efectos individuales

for (i in seq_along(vars)) {
  if (eliminar[i]) next

  v <- strsplit(vars[i], " : ", fixed = TRUE)[[1]]

  for (j in seq.int(i + 1, length(vars))) {
    posteriores <- strsplit(vars[j], " : ", fixed = TRUE)[[1]]

    if (any(v %in% posteriores)) {
      eliminar[j] <- TRUE
    }
  }
}

effect_filtrado <- effect_ord[!eliminar, ]
effect_filtrado 

## Finalmente, decidido el orden, se hace el cálculo de los efectos que van 
## entrando como individuales o pareja
calc_contrib_indiv_or_pair(mod_rf, pepito, 
                           xx_hd_all, row.names(effect_filtrado))


##########################################################################
## Construir un data frame uniendo todo, para explicar un caso concreto

df1 <- data.frame(Var=names(contrib_first), Contr_First_Indiv = contrib_first)
w2 <- contrib_first_inter$contrib_pair 
df2 <- data.frame(Var=names(w2), Contr_First_Inter = w2)
df3 <- data.frame(Var=names(net_effect_inter), Net_Eff_Inter = net_effect_inter)

df_all <- data.frame( Var = c(df1$Var, df2$Var) )
df_all <- merge( df_all, df2 , all.x=T )
df_all <- merge( df_all, df1 , all.x=T )
df_all <- merge( df_all, df3 , all.x=T )

df_all$Contr_Order <- NA
df_all$Contr_Order [ !is.na(df_all$Contr_First_Indiv) ] = 
   abs(df_all$Contr_First_Indiv) [ !is.na(df_all$Contr_First_Indiv) ]
df_all$Contr_Order [ !is.na(df_all$Net_Eff_Inter) ] = 
   abs(df_all$Net_Eff_Inter) [ !is.na(df_all$Net_Eff_Inter) ]

print(df_all[ order(df_all$Contr_Order, decreasing=T) , 1:4 ], row.names = FALSE)



################################################################################
################################################################################
## 1.3.- Shappley Additive (SHAP). Explaining the Contributions of the Variables
################################################################################
################################################################################

################################################################################
## Shappley Additive (SHAP) - Random Forest

shap_rf_pep <-  predict_parts(explainer = exp_rf, new_observation = pepito,
                              type = "shap", B = 50 )
shap_rf_pep

dev.new()
plot(shap_rf_pep, show_boxplots = F)  


################################################################################
## Explaining Shappley Additive (SHAP)

calc_contrib_indiv(mod_rf, pepito, xx_hd_all, name_pred)$contrib_ind 

results <- NULL
B = 50
for( i in 1:B)
{ # Random the order of the variables
  names_random <- sample(name_pred)
  out <- calc_contrib_indiv(mod_rf, pepito, xx_hd_all, names_random)$contrib_ind
  results <- c(results, out)                                                     
}

df_summary <- data.frame(
  variable = names(results),
  value = as.numeric(results)
) %>%
  group_by(variable) %>%
  summarise(
    mean = mean(value),
    min  = min(value),
    max  = max(value),
    .groups = "drop"
  )

df_summary
shap_rf_pep



################################################################################
################################################################################
## 1.4.- Ceteris-paribus Profiles
################################################################################
################################################################################

observ_select <- pepito
#observ_select <- juanita

## Observación a explicar
predict_obs <- predict(mod_rf, observ_select, type="prob")[ , "Yes"]
  
## "oldpeak" a explorar: valores observados en el dataframe 
oldpeak_to_explore <- sort(unique(xx_hd_all$oldpeak))
changed_predict <- NULL

## Se calculan las predicciones de RF para observaciones iguales a la observación
## seleccionada (Pepito), cambiando solo "oldpeak"
for(changed_oldpeak in oldpeak_to_explore)
{ changed_obs         <- observ_select      ## se copia la observación
  changed_obs$oldpeak <- changed_oldpeak    ## solo se cambia "oldpeak"
  changed_predict <- c( changed_predict,
                        predict(mod_rf, changed_obs, type="prob")[ ,"Yes"])
}

## Plot de línea de la predicción respecto a oldpeak, a la que se añade un punto
## para la observación de interés, Pepito
df_pred_oldpeak <- data.frame(oldpeak_to_explore = oldpeak_to_explore, 
                              changed_predict = changed_predict ) 

dev.new()
ggplot(data = df_pred_oldpeak, aes(x = oldpeak_to_explore, y = changed_predict)) +
  geom_line() + 
  geom_point( x = pepito$oldpeak, y = predict_obs, size = 3) +
  theme_bw()



################################################################################
################################################################################
################################################################################
## 2.- Técnicas de Explicación Global
################################################################################
################################################################################
################################################################################


################################################################################
################################################################################
## 2.1.- Medidas de Importancia de las Variables
################################################################################
################################################################################

## Probabilidades predichas en la muestra de testing y cálculo del AUC
pred_prob_test <- predict(mod_rf, xx_hd_test, type="prob")[ , "Yes"]
auc_test <- as.numeric(roc(xx_hd_test$diagnosis, pred_prob_test)$auc)
auc_test

## Vectores para salvar los resultados: AUC y nombre de la variable permutada
auc_perm <- NULL
name_var_perm <- NULL

num_obs <- nrow(xx_hd_test)
B=100

## Bucle por permutación
for( perm in 1:B )
{ ## Permuta las filas
  rows_perm <- sample(1:num_obs)
  
  ## Bucle por variable
  for( name_var in name_pred )
  { 
    ## Se hace una copia del df y se permuta las filas de una única variable
    new_df_perm <- xx_hd_test 
    new_df_perm[ , name_var ] <- xx_hd_test[ rows_perm , name_var ]
    
    ## Probabilidades predichas permutando la variable y cálculo del AUC
    pred_prob_perm <- predict(mod_rf, new_df_perm, type="prob")[ , "Yes"]
    auc_perm <- c( auc_perm, 
        as.numeric(roc(new_df_perm$diagnosis, pred_prob_perm, quiet = TRUE)$auc))
    name_var_perm <- c( name_var_perm , name_var )
  }
}

## loss function: pérdida en AUC al permutar la variable                                                                                       
loss_auc <- auc_test - auc_perm

## data frame para mostrar los resultados
res_perm <- data.frame( variable = name_var_perm, 
                        auc_perm = auc_perm, loss_auc = loss_auc ) 
dim(res_perm)
head(res_perm)

res_perm %>%
  group_by(variable) %>%
  summarise( mean_auc = mean(auc_perm),
             mean_loss = mean(loss_auc) ) %>%
  arrange(desc(mean_loss))


######################################                
## Feature Importance con DALEX
exp_rf_test   <- DALEX::explain( mod_rf, data = xx_hd_test[ , - col_resp], 
                                 y = xx_hd_test$diagnosis == "Yes", label = "RF")

imp_rf <- model_parts(exp_rf_test, loss_function = loss_one_minus_auc, B = 50, 
                      type="difference" )
imp_rf

#dev.new()
#plot(imp_rf, bar_width = 6, show_boxplots = TRUE)


################################################################################
################################################################################
## 2.2.- Partial-Dependence Profiles (PDP)
################################################################################
################################################################################

## Valores a explorar: valores observados en el dataframe
old_to_explore <- sort(unique(xx_hd_all$oldpeak))

## Matriz preparada para las predicciones
predict_matrix <- matrix( NA, nrow(xx_hd_all), length(old_to_explore) )

## Bucle para las observaciones
for(w_row in 1:nrow(xx_hd_all))
{
  ## Copia la observación tantas veces como valores de oldpeak
  newdata <- xx_hd_all[ rep(w_row, length(old_to_explore)) , ]

  ## Sustituye oldpeak por todos los valores a explorar
  newdata$oldpeak <- old_to_explore

  ## Una única llamada a predict()
  predict_matrix[w_row, ] <- predict( mod_rf, newdata, type = "prob" )[,"Yes"]
}

## Cálculo del Partial-Dependence Profile promediando las predicciones de todas
## las observaciones, para cada valor
average_predict <- colMeans(predict_matrix)

df_pred_old <- data.frame(
  old_to_explore = old_to_explore,
  average_predict = average_predict ) 

## Plot de línea de las medias de las predicciones respecto a la variable (pag.212)
df_pred_old <- data.frame(old_to_explore = old_to_explore, 
                          average_predict = average_predict ) 
dev.new()
ggplot(data = df_pred_old, aes(x = old_to_explore, y = average_predict)) +
  geom_line() + 
  theme_bw()


## Partial Dependence Profiles de DALEX con toda la muestra
pdp_rf_old <- model_profile(explainer = exp_rf, type = "partial", N = NULL,
                            variables = "oldpeak") 
dev.new()
plot(pdp_rf_old) + 
  ggtitle("Partial-dependence profile for oldpeak")                      



################################################################################
################################################################################
## 2.3.- Accumulated Local Effects (ALE)
################################################################################
################################################################################

## Accumulated Local Effects de DALEX con toda la muestra
ale_rf <- model_profile(explainer = exp_rf, type = "accumulated", N = NULL,
                        variables = "age")
dev.new()
g5 <- plot(ale_rf) + ggtitle("Accumulated Local Profiles")
print(g5)


################################################################################
## Explicando Accumulated Local Effects (ALE)
################################################################################

## Número de intervalos
n_bins <- 15

## Límites de los intervalos de edad (deciles)
age_breaks <- quantile(xx_hd_all$age,
                       probs = seq(0, 1, length.out = n_bins + 1),
                       na.rm = TRUE)
age_breaks
length(age_breaks)

age_breaks <- unique(age_breaks) ## Por si hay deciles repetidos

## Intervalo al que pertenece cada observación
age_bin <- cut(xx_hd_all$age,
               breaks = age_breaks,
               include.lowest = TRUE,
               labels = FALSE)
length(age_bin)
table(age_bin)


################################################################################
## Cambio local de la predicción en cada intervalo
################################################################################

## Para salvar las predicciones medias en los límites de cada intervalo, y el cambio
predict_low_mean  <- numeric(length(age_breaks) - 1)
predict_high_mean <- numeric(length(age_breaks) - 1)
average_change    <- numeric(length(age_breaks) - 1)       ## longitud = n_bins 
length(average_change)

## Para cada intervalo
for (j in 1:(length(age_breaks) - 1))
{
  ## Observaciones que están en el intervalo j (número de fila en dataframe)
  observations <- which(age_bin == j)

  ## 2 Data frames, copia con todas las observaciones de ese intervalo
  changed_obs_low  <- xx_hd_all[observations, ]
  changed_obs_high <- xx_hd_all[observations, ]

  ## En las observaciones del intervalo, solo cambia la Edad en los dos extremos del intervalo
  changed_obs_low$age  <- age_breaks[j]
  changed_obs_high$age <- age_breaks[j + 1]

  ## Predicciones en los dos data frames, con edades extremas
  predict_low <- predict(mod_rf, changed_obs_low, type = "prob")[, "Yes"]
  predict_high <- predict(mod_rf, changed_obs_high, type = "prob")[, "Yes"]

  predict_low_mean[j]  <- mean(predict_low,  na.rm = TRUE)
  predict_high_mean[j] <- mean(predict_high, na.rm = TRUE)

  ## Cambio locales de la predicción
  changed_predict <- predict_high - predict_low

  ## Promedio del cambio local en el intervalo
  average_change[j] <- mean(changed_predict, na.rm = TRUE)
}
round(average_change, 3)


################################################################################
## Acumulación y centrado de los efectos locales
################################################################################

## Efectos acumulados
ale <- c(0, cumsum(average_change))

## Centro de los efectos:
## para cada observación se toma el valor ALE correspondiente a su intervalo
#ale_for_obs <- ale[age_bin + 1]

## Se centra el ALE para que su media sea 0
#ale <- ale - mean(ale_for_obs, na.rm = TRUE)
ale_centreded <- ale - mean(ale, na.rm = TRUE)      ## Reproduce mejor el resultado de DALEX

## Se vuelve a la escala de probabilidad añadiendo la predicción media
mean_prediction <- mean( predict(mod_rf, xx_hd_all, type = "prob")[, "Yes"] )

ale_scaled <- ale_centreded + mean_prediction


################################################################################
## Plots para explicar el ALE:
## 1.- Gráfico de segmentos por intervalo ? muestra el cálculo local.
## 2.- ALE acumulado: muestra cómo se suman los efectos.
## 3.- ALE centrado: muestra el ajuste vertical.
## 4.- ALE final (scaled): muestra la curva interpretada como predicción.
################################################################################


## df para el gráfico
df_step <- data.frame(
  interval = 1:(length(age_breaks) - 1),
  age_low  = age_breaks[-length(age_breaks)],
  age_high = age_breaks[-1],
  pred_low_mean  = predict_low_mean,
  pred_high_mean = predict_high_mean,
  average_change = average_change
)



## 1.- Plot cambios en predicción
dev.new()
g1 <- ggplot(df_step) +
  # Segmento entre pred_low_mean y pred_high_mean
  geom_segment(aes(x = age_low, xend = age_high,
                   y = pred_low_mean, yend = pred_high_mean),
               color = "grey40") +
  
  # Punto en el límite inferior
  geom_point(aes(x = age_low, y = pred_low_mean),
             color = "blue", size = 2) +
  
  # Punto en el límite superior
  geom_point(aes(x = age_high, y = pred_high_mean),
             color = "red", size = 2) +
  
  # Texto con average_change encima del segmento
  geom_text(aes(x = (age_low + age_high)/2,
                y = (pred_low_mean + pred_high_mean)/2,
                label = round(average_change, 3)),
            vjust = -0.5, size = 3) +
  
  labs(title = "Cambios locales de predicción por intervalo (ALE - Paso 1)",
       x = "Age",
       y = "Predicción media en los extremos del intervalo") +
  theme_bw()
print(g1)

## df para ALE
df_ale <- data.frame(
  age            = age_breaks,
  ale            = ale,
  ale_centreded  = ale_centreded,
  ale_scaled     = ale_scaled
)


## 2.- ALE acumulado (sin centrar)
dev.new()
g2 <- ggplot(df_ale) +
  geom_line(aes(x = age, y = ale), color = "darkorange", linewidth = 1) +
  geom_point(aes(x = age, y = ale), color = "darkorange") +
  labs(title = "ALE acumulado (sin centrar)",
       x = "Age",
       y = "Efecto acumulado") +
  theme_bw()
print(g2)


## 3.- ALE centrado
dev.new()
g3 <- ggplot(df_ale) +
  geom_line(aes(x = age, y = ale_centreded), color = "purple", linewidth = 1) +
  geom_point(aes(x = age, y = ale_centreded), color = "purple") +
  geom_hline(yintercept = 0, linewidth = 0.8) + 
  labs(title = "ALE centrado",
       x = "Age",
       y = "Efecto acumulado centrado") +
  theme_bw()
print(g3)

## 4.- ALE Profiles (escala del modelo)
dev.new()
g4 <- ggplot(df_ale) +
  geom_line(aes(x = age, y = ale_scaled), color = "blue", linewidth = 1) +
  geom_point(aes(x = age, y = ale_scaled), color = "blue") +
  labs(title = "ALE Profiles (escala del modelo)",
       x = "Age",
       y = "Predicción ajustada") +
  theme_bw()
print(g4)

## Los 4 gráficos juntos
library(ggpubr)
dev.new()
ggarrange(g2, g3, g4, g5, ncol = 2, nrow = 2)



