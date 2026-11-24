################################################################################
################################################################################
## Taller: Explicando Predicciones y Modelos de Machine Learning
##
## Script: Explicando Predicciones y Modelos de Machine Learning. Statlog Heart
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
## Librarías
################################################################################

## Librerías generales
library(tidyverse)
library(gridExtra)
library(lattice)

## Librerías construcción de modelos predictivos
library(rpart)
library(randomForest)
library(nnet)
library(e1071)
library(caret)

## Librerías Explicación de Predicciones y Modelos 
library(DALEX)
library(DALEXtra)    ## función predict_surrogate() LIME
library(lime)
library(localModel)
library(iml)
library(vivo)
library(hstats)
library(pdp)


################################################################################
## Lectura de Datos. Heart Disease
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
name_pred <- names(xx_hd_all)[ - col_resp ]         ## Nombre Predictores
name_pred_cont  <- c( "age", "blood_press", "chol","max_heart_rate", "oldpeak" )
name_pred_categ <- c( "sex", "chest_pain", "sugar", "ecg", "angina",
                       "slope", "major_vessels", "thal" )


################################################################################
## Explainer sobre la muestra completa
################################################################################

exp_rlog <- DALEX::explain( mod_rlog, data = xx_hd_all[ , - col_resp],
                            y = xx_hd_all$diagnosis == "Yes", label = "RLog")
exp_cart <- DALEX::explain( mod_cart, data = xx_hd_all[ , - col_resp], 
                             y = xx_hd_all$diagnosis == "Yes", label = "CART") 
exp_nn   <- DALEX::explain( mod_nn, data = xx_hd_all[ , - col_resp], 
                            y = xx_hd_all$diagnosis == "Yes", label = "NN")
exp_svm  <- DALEX::explain( mod_svm, data = xx_hd_all[ , - col_resp], 
                            y = xx_hd_all$diagnosis == "Yes", label = "SVM")    
exp_rf   <- DALEX::explain( mod_rf, data = xx_hd_all[ , - col_resp], 
                            y = xx_hd_all$diagnosis == "Yes", label = "RF")


################################################################################ 
################################################################################     
################################################################################ 
## 1.- Instance Level Explanations (Local Explanations)
################################################################################
################################################################################
################################################################################  

################################################################################
## Observaciones a explicar
################################################################################

## Probabilidad media del modelo de random forest
mean(predict(mod_rf, xx_hd_all, type="prob")[ , "Yes"])

## Observación 1: Pepito
(pepito <- as.data.frame(xx_hd_all[36, - col_resp ] ))
predict(mod_rf, pepito, type="prob")[ , "Yes"]

## Observación 2: Juanita
(juanita <- as.data.frame(xx_hd_all[15, - col_resp ] ))
predict(mod_rf, juanita, type="prob")[ , "Yes"]    



################################################################################
################################################################################
## Break-down Plots for Additive Attributions
################################################################################
################################################################################

## Break-down plot con el orden por defecto
bd_rf_pep <- predict_parts(explainer = exp_rf, new_observation = pepito,
                           type = "break_down" )
bd_rf_pep

bd_rf_jua <- predict_parts(explainer = exp_rf, new_observation = juanita,
                           type = "break_down" )
bd_rf_jua

## Gráfico. Se cambian los colores por defecto: verde cuando baja la 
## probabilidad, y rojo cuando sube
cols <- DALEX::colors_breakdown_drwhy()
cols[c("-1", "1")] <- cols[c("1", "-1")]

dev.new()
plot(bd_rf_pep, vcolors = cols)
dev.new()
plot(bd_rf_jua, vcolors = cols)


######################################
## Break-down plot cambiando el orden (según las columnas dentro del data frame)
bd_rf_pep_2 <- predict_parts(explainer = exp_rf, new_observation = pepito,
                             order = name_pred,
                             type = "break_down" )
dev.new()
plot(bd_rf_pep_2, vcolors = cols)


######################################
## Break-down plot con la distribuión condicional de las predicciones 
bd_rf_pep_3 <- predict_parts(explainer = exp_rf, new_observation = pepito,
                             type = "break_down", keep_distributions = TRUE )
dev.new()
plot(bd_rf_pep_3, vcolors = cols, plot_distributions = TRUE)
                                    

################################################################################
################################################################################
## Break-down Plots for Interactions
################################################################################
################################################################################

bd_rf_int_pep <- predict_parts(explainer = exp_rf, new_observation = pepito,
                               type = "break_down_interactions" )
bd_rf_int_pep

bd_rf_int_jua <- predict_parts(explainer = exp_rf, new_observation = juanita,
                               type = "break_down_interactions" )
bd_rf_int_jua

dev.new()
plot(bd_rf_int_pep, vcolors = cols)   
dev.new()
plot(bd_rf_int_jua, vcolors = cols)  


################################################################################
################################################################################
## Shapley Additive Explanations (SHAP)
################################################################################
################################################################################

shap_rf_pep <-  predict_parts(explainer = exp_rf, new_observation = pepito,
                              type = "shap", B = 25 )
shap_rf_pep

dev.new()
plot(shap_rf_pep, vcolors = cols)  

dev.new()
plot(shap_rf_pep, vcolors = cols, show_boxplots = F)  


################################################################################
################################################################################
## Local Interpretable Model-agnostic Explanations (LIME)
################################################################################
################################################################################

################################################################################
## LIME with the lime package

## estas 2 líneas están en el github
model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

lime_rf_pep_1 <- predict_surrogate( explainer = exp_rf, new_observation = pepito,
                                    n_features = 3, n_permutations = 1000,
                                    type = "lime" )
as.data.frame(lime_rf_pep_1)
dev.new()
plot(lime_rf_pep_1)


################################################################################
## LIME with the localModel package

lime_rf_pep_2 <- predict_surrogate( explainer = exp_rf, new_observation = pepito,
                                    size = 1000, type = "localModel" )
as.data.frame(lime_rf_pep_2)
dev.new()
plot(lime_rf_pep_2)

dev.new()
plot_interpretable_feature(lime_rf_pep_2, "oldpeak")
     

################################################################################
## LIME with the iml package

lime_rf_pep_3 <- predict_surrogate( explainer = exp_rf, new_observation = pepito,
                                    k = 3, type = "iml" )
lime_rf_pep_3$results[ , c(1:5,7)]

dev.new()
plot(lime_rf_pep_3)


################################################################################
################################################################################
## Ceteris-paribus Profiles
################################################################################
################################################################################

cp_rf_pep <- predict_profile(explainer = exp_rf, new_observation = pepito )
#cp_rf_pep
#dim(cp_rf_pep)
#table(cp_rf_pep$"_vname_")
#cp_rf_pep %>% filter( cp_rf_pep$"_vname_" == "major_vessels" )

## Variables continuas                             
dev.new()
plot(cp_rf_pep, variables = c("age", "blood_press", "chol", 
                              "max_heart_rate", "oldpeak") ) +
  ggtitle("Ceteris-paribus profile", "") + ylim(0.6, 1)

## Variables categóricas
dev.new()
plot(cp_rf_pep, variables = c( "major_vessels", "thal", "sex"), 
     variable_type = "categorical", categorical_type = "bars"  ) +
  ggtitle("Ceteris-paribus profile", "") 


## Dos observaciones juntas
cp_rf_2obs <- predict_profile(explainer = exp_rf,  
                              new_observation = rbind(pepito, juanita) )
                              
dev.new()
plot(cp_rf_2obs, color = "_ids_", variables = c("oldpeak", "max_heart_rate") ) +
  scale_color_manual(name = "Patient:", 
                     values = c("#8bdcbe", "#4387bf"),
                     labels = c("Juanita", "Pepito"))


## Comparación de modelos
cp_svm_pep <- predict_profile(explainer = exp_svm, new_observation = pepito )

dev.new()
plot(cp_rf_pep, cp_svm_pep, color = "_label_", 
     variables = c("oldpeak", "max_heart_rate")  ) +
  ggtitle("Ceteris-paribus Profiles for Pepito", "")      
                     
                     
################################################################################
################################################################################
## Ceteris-paribus Oscillations
################################################################################
################################################################################

## Oscilaciones con distribución uniforme
oscil_uni <- predict_parts(explainer = exp_rf, new_observation = pepito,
                           type = "oscillations_uni" )
#oscil_uni
oscil_uni$"_ids_" <- "Pepito"

dev.new()
plot(oscil_uni) +
  ggtitle("Ceteris-paribus Oscillations", "(Uniform distribution)")


## Oscilaciones con distribución empírica
oscil_emp <- predict_parts(explainer = exp_rf, new_observation = pepito,
                           type = "oscillations_emp" )
#oscil_emp
oscil_emp$"_ids_" <- "Pepito"

dev.new()
plot(oscil_emp) +
  ggtitle("Ceteris-paribus Oscillations", "(Empirical distribution)")


################################################################################
################################################################################
## Local-diagnostics plots
################################################################################
################################################################################

################################################################################
## Local-fidelity Plot
ldp_rf_pep <- predict_diagnostics(explainer = exp_rf, new_observation = pepito,
                                  neighbors = 30)
ldp_rf_pep
dev.new()
plot(ldp_rf_pep)


################################################################################
## Local-stability Plot
lsp_rf_old <- predict_diagnostics(explainer = exp_rf, new_observation = pepito,
                                  neighbors = 10,
                                  variables = c("oldpeak"))
#lsp_rf_old
dev.new()
plot(lsp_rf_old)

## Variable categórica
lsp_rf_ves <- predict_diagnostics(explainer = exp_rf, new_observation = pepito,
                                  neighbors = 10,
                                  variables = c("major_vessels"))
#lsp_rf_ves
dev.new()
plot(lsp_rf_ves)



################################################################################ 
################################################################################     
################################################################################ 
## 2.- Global Level Explanations
################################################################################
################################################################################ 
################################################################################ 

################################################################################
## Explainer sobre la muestra de testing
################################################################################

exp_rlog_test <- DALEX::explain( mod_rlog, data = xx_hd_test[ , - col_resp],
                                 y = xx_hd_test$diagnosis == "Yes", label = "RLog")
exp_cart_test <- DALEX::explain( mod_cart, data = xx_hd_test[ , - col_resp], 
                                 y = xx_hd_test$diagnosis == "Yes", label = "CART")    
exp_rf_test   <- DALEX::explain( mod_rf, data = xx_hd_test[ , - col_resp], 
                                 y = xx_hd_test$diagnosis == "Yes", label = "RF")
exp_nn_test   <- DALEX::explain( mod_nn, data = xx_hd_test[ , - col_resp], 
                                 y = xx_hd_test$diagnosis == "Yes", label = "NN")
exp_svm_test  <- DALEX::explain( mod_svm, data = xx_hd_test[ , - col_resp], 
                                 y = xx_hd_test$diagnosis == "Yes", label = "SVM")    


################################################################################
## Performance
################################################################################  

perm_rlog <- model_performance(exp_rlog_test)
perm_cart <- model_performance(exp_cart_test)
perm_rf <- model_performance(exp_rf_test)
perm_nn <- model_performance(exp_nn_test)
perm_svm <- model_performance(exp_svm_test)

## Tabla con todas las medidas
perf_mat <- rbind(unlist(perm_rlog$measures),
                  unlist(perm_cart$measures),
                  unlist(perm_rf$measures),
                  unlist(perm_nn$measures),
                  unlist(perm_svm$measures))
rownames(perf_mat) <- c("RLog","CART","RF","NN", "SVM")
kableExtra::kable(perf_mat, format = "simple")


################################################################################ 
################################################################################                  
## Feature Importance
################################################################################  
################################################################################ 

######################################                
## Feature Importance. Un modelo
imp_rf <- model_parts(exp_rf_test, loss_function = loss_one_minus_auc, B = 50, 
                      type="difference" )
imp_rf

dev.new()
plot(imp_rf, bar_width = 6, show_boxplots = TRUE)


######################################                 
## Feature Importance. Comparación de Modelos
imp_rlog <- model_parts(exp_rlog_test, type="difference")
imp_cart <- model_parts(exp_cart_test, type="difference")
imp_rf <- model_parts(exp_rf_test, type="difference")
imp_nn <- model_parts(exp_nn_test, type="difference")
imp_svm <- model_parts(exp_svm_test, type="difference")

dev.new()
plot(imp_rlog, imp_rf, imp_svm,
     max_vars = 8, bar_width = 4, show_boxplots = FALSE) +
     ggtitle("Feature Importance", "") 

## Alternativa al gráfico por defecto
p1 <- plot(imp_rlog, max_vars = 8, bar_width = 4, show_boxplots = FALSE)
p2 <- plot(imp_rf,   max_vars = 8, bar_width = 4, show_boxplots = FALSE)
p3 <- plot(imp_svm,  max_vars = 8, bar_width = 4, show_boxplots = FALSE)
p4 <- plot(imp_nn,   max_vars = 8, bar_width = 4, show_boxplots = FALSE)

dev.new()
grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2)

######################################
## Feature Importance. Ejemplo agrupando 2 variables

grupos <- c( 
  list("age + max_heart_rate" = c("age", "max_heart_rate")),
  as.list(setdiff(name_pred, c("age", "max_heart_rate")))
)
names(grupos)[-1] <- setdiff(name_pred, c("age", "max_heart_rate"))

imp_rf_2 <- model_parts(exp_rf_test, loss_function = loss_one_minus_auc, B = 50,
                        type = "difference",
                        variable_groups = grupos )
imp_rf_2

dev.new()
plot(imp_rf_2, bar_width = 6, show_boxplots = FALSE)


################################################################################ 
################################################################################ 
## Partial Dependence Profiles
################################################################################  
################################################################################ 

######################################
## Partial Dependence Profiles. Variable continua
pdp_rf_old <- model_profile(explainer = exp_rf, type = "partial", 
                            variables = "oldpeak") 

dev.new()
plot(pdp_rf_old) + 
  ggtitle("Partial-dependence profile for oldpeak")                      

dev.new()
plot(pdp_rf_old, geom = "profiles") + 
  ggtitle("Ceteris-paribus plots and partial-dependence profile for oldpeak")   


######################################                      
## Grouped Partial Dependence Profiles                      
pdp_rf_sex <- model_profile(exp_rf, type = "partial", 
                            variables = "oldpeak", groups = "sex" ) 
                
dev.new()
plot(pdp_rf_sex, geom = "profiles") + 
  ggtitle("Grouped partial-dependence profile for oldpeak")   


######################################                     
## Clustered Partial Dependence Profiles                      
pdp_rf_clust <- model_profile(exp_rf, type = "partial", 
                              variables = "oldpeak", k = 3 ) 
                
dev.new()
plot(pdp_rf_clust, geom = "profiles") + 
  ggtitle("Clustered  partial-dependence profile for oldpeak")   
                        
              
######################################
## Partial Dependence Profiles. Variables categóricas
pdp_rf_ves <- model_profile(explainer = exp_rf, variables = "major_vessels" ) 
pdp_rf_ves
g1 <- plot(pdp_rf_ves) + ggtitle("PDP for major_vessels")                      

pdp_rf_thal <- model_profile(explainer = exp_rf, variables = "thal" ) 
pdp_rf_thal
g2 <- plot(pdp_rf_thal) + ggtitle("PDP for thal")   

pdp_rf_sex <- model_profile(explainer = exp_rf, variables = "sex" ) 
pdp_rf_sex
g3 <- plot(pdp_rf_sex) + ggtitle("PDP for sex")   

dev.new()
grid.arrange(g1, g2, g3, nrow = 1, ncol = 3)


######################################                      
## Partial Dependence Profiles. Comparación de Modelos
pdp_rlog <- model_profile(exp_rlog, variables = name_pred_cont)$agr_profiles
pdp_cart <- model_profile(exp_cart, variables = name_pred_cont)$agr_profiles
pdp_rf   <- model_profile(exp_rf, variables = name_pred_cont)$agr_profiles
pdp_nn   <- model_profile(exp_nn, variables = name_pred_cont)$agr_profiles
pdp_svm  <- model_profile(exp_svm, variables = name_pred_cont)$agr_profiles

dev.new()
plot(pdp_rlog, pdp_rf, pdp_svm) +
  ggtitle("Contrastive partial-dependence profiles for selected variables","")


######################################                       
## Medidas de importancia de las variables basado en PDPs (paquete "vivo")        
prof_rf <- model_profile(exp_rf)   ## PDP de DALEX
imp_pdp <- global_variable_importance(prof_rf)
dev.new()
plot(imp_pdp)   


################################################################################
################################################################################ 
## Accumulated Local Profiles
################################################################################
################################################################################

######################################
## Partial Dependence, Local Dependence, Accumulated Local Profiles

######################################
## Random Forest
pdp_rf <- model_profile(exp_rf, variables = name_pred_cont, N = NULL, 
                        type = "partial" )
ale_rf <- model_profile(exp_rf, variables = name_pred_cont, N = NULL, 
                        type = "accumulated") 

## Todos los gráficos juntos
dev.new()
pdp_rf$agr_profiles$`_label_` = "partial dependence"
ale_rf$agr_profiles$`_label_` = "accumulated local"
plot(pdp_rf, ale_rf) 


######################################
## SVM
pdp_svm <- model_profile(exp_svm, variables = name_pred_cont, N = NULL, 
                        type = "partial")
ale_svm <- model_profile(exp_svm, variables = name_pred_cont, N = NULL, 
                        type = "accumulated") 

## Todos los gráficos juntos
dev.new()
pdp_svm$agr_profiles$`_label_` = "partial dependence"
ale_svm$agr_profiles$`_label_` = "accumulated local"
plot(pdp_svm, ale_svm) 


           
################################################################################
################################################################################
## Interacciones entre variables
################################################################################
################################################################################

################################################################################
## H-statistic
################################################################################

h_rf <- hstats( exp_rf, pairwise_m = length(name_pred), 
                n_max = 100, verbose = TRUE )

## H-statistic global y por parejas
#dev.new()
#plot(h_rf)

## H-statistic interacción global por variable
head(h2_overall(h_rf))   
## H-statistic por parejas
head(h2_pairwise(h_rf))

## H-statistic por parejas
dev.new()
plot(h2_pairwise(h_rf))


## Interacciones de orden 3
h_rf_3ways <- hstats( exp_rf, pairwise_m = 4, threeway_m = 4,
                      n_max = 100, verbose = TRUE )


################################################################################
## Interpretación de la Interacción. Se usan funciones del paquete "pdp"
################################################################################
                     
## PDP para las 2 variables juntas 
pdp_2d <- partial( object = mod_rf, pred.var = c("age", "max_heart_rate"),
                   which.class = "Yes", prob = TRUE )
head(pdp_2d)                
dev.new()
plotPartial(pdp_2d, levelplot = TRUE) 

## PDPs marginales
pdp_age <- partial( mod_rf, pred.var = "age", which.class = "Yes", prob = TRUE )
dev.new()
plotPartial(pdp_age, levelplot = TRUE) 

pdp_mhr <- partial( mod_rf, pred.var = "max_heart_rate", which.class = "Yes", 
                    prob = TRUE )
dev.new()
plotPartial(pdp_mhr, levelplot = TRUE) 

## Añade los PDP marginales al PDP 2D para calcular la superficie de interacción
int_surface <-
  pdp_2d %>%
  left_join( pdp_age %>% select(age, yhat_age = yhat), by = "age" ) %>%
  left_join( pdp_mhr %>% select(max_heart_rate, yhat_mhr = yhat), 
                         by = "max_heart_rate" )    

# Calcula la referencia, media de las predicciones en el plot 2D
mean_yhat <- mean(pdp_2d$yhat)

## El signo de mean_yhat es + ya que los PDP marginales ya contienen el intercept 
## implícitamente, y si no se añade se estaría restando dos veces
int_surface <- int_surface %>%
  mutate( interaction = yhat - yhat_age - yhat_mhr + mean_yhat )

## Plot de la interacción
w_max <- max(abs(int_surface$interaction))
dev.new()
levelplot(
  interaction ~ age * max_heart_rate,
  data = int_surface,
  at = seq(- w_max, w_max, length = 100),
  col.regions = colorRampPalette(c("darkblue", "white", "darkred"))(99) )

## Predicción Media
mean_yhat


#################################
## Ejemplo 1: chequeando, age = 70.38, max_heart_rate = 180.02
pred_age_70 = pdp_age$yhat[ pdp_age$age > 70.38461 ][1]
round(pred_age_70, 4)
pred_mhr_180 = pdp_mhr$yhat[ pdp_mhr$max_heart_rate == 180.02 ]
round(pred_mhr_180, 4)
pred_age_70_mhr_180 = pdp_2d$yhat[ pdp_2d$age > 70.38461 & 
                                   pdp_2d$max_heart_rate == 180.02 ][1]
round(pred_age_70_mhr_180, 4)
## Interacción = PDP conjunto - suma de efectos principales + media global
pred_age_70_mhr_180 - pred_age_70 - pred_mhr_180 + mean_yhat


#################################
## Ejemplo 2: chequeando, age = 70.38, max_heart_rate = 115.82
pred_age_70 = pdp_age$yhat[ pdp_age$age > 70.38461 ][1]
round(pred_age_70, 4)
pred_mhr_115 = pdp_mhr$yhat[ pdp_mhr$max_heart_rate == 115.82 ]
round(pred_mhr_115, 4)
pred_age_70_mhr_115 = pdp_2d$yhat[ pdp_2d$age > 70.38461 & 
                                   pdp_2d$max_heart_rate == 115.82 ][1]
round(pred_age_70_mhr_115, 4)
## Interacción = PDP conjunto - suma de efectos principales + media global
pred_age_70_mhr_115 - pred_age_70 - pred_mhr_115 + mean_yhat


 