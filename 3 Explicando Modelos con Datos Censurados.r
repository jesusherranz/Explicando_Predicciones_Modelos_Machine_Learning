################################################################################
################################################################################
## Taller: Explicando Predicciones y Modelos de Machine Learning
##
## Script: Explicando Modelos con Datos Censurados
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

## Librerías generales
library(tidyverse)
library(survival)
library(randomForestSRC)
library(DALEX)


################################################################################
## Funciones predict() 
################################################################################

## Predicción de la supervivencia a 2 años del modelo de Regresión de Cox
pred_cox_surv_2y <- function(model, newdata) {

  surv <- survfit( model, newdata = newdata, se.fit = FALSE )

  as.vector( summary(surv, times = 2)$surv )
}

## Predicción de la supervivencia a 2 años del modelo de Random Survival Forest
pred_rsf_surv_2y <- function(model, newdata) {

  p <- predict(model, newdata)

  idx <- which.min( abs(p$time.interest - 2) )

  p$survival[, idx]
}


################################################################################
## Lectura de Datos. whas500.csv
################################################################################

## Lectura del fichero
path = "C://Taller Explicando Predicciones y Modelos de Machine Learning/Datos/"
xx_wh_all <- read.delim( paste0(path, "whas500.csv"), sep= ";" )
dim(xx_wh_all)
range(xx_wh_all$lenfol)

## Transformación, de días a años
head(xx_wh_all$lenfol)
xx_wh_all$lenfol <- xx_wh_all$lenfol / 365.25
head(xx_wh_all$lenfol)

## Variables categóricas
xx_wh_all <- xx_wh_all %>%
  select( age, gender, hr, sysbp, diasbp, bmi, cvd, afb, sho, chf, av3, miord,
          mitype, lenfol, fstat ) %>%
  mutate(
    gender = factor(gender, levels = c(0,1), labels = c("Hombre", "Mujer")),
    cvd = factor(cvd, levels = c(0,1), labels = c("No enf.card.", "Si enf.card.")),
    afb = factor(afb, levels = c(0,1), labels = c("No fibril.", "Si fibril.")),
    sho = factor(sho, levels = c(0,1), labels = c("No shock", "Si shock")),
    chf = factor(chf, levels = c(0,1), labels = c("No complic.", "Si complic.")),
    av3 = factor(av3, levels = c(0,1), labels = c("No bloqueo", "Si bloqueo")),            
    miord = factor(miord, levels = c(0,1), labels = c("Primero", "Recurrente")),    
    mitype = factor(mitype, levels = c(0,1), labels = c("No Q-wave", "Si Q-wave"))     
  )
head(xx_wh_all)


# Columnas y nombres de las variables
col_resp_stat <- which(names(xx_wh_all) == "fstat")   ## Columna status
col_resp_time <- which(names(xx_wh_all) == "lenfol")  ## Columna time

col_pred <- names(xx_wh_all)[ - c(col_resp_stat, col_resp_time) ]


################################################################################
## Construcción de los Modelos Predictivos
################################################################################

## Modelo de Regresión de Cox con todas las variables
mod_cox <- coxph( Surv(lenfol, fstat) ~ . , data=xx_wh_all )

## Modelo de Random Survival Forest con todas las variables
mod_rsf <- rfsrc( Surv(lenfol, fstat) ~ ., data = xx_wh_all, ntree = 1000 )


################################################################################
## Observación a explorar
################################################################################

pepito <- xx_wh_all[ 60 , ]
pepito

## Curva de supervivencia predicha por el modelo de Cox
dev.new()
surv_pep <- survfit( mod_cox, newdata=pepito, se=F ) 
plot( surv_pep, mark.time=F, conf.int=F, col="black", lwd=2, xlim=c(0,6) )
abline(v=2, lty=2, col="red", lwd=2)

## Predicción del modelo de Cox para la observación y Predicción media
pred_cox_surv_2y(mod_cox, pepito) 
mean(pred_cox_surv_2y(mod_cox, xx_wh_all) )

## Predicción de Random Survival Forest para la observación y Predicción media
pred_rsf_surv_2y(mod_rsf, pepito) 
mean(pred_rsf_surv_2y(mod_rsf, xx_wh_all) )


################################################################################
## Explainer 
################################################################################

## Explainer Modelo de Cox
exp_cox_2y <- explain( model = mod_cox,
                       data = xx_wh_all %>% select(-lenfol, -fstat), 
                       y = NULL,
                       predict_function = pred_cox_surv_2y,
                       label = "Survival at 2 yrs" )

## Explainer Random Survival Forest
exp_rsf_2y <- explain( model = mod_rsf,
                       data = xx_wh_all %>% select(-lenfol, -fstat),
                       y = NULL,
                       predict_function = pred_rsf_surv_2y,
                       label = "RSF Survival at 2 yrs" )


################################################################################
## Break-down Plots for Additive Attributions
## El breakdown explica probabilidad de supervivencia, el signo de cada
## contribución debe ser el "inverso" de los coeficientes del modelo

bd_cox_pep <- predict_parts( exp_cox_2y, 
                             new_observation = pepito %>% select(-lenfol, -fstat),
                             type = "break_down")
bd_cox_pep
dev.new()
plot(bd_cox_pep)


bd_rsf_pep <- predict_parts( exp_rsf_2y, 
                             new_observation = pepito %>% select(-lenfol, -fstat),
                             type = "break_down")
bd_rsf_pep
dev.new()
plot(bd_rsf_pep)

