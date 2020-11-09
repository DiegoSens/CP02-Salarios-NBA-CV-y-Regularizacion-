---
##### PRÁCTICA NBA SALARIOS 2 - PREDICCIÓN#####

rm(list = ls())

library(dplyr)
library(tidyverse)
library(ggplot2)

#Importamos y observamos los datos
datos_nba <- read.csv("nba.csv")
datos_nba
dim(datos_nba)
summary(datos_nba)

#Eliminación de NAs
datos_nba <- na.omit(datos_nba)

#Ver si hay duplicados
duplicated(datos_nba)
nrow(datos_nba[duplicated(datos_nba),])
datos_nba <- datos_nba[!duplicated(datos_nba$Player),]

#Eliminación de las variables categóricas
datos_nba <- datos_nba[c(-1,-3,-6)]

#####CROSS-VALIDATION##### Separa la muestra en parte de train y de test, para entrenar primero y predecir luego
#VALIDATION SET
library(rsample)
set.seed(123)
data_split <- initial_split(datos_nba, prob = 0.80, strata = Salary) #Separación de datos
data_train <- training(data_split) #Datos para entrenar
data_test  <-  testing(data_split) #Datos para testar
regres_train1 <- lm(Salary~., data = datos_nba) #Regresión 1
regres_train2 <- lm(Salary~ -PER-TS.-FTr, data = datos_nba) #Regresión 2
c(AIC(regres_train1),AIC(regres_train2)) #El mejor dato es el menor, de la primera regres por tanto

pred_1 <- predict(regres_train1,newdata = data_test)
MSE0 <- mean((data_test$Salary-pred_1)^2)
pred_2 <- predict(regres_train2,newdata = data_test)
MSE1 <- mean((data_test$Salary-pred_2)^2)
c(MSE0,MSE1)


#####Leave-One-Out Cross-Validation#####Estimar el modelo quitando un dato, repetido n veces.
library(glmnet)

#Consiste en estimar el modelo con todos los datos menos uno, y me ofrece los coeficientes
library(boot)
set.seed(123)
glm.fit1 = glm(Salary~., datos_nba, family = gaussian())
coef(glm.fit1)
cv.err <- cv.glm(datos_nba, glm.fit1)
cv.err$delta


glm.fit2=glm(Salary~-PER-TS.-FTr,datos_nba,family = gaussian())
cv.err2 =cv.glm(datos_nba,glm.fit2)
cv.err2$delta


#####K-Fold Cross-Validation##### Dividir la muestra en grupos o folds de igual tamaño
set.seed(123)
cv.err =cv.glm(datos_nba,glm.fit1,K=10)
cv.err$delta

glm.fit2=glm(Salary~.-MP-PER-TS.,datos_nba,family = gaussian())
cv.err2 =cv.glm(datos_nba,glm.fit2,K=10)
cv.err2$delta


#####Shrinkage Methods (Métodos de contracción)#####
#ajustamos un modelo que contiene todos los p predictores usando una técnica que restringe o regulariza las estimaciones de coeficientes, o equivalentemente, que reduce las estimaciones de coeficientes hacia cero.
#La reducción de las estimaciones de los coeficientes tiene el efecto de reducir significativamente su varianza. Las dos técnicas más conocidas para reducir las estimaciones de coeficientes hacia cero son la regresión de cresta (Ridge) y el lazo (Lasso).

#####REGULARIZACIÓN#####
#La regresión Ridge es similar a los mínimos cuadrados, excepto que los coeficientes se estiman minimizando una cantidad ligeramente diferente.
#Esta penalización tiene el efecto de reducir las estimaciones del coeficiente hacia cero.

library(rsample)  # data splitting 
library(glmnet)   # implementing regularized regression approaches
library(dplyr)    # basic data manipulation procedures
library(ggplot2)  # plotting

set.seed(123)
ames_split <- initial_split(datos_nba, prop = .7, strata = "Salary")
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

# Create training and testing feature model matrices and response vectors.
# we use model.matrix(...)[, -1] to discard the intercept
ames_train_x <- model.matrix(Salary~., ames_train)[, -1]
ames_train_y <- log(ames_train$Salary)

ames_test_x <- model.matrix(Salary ~., ames_test)[, -1]
ames_test_y <- log(ames_test$Salary)


#Es fundamental que las variables independientes (x’s) estén estandarizadas al realizar una regresión regularizada. glmnet realiza esto por nosotros. 
ames_ridge <- glmnet(
  x = ames_train_x,
  y = ames_train_y,
  alpha = 0
)


#####Tuning \(\lambda\)#####
# Apply CV Ridge regression to ames data
ames_ridge_cv <- cv.glmnet(
  x = ames_train_x,
  y = ames_train_y,
  alpha = 0
)

# plot results
plot(ames_ridge_cv)
#Para cada lambda que tiene estima el modelo. Divide la muestra en test y train 
#y el train lo vuelve  a dividir. La línea roja es el error de predicción y el 
#error va subiendo. El modelo más adecuado sería es que minimiza los errores, 
#que es la primera banda.

min(ames_ridge_cv$cvm)       # minimum MSE

ames_ridge_cv$lambda.min     # lambda for this min MSE

log(ames_ridge_cv$lambda.min)

ames_ridge_cv$cvm[ames_ridge_cv$lambda == ames_ridge_cv$lambda.1se]  # 1 st.error of min MSE

ames_ridge_cv$lambda.1se  # lambda for this MSE

log(ames_ridge_cv$lambda.1se)

plot(ames_ridge, xvar = "lambda")
abline(v = log(ames_ridge_cv$lambda.1se), col = "red", lty = "dashed")
#La línea roja dice cual es el mejor modelo y lo que esta cogiendo. De todos los posibles, elijo el más simple.


#####Ventajas y Desventajas#####
coef(ames_ridge_cv, s = "lambda.1se") %>%
  broom::tidy() %>%
  filter(row != "(Intercept)") %>%
  top_n(25, wt = abs(value)) %>%
  ggplot(aes(value, reorder(row, value))) +
  geom_point() +
  ggtitle("Top 25 influential variables") +
  xlab("Coefficient") +
  ylab(NULL)


#####LASSO#####
## Apply lasso regression to ames data: alpha=1
ames_lasso <- glmnet(
  x = ames_train_x,
  y = ames_train_y,
  alpha = 1
)

plot(ames_lasso, xvar = "lambda")

#####Tuning - CV#####
# Apply CV Ridge regression to ames data
ames_lasso_cv <- cv.glmnet(
  x = ames_train_x,
  y = ames_train_y,
  alpha = 1
)
# plot results
plot(ames_lasso_cv)

min(ames_lasso_cv$cvm)       # minimum MSE

ames_lasso_cv$lambda.min     # lambda for this min MSE

ames_lasso_cv$cvm[ames_lasso_cv$lambda == ames_lasso_cv$lambda.1se]  # 1 st.error of min MSE

ames_lasso_cv$lambda.1se  # lambda for this MSE

plot(ames_lasso, xvar = "lambda")
abline(v = log(ames_lasso_cv$lambda.min), col = "red", lty = "dashed")
abline(v = log(ames_lasso_cv$lambda.1se), col = "red", lty = "dashed")


#####Ventajas y Desventajas#####
coef(ames_lasso_cv, s = "lambda.1se") %>%
  tidy() %>%
  filter(row != "(Intercept)") %>%
  ggplot(aes(value, reorder(row, value), color = value > 0)) +
  geom_point(show.legend = FALSE) +
  ggtitle("Influential variables") +
  xlab("Coefficient") +
  ylab(NULL)

# minimum Ridge MSE
min(ames_ridge_cv$cvm)

# minimum Lasso MSE
min(ames_lasso_cv$cvm)


#####Elastic Net (Red elástica)#####
#La red elástica es otra penalización que incorpora la selección variable del lazo y la contracción de predictores correlacionados como la regresión de ridge.
lasso    <- glmnet(ames_train_x, ames_train_y, alpha = 1.0) 
elastic1 <- glmnet(ames_train_x, ames_train_y, alpha = 0.25) 
elastic2 <- glmnet(ames_train_x, ames_train_y, alpha = 0.75) 
ridge    <- glmnet(ames_train_x, ames_train_y, alpha = 0.0)

par(mfrow = c(2, 2), mar = c(6, 4, 6, 2) + 0.1)
plot(lasso, xvar = "lambda", main = "Lasso (Alpha = 1)\n\n\n")
plot(elastic1, xvar = "lambda", main = "Elastic Net (Alpha = .25)\n\n\n")
plot(elastic2, xvar = "lambda", main = "Elastic Net (Alpha = .75)\n\n\n")
plot(ridge, xvar = "lambda", main = "Ridge (Alpha = 0)\n\n\n")


#####Tuning#####
# maintain the same folds across all models
fold_id <- sample(1:10, size = length(ames_train_y), replace=TRUE)

# search across a range of alphas
tuning_grid <- tibble::tibble(
  alpha      = seq(0, 1, by = 0.01),
  mse_min    = NA,
  mse_1se    = NA,
  lambda_min = NA,
  lambda_1se = NA
)
tuning_grid
#--
for(i in seq_along(tuning_grid$alpha)) {
  
  # fit CV model for each alpha value
  fit <- cv.glmnet(ames_train_x, ames_train_y, alpha = tuning_grid$alpha[i], foldid = fold_id)
  
  # extract MSE and lambda values
  tuning_grid$mse_min[i]    <- fit$cvm[fit$lambda == fit$lambda.min]
  tuning_grid$mse_1se[i]    <- fit$cvm[fit$lambda == fit$lambda.1se]
  tuning_grid$lambda_min[i] <- fit$lambda.min
  tuning_grid$lambda_1se[i] <- fit$lambda.1se
}

tuning_grid
#--
tuning_grid %>%
  mutate(se = mse_1se - mse_min) %>%
  ggplot(aes(alpha, mse_min)) +
  geom_line(size = 2) +
  geom_ribbon(aes(ymax = mse_min + se, ymin = mse_min - se), alpha = .25) +
  ggtitle("MSE ± one standard error")


#####Predicción#####
# some best model
cv_lasso   <- cv.glmnet(ames_train_x, ames_train_y, alpha = 1.0)
min(cv_lasso$cvm)

# predict
pred <- predict(cv_lasso, s = cv_lasso$lambda.min, ames_test_x)
mean((ames_test_y - pred)^2)


