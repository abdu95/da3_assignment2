# using the bisnode-firmd dataset
#########################################################################################



# ------------------------------------------------------------------------------------------------------
#### SET UP
# CLEAR MEMORY
rm(list=ls())

# Import libraries
library(haven)
library(glmnet)
library(purrr)
library(margins)
library(skimr)
library(kableExtra)
library(Hmisc)
library(cowplot)
library(gmodels) 
library(lspline)
library(sandwich)
library(modelsummary)
library(ggplot2)

library(rattle)
library(caret)
library(pROC)
library(ranger)
library(rpart)
library(partykit)
library(rpart.plot)
library(viridis)


# load theme and functions
source("codes/theme_bg.R")
source("codes/da_helper_functions.R")
options(digits = 3) 

data_in <- "data/clean/"
data_out <- "data/output/"
output <- "data/output/"


#-----------------------------------------------------------------------------------------

# USES INTERMEDIATE OUTPUT by data-prep.R


# Loading and preparing data ----------------------------------------------

# Use R format so it keeps factor definitions
# data <- read_csv(paste0(data_out,"bisnode_firms_clean.csv"))
data <- read_rds(paste(data_out,"bisnode_firms_clean.rds", sep = "/"))

#summary
datasummary_skim(data, type='numeric', histogram = TRUE)
# datasummary_skim(data, type="categorical")


# Define variable sets ----------------------------------------------
# (making sure we use ind2_cat, which is a factor)


rawvars <-  c("curr_assets", "curr_liab", "extra_exp", "extra_inc", "extra_profit_loss", "fixed_assets",
              "inc_bef_tax", "intang_assets", "inventories", "liq_assets", "material_exp", "personnel_exp",
              "profit_loss_year", "sales", "share_eq", "subscribed_cap")

engvar <- c("total_assets_bs", "fixed_assets_bs", "liq_assets_bs", "curr_assets_bs",
            "share_eq_bs", "subscribed_cap_bs", "intang_assets_bs", "extra_exp_pl",
            "extra_inc_pl", "extra_profit_loss_pl", "inc_bef_tax_pl", "inventories_pl",
            "material_exp_pl", "profit_loss_year_pl", "personnel_exp_pl")
engvar2 <- c("extra_profit_loss_pl_quad", "inc_bef_tax_pl_quad",
             "profit_loss_year_pl_quad", "share_eq_bs_quad")
engvar3 <- c(grep("*flag_low$", names(data), value = TRUE),
             grep("*flag_high$", names(data), value = TRUE),
             grep("*flag_error$", names(data), value = TRUE),
             grep("*flag_zero$", names(data), value = TRUE))
d1 <-  c("d1_sales_mil_log_mod", "d1_sales_mil_log_mod_sq",
         "flag_low_d1_sales_mil_log", "flag_high_d1_sales_mil_log")
hr <- c("female", "ceo_age", "flag_high_ceo_age", "flag_low_ceo_age",
        "flag_miss_ceo_age", "ceo_count", "labor_avg_mod",
        "flag_miss_labor_avg", "foreign_management")
firm <- c("age", "age2", "new", "ind2_cat", "m_region_loc", "urban_m")

# interactions for logit, LASSO
interactions1 <- c("ind2_cat*age", "ind2_cat*age2",
                   "ind2_cat*d1_sales_mil_log_mod", "ind2_cat*sales_mil_log",
                   "ind2_cat*ceo_age", "ind2_cat*foreign_management",
                   "ind2_cat*female",   "ind2_cat*urban_m", "ind2_cat*labor_avg_mod")
interactions2 <- c("sales_mil_log*age", "sales_mil_log*female",
                   "sales_mil_log*profit_loss_year_pl", "sales_mil_log*foreign_management")


X1 <- c("sales_mil_log", "sales_mil_log_sq", "d1_sales_mil_log_mod", "profit_loss_year_pl", "ind2_cat")
X2 <- c("sales_mil_log", "sales_mil_log_sq", "d1_sales_mil_log_mod", "profit_loss_year_pl", "fixed_assets_bs","share_eq_bs","curr_liab_bs ",   "curr_liab_bs_flag_high ", "curr_liab_bs_flag_error",  "age","foreign_management" , "ind2_cat")
X3 <- c("sales_mil_log", "sales_mil_log_sq", firm, engvar,                   d1)
X4 <- c("sales_mil_log", "sales_mil_log_sq", firm, engvar, engvar2, engvar3, d1, hr)
X5 <- c("sales_mil_log", "sales_mil_log_sq", firm, engvar, engvar2, engvar3, d1, hr, interactions1, interactions2)

# for LASSO
logitvars <- c("sales_mil_log", "sales_mil_log_sq", engvar, engvar2, engvar3, d1, hr, firm, interactions1, interactions2)

# for RF (no interactions, no modified features)
rfvars  <-  c("sales_mil", "d1_sales_mil_log", rawvars, hr, firm)

data$balsheet_notfullyear <- NULL

# Check simplest model X1
ols_modelx1 <- lm(formula(paste0("comp_growth ~", paste0(X1, collapse = " + "))),
                  data = data)
summary(ols_modelx1)

glm_modelx1 <- glm(formula(paste0("comp_growth ~", paste0(X1, collapse = " + "))),
                   data = data, family = "binomial")
summary(glm_modelx1)

# Check model X2
glm_modelx2 <- glm(formula(paste0("comp_growth ~", paste0(X2, collapse = " + "))),
                   data = data, family = "binomial")
summary(glm_modelx2)


#calculate average marginal effects (dy/dx) for logit
mx2 <- margins(glm_modelx2)

sum_table <- summary(glm_modelx2) %>%
  coef() %>%
  as.data.frame() %>%
  select(Estimate) %>%
  mutate(factor = row.names(.)) %>%
  merge(summary(mx2)[,c("factor","AME")])

kable(x = sum_table, format = "latex", digits = 3,
      col.names = c("Variable", "Coefficient", "dx/dy"),
      caption = "Average Marginal Effects (dy/dx) for Logit Model") %>%
  cat(.,file= paste0(output,"AME_logit_X2.tex"))


# baseline model is X4 (all vars, but no interactions) -------------------------------------------------------

ols_model <- lm(formula(paste0("comp_growth ~", paste0(X4, collapse = " + "))),
                data = data)
summary(ols_model)

glm_model <- glm(formula(paste0("comp_growth ~", paste0(X4, collapse = " + "))),
                 data = data, family = "binomial")
summary(glm_model)

#calculate average marginal effects (dy/dx) for logit
# vce="none" makes it run much faster, here we do not need variances

m <- margins(glm_model, vce = "none")

sum_table2 <- summary(glm_model) %>%
  coef() %>%
  as.data.frame() %>%
  select(Estimate, `Std. Error`) %>%
  mutate(factor = row.names(.)) %>%
  merge(summary(m)[,c("factor","AME")])

kable(x = sum_table2, format = "latex", digits = 3,
      col.names = c("Variable", "Coefficient", "SE", "dx/dy"),
      caption = "Average Marginal Effects (dy/dx) for Logit Model") %>%
  cat(.,file= paste0(output,"AME_logit_X4.tex"))



# separate datasets -------------------------------------------------------

set.seed(13505)

train_indices <- as.integer(createDataPartition(data$comp_growth, p = 0.8, list = FALSE))
data_train <- data[train_indices, ]
data_holdout <- data[-train_indices, ]

dim(data_train)
dim(data_holdout)

Hmisc::describe(data$comp_growth_f)
Hmisc::describe(data_train$comp_growth_f)
Hmisc::describe(data_holdout$comp_growth_f)

#######################################################x
# PART I PREDICT PROBABILITIES
# Predict logit models ----------------------------------------------
#######################################################x

# 5 fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryExtended,
  savePredictions = TRUE
)


# Train Logit Models ----------------------------------------------

logit_model_vars <- list("X1" = X1, "X2" = X2, "X3" = X3, "X4" = X4, "X5" = X5)

CV_RMSE_folds <- list()
logit_models <- list()


for (model_name in names(logit_model_vars)) {
  
  features <- logit_model_vars[[model_name]]
  
  set.seed(13505)
  glm_model <- train(
    formula(paste0("comp_growth_f ~", paste0(features, collapse = " + "))),
    method = "glm",
    data = data_train,
    family = binomial,
    trControl = train_control,
    na.action=na.exclude
  )
  
  logit_models[[model_name]] <- glm_model
  # Calculate RMSE on test for each fold
  CV_RMSE_folds[[model_name]] <- glm_model$resample[,c("Resample", "RMSE")]
  
}

CV_RMSE_folds 
# more predictor variables --> RMSE is growing
logit_models 


# Logit lasso -----------------------------------------------------------

lambda <- 10^seq(-1, -4, length = 10)
grid <- expand.grid("alpha" = 1, lambda = lambda)

set.seed(13505)
system.time({
  logit_lasso_model <- train(
    formula(paste0("comp_growth_f ~", paste0(logitvars, collapse = " + "))),
    data = data_train,
    method = "glmnet",
    preProcess = c("center", "scale"),
    family = "binomial",
    trControl = train_control,
    tuneGrid = grid,
    na.action=na.exclude
  )
})

tuned_logit_lasso_model <- logit_lasso_model$finalModel
best_lambda <- logit_lasso_model$bestTune$lambda
logit_models[["LASSO"]] <- logit_lasso_model
lasso_coeffs <- as.matrix(coef(tuned_logit_lasso_model, best_lambda))
write.csv(lasso_coeffs, paste0(data_out, "lasso_logit_coeffs.csv"))

CV_RMSE_folds[["LASSO"]] <- logit_lasso_model$resample[,c("Resample", "RMSE")]





#############################################x
# PART I
# No loss fn
########################################

# Draw ROC Curve and calculate AUC for each folds --------------------------------
CV_AUC_folds <- list()

for (model_name in names(logit_models)) {
  
  auc <- list()
  model <- logit_models[[model_name]]
  for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
    cv_fold <-
      model$pred %>%
      filter(Resample == fold)
    
    roc_obj <- roc(cv_fold$obs, cv_fold$fast)
    auc[[fold]] <- as.numeric(roc_obj$auc)
  }
  
  CV_AUC_folds[[model_name]] <- data.frame("Resample" = names(auc),
                                           "AUC" = unlist(auc))
}

# For each model: average RMSE and average AUC for models ----------------------------------

CV_RMSE <- list()
CV_AUC <- list()

for (model_name in names(logit_models)) {
  CV_RMSE[[model_name]] <- mean(CV_RMSE_folds[[model_name]]$RMSE)
  CV_AUC[[model_name]] <- mean(CV_AUC_folds[[model_name]]$AUC)
}

# X1 has the lowest RMSE
CV_RMSE
# X5 has the lowest AUC
CV_AUC

# We have 6 models, (5 logit and the logit lasso). For each we have a 5-CV RMSE and AUC.
# We pick our preferred model based on that. -----------------------------------------------

nvars <- lapply(logit_models, FUN = function(x) length(x$coefnames))
nvars[["LASSO"]] <- sum(lasso_coeffs != 0)

logit_summary1 <- data.frame("Number of predictors" = unlist(nvars),
                             "CV RMSE" = unlist(CV_RMSE),
                             "CV AUC" = unlist(CV_AUC))

# X1 has lowest CV RMSE
# X5 has lowest CV AUC

kable(x = logit_summary1, format = "latex", booktabs=TRUE,  digits = 3, row.names = TRUE,
      linesep = "", col.names = c("Number of predictors","CV RMSE","CV AUC")) %>%
  cat(.,file= paste0(output, "logit_summary1.tex"))

# Take best model and estimate RMSE on holdout  -------------------------------------------

best_logit_no_loss <- logit_models[["X5"]]


logit_predicted_probabilities_holdout <- predict(best_logit_no_loss, newdata = data_holdout, type = "prob")
data_holdout[,"best_logit_no_loss_pred"] <- logit_predicted_probabilities_holdout[,"fast"]
RMSE(data_holdout[, "best_logit_no_loss_pred", drop=TRUE], data_holdout$comp_growth)

# 0.051

# no this codes mean you’ve chosen X1 as the best model. it’s the result
# the fight between rmse and auc. I think (by googling) rmse matters more but yet we didn’t know how to argue for it


# discrete ROC (with thresholds in steps) on holdout -------------------------------------------------

# based on what we are defining this threshold? is this the right way?
thresholds <- seq(0.05, 0.75, by = 0.05)

cm <- list()
true_positive_rates <- c()
false_positive_rates <- c()
for (thr in thresholds) {
  holdout_prediction <- ifelse(data_holdout[,"best_logit_no_loss_pred"] < thr, "slow", "fast") %>%
    factor(levels = c("slow", "fast"))
  cm_thr <- confusionMatrix(holdout_prediction,data_holdout$comp_growth_f)$table
  cm[[as.character(thr)]] <- cm_thr
  true_positive_rates <- c(true_positive_rates, cm_thr["fast", "fast"] /
                             (cm_thr["fast", "fast"] + cm_thr["slow", "fast"]))
  false_positive_rates <- c(false_positive_rates, cm_thr["fast", "slow"] /
                              (cm_thr["fast", "slow"] + cm_thr["slow", "fast"]))
}

tpr_fpr_for_thresholds <- tibble(
  "threshold" = thresholds,
  "true_positive_rate" = true_positive_rates,
  "false_positive_rate" = false_positive_rates
)

discrete_roc_plot <- ggplot(
  data = tpr_fpr_for_thresholds,
  aes(x = false_positive_rate, y = true_positive_rate, color = threshold)) +
  labs(x = "False positive rate (1 - Specificity)", y = "True positive rate (Sensitivity)") +
  geom_point(size=2, alpha=0.8) +
  scale_color_viridis(option = "D", direction = -1) +
  scale_x_continuous(expand = c(0.01,0.01), limit=c(0,1), breaks = seq(0,1,0.1)) +
  scale_y_continuous(expand = c(0.01,0.01), limit=c(0,1), breaks = seq(0,1,0.1)) +
  theme_bg() +
  theme(legend.position ="right") +
  theme(legend.title = element_text(size = 4), 
        legend.text = element_text(size = 4),
        legend.key.size = unit(.4, "cm")) 
discrete_roc_plot
save_fig("figure-roc-discrete", data_out, "small")


# continuous ROC on holdout with best model (Logit 1) -------------------------------------------

roc_obj_holdout <- roc(data_holdout$comp_growth, data_holdout$best_logit_no_loss_pred)
createRocPlot(roc_obj_holdout, "best_logit_no_loss_roc_plot_holdout")

##### strange plot

# Confusion table with different tresholds ----------------------------------------------------------

# default: the threshold 0.5 is used to convert probabilities to binary classes
logit_class_prediction <- predict(best_logit_no_loss, newdata = data_holdout)
summary(logit_class_prediction)

table(data_holdout$comp_growth)

# confusion matrix: summarize different type of errors and successfully predicted cases
# positive = "yes": explicitly specify the positive case
cm_object1 <- confusionMatrix(logit_class_prediction, data_holdout$comp_growth_f, positive = "fast")
cm1 <- cm_object1$table
cm1



# we can apply different thresholds

# 0.5 same as before
holdout_prediction <-
  ifelse(data_holdout$best_logit_no_loss_pred < 0.5, "slow", "fast") %>%
  factor(levels = c("slow", "fast"))
cm_object1b <- confusionMatrix(holdout_prediction,data_holdout$comp_growth_f)
cm1b <- cm_object1b$table
cm1b

# a sensible choice: mean of predicted probabilities
mean_predicted_default_prob <- mean(data_holdout$best_logit_no_loss_pred)
mean_predicted_default_prob
# average probability of being 1 (fast)

holdout_prediction <-
  ifelse(data_holdout$best_logit_no_loss_pred < mean_predicted_default_prob, "slow", "fast") %>%
  factor(levels = c("slow", "fast"))
cm_object2 <- confusionMatrix(holdout_prediction,data_holdout$comp_growth_f)
cm2 <- cm_object2$table
cm2


# Calibration curve -----------------------------------------------------------
# how well do estimated vs actual event probabilities relate to each other?


create_calibration_plot(data_holdout, 
                        file_name = "figure-logit-m1-calibration", 
                        prob_var = "best_logit_no_loss_pred", 
                        actual_var = "comp_growth",
                        n_bins = 10)




#############################################x
# PART II.
# We have a loss function
########################################

# Introduce loss function
# relative cost of of a false negative classification (as compared with a false positive classification)
FP=1
FN=10
cost = FN/FP
# the prevalence, or the proportion of cases in the population (n.cases/(n.controls+n.cases))
prevelance = sum(data_train$comp_growth)/length(data_train$comp_growth)

# Draw ROC Curve and find optimal threshold with loss function --------------------------

best_tresholds <- list()
expected_loss <- list()
logit_cv_rocs <- list()
logit_cv_threshold <- list()
logit_cv_expected_loss <- list()

for (model_name in names(logit_models)) {
  
  model <- logit_models[[model_name]]
  colname <- paste0(model_name,"_prediction")
  
  best_tresholds_cv <- list()
  expected_loss_cv <- list()
  
  for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
    cv_fold <-
      model$pred %>%
      filter(Resample == fold)
    
    roc_obj <- roc(cv_fold$obs, cv_fold$fast)
    best_treshold <- coords(roc_obj, "best", ret="all", transpose = FALSE,
                            best.method="youden", best.weights=c(cost, prevelance))
    best_tresholds_cv[[fold]] <- best_treshold$threshold
    expected_loss_cv[[fold]] <- (best_treshold$fp*FP + best_treshold$fn*FN)/length(cv_fold$fast)
  }
  
  # average
  best_tresholds[[model_name]] <- mean(unlist(best_tresholds_cv))
  expected_loss[[model_name]] <- mean(unlist(expected_loss_cv))
  
  # for fold #5
  logit_cv_rocs[[model_name]] <- roc_obj
  logit_cv_threshold[[model_name]] <- best_treshold
  logit_cv_expected_loss[[model_name]] <- expected_loss_cv[[fold]]
  
}

logit_summary2 <- data.frame("Avg of optimal thresholds" = unlist(best_tresholds),
                             "Threshold for Fold5" = sapply(logit_cv_threshold, function(x) {x$threshold}),
                             "Avg expected loss" = unlist(expected_loss),
                             "Expected loss for Fold5" = unlist(logit_cv_expected_loss))

kable(x = logit_summary2, format = "latex", booktabs=TRUE,  digits = 3, row.names = TRUE,
      linesep = "", col.names = c("Avg of optimal thresholds","Threshold for fold #5",
                                  "Avg expected loss","Expected loss for fold #5")) %>%
  cat(.,file= paste0(data_out, "logit_summary1.tex"))

# Create plots based on Fold5 in CV ----------------------------------------------

for (model_name in names(logit_cv_rocs)) {
  
  r <- logit_cv_rocs[[model_name]]
  best_coords <- logit_cv_threshold[[model_name]]
  createLossPlot(r, best_coords,
                 paste0(model_name, "_loss_plot"))
  createRocPlotWithOptimal(r, best_coords,
                           paste0(model_name, "_roc_plot"))
}

# Pick best model based on average expected loss ----------------------------------

best_logit_with_loss <- logit_models[["X1"]]
best_logit_optimal_treshold <- best_tresholds[["X1"]]

logit_predicted_probabilities_holdout <- predict(best_logit_with_loss, newdata = data_holdout, type = "prob")
data_holdout[,"best_logit_with_loss_pred"] <- logit_predicted_probabilities_holdout[,"fast"]

# ROC curve on holdout
roc_obj_holdout <- roc(data_holdout$comp_growth, data_holdout[, "best_logit_with_loss_pred", drop=TRUE])

# Get expected loss on holdout
holdout_treshold <- coords(roc_obj_holdout, x = best_logit_optimal_treshold, input= "threshold",
                           ret="all", transpose = FALSE)
expected_loss_holdout <- (holdout_treshold$fp*FP + holdout_treshold$fn*FN)/length(data_holdout$comp_growth)
expected_loss_holdout

# Confusion table on holdout with optimal threshold
holdout_prediction <-
  ifelse(data_holdout$best_logit_with_loss_pred < best_logit_optimal_treshold, "slow", "fast") %>%
  factor(levels = c("slow", "fast"))
cm_object3 <- confusionMatrix(holdout_prediction,data_holdout$comp_growth_f)
cm3 <- cm_object3$table
cm3

#################################################
# PREDICTION WITH RANDOM FOREST
#################################################

# -----------------------------------------------
# RANDOM FOREST GRAPH EXAMPLE
# -----------------------------------------------
