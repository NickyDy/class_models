library(tidyverse)
library(tidymodels)
library(themis)
library(vip)
library(xgboost)
library(rpart.plot)

churn<-read_csv("csv/churn.csv") %>% janitor::clean_names()
df<-churn %>% slice_head(n = 1000)
val<-churn %>% slice_tail(n = 100) %>% select(-churn)

df %>% count(churn)
skimr::skim(df)
glimpse(df)
diagnose_numeric(df) %>% flextable()
plot_intro(df)
plot_histogram(df)
plot_correlate(df)
df %>% plot_bar_category(top = 15)
df %>% plot_bar(by  = "profit")
df %>% select(profit, where(is.numeric)) %>% plot_boxplot(by = "profit")

#Modeling-------------------------------------------------
# Set seed for reproducibility
set.seed(2019)
df_split <- initial_split(df, strata = churn)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, v = 5, strata = churn)

# Recipe
churn_rec <- recipe(churn ~., data = df_train) %>%
	step_normalize(all_numeric_predictors()) %>%
	step_dummy(all_nominal_predictors()) %>%
	step_smote(churn)
churn_rec
#train_prep <- model_rec %>% prep() %>% juice()
#glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(roc_auc, accuracy)

# Specify SVM-------------------------------------
svm_spec <- 
	svm_poly(
		mode = "classification",
		engine = "kernlab",
		cost = tune(),
		degree = tune(),
		scale_factor = tune())

# Workflow
svm_wf <-
	workflow() %>%
	add_recipe(model_rec) %>% 
	add_model(svm_spec)

# Grid
svm_grid <- grid_latin_hypercube(
	cost(),
	degree(),
	scale_factor(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
svm_tune <- svm_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = svm_grid)

# Select best metric
svm_best <- svm_tune %>% select_best(metric = "accuracy")
autoplot(svm_tune)

svm_best

# Train results
svm_train_results <- svm_tune %>%
	filter_parameters(parameters = svm_best) %>%
	collect_metrics()
svm_train_results

# Last fit
svm_test_results <- svm_wf %>% 
	finalize_workflow(svm_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

svm_results <- svm_test_results %>% collect_metrics()
svm_results

#----------
svm_fit <- svm_wf %>%
	finalize_workflow(svm_best) %>%
	fit(df_test)
svm_fit

pred_svm<-predict(svm_fit, new_data = val, type = "prob")
pred_svm

#SVM Results
svm_results%>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

#SVM
collect_predictions(svm_test_results) %>%
	conf_mat(churn, .pred_class) %>%
	pluck(1) %>%
	as_tibble() %>%
	ggplot(aes(Prediction, Truth, alpha = n)) +
	geom_tile(show.legend = FALSE) +
	geom_text(aes(label = n), colour = "white", alpha = 1, size = 8) +
	labs(
		y = "Actual result",
		x = "Predicted result",
		fill = NULL,
		title = "Confusion Matrix - SVM")

svm_test_results %>%
	collect_predictions() %>%
	roc_curve(churn, .pred_churn) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity)) +
	geom_line(size = 1.5, color = "midnightblue") +
	geom_abline(
		lty = 2, alpha = 0.5,
		color = "gray50",
		size = 1.2) + 
	labs(title = "SVM - ROC curve", subtitle = "AUC = ")

svm_test_results %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_churn, fill = churn), alpha = 0.5) +
	labs(title = "Density function from probabilities - SVM")

svm_preds <- collect_predictions(svm_test_results) %>%
	bind_cols(df_test %>% select(-churn))
svm_preds %>%
	select(.pred_churn, calls_outgoing_count, calls_outgoing_duration_max, 
				 last_100_calls_outgoing_duration, user_no_outgoing_activity_in_days, sms_outgoing_inactive_days) %>%
	rename(prob = .pred_churn) %>%
	mutate(prob = prob *100) %>%
	slice_max(prob, n = 10) %>%
	group_by(calls_outgoing_count) %>%
	arrange(desc(prob))
