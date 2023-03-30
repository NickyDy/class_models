library(tidyverse)
library(tidymodels)
library(themis)

df <- read_rds("subtype.rds")

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
df_split <- initial_split(df, strata = subtype)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, strata = subtype)

# Recipe
knn_rec <- recipe(subtype ~., data = df_train) %>%
  step_zv(all_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  step_log(base = exp(1), all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_smote(subtype)
knn_rec

#train_prep <- model_rec %>% prep() %>% juice()
#glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(roc_auc, accuracy)

# Specify KNN-------------------------------------
knn_spec <- 
	nearest_neighbor(
		mode = "classification",
		engine = "kknn",
		neighbors = tune(),
		weight_func = tune(),
		dist_power = tune())

# Workflow
knn_wf <-
	workflow() %>%
	add_recipe(knn_rec) %>% 
	add_model(knn_spec)

# Grid
knn_grid <- grid_latin_hypercube(
	neighbors(),
	dist_power(),
	weight_func(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
knn_tune <- knn_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = knn_grid)

# Select best metric
knn_best <- knn_tune %>% select_best(metric = "accuracy")
autoplot(knn_tune)

knn_best

# Train results
knn_train_results <- knn_tune %>%
	filter_parameters(parameters = knn_best) %>%
	collect_metrics()
knn_train_results

# Last fit
knn_test_results <- knn_wf %>% 
	finalize_workflow(knn_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

knn_results <- knn_test_results %>% collect_metrics()
knn_results
#----------
knn_fit <- knn_wf %>%
	finalize_workflow(knn_best) %>%
	fit(df_test)
knn_fit

pred_knn<-predict(knn_fit, new_data = val, type = "prob")
pred_knn

#KNN Results
knn_results%>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

#KNN
collect_predictions(knn_test_results) %>%
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
		title = "Confusion Matrix - KNN")

knn_test_results %>%
	collect_predictions() %>%
	roc_curve(churn, .pred_churn) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity)) +
	geom_line(size = 1.5, color = "midnightblue") +
	geom_abline(
		lty = 2, alpha = 0.5,
		color = "gray50",
		size = 1.2) + 
	labs(title = "KNN - ROC curve", subtitle = "AUC = ")

knn_test_results %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_churn, fill = churn), alpha = 0.5) +
	labs(title = "Density function from probabilities - KNN")

knn_preds <- collect_predictions(knn_test_results) %>%
	bind_cols(df_test %>% select(-churn))
knn_preds %>%
	select(.pred_churn, calls_outgoing_count, calls_outgoing_duration_max, 
				 last_100_calls_outgoing_duration, user_no_outgoing_activity_in_days, sms_outgoing_inactive_days) %>%
	rename(prob = .pred_churn) %>%
	mutate(prob = prob *100) %>%
	slice_max(prob, n = 10) %>%
	group_by(calls_outgoing_count) %>%
	arrange(desc(prob))

library(vip)
log_test_results %>%
  extract_fit_engine() %>%
  vi()

log_test_results %>% 
  pluck(".workflow", 1) %>%
  extract_fit_parsnip() %>% 
  vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
  labs(title = "Variable Importance - Logistic Regression")

library(vetiver)
v <- log_test_results %>%
  extract_workflow() %>%
  vetiver_model("defaulted")
v
augment(v, slice_sample(df_test, n = 10))

library(plumber)
pr() %>% 
  vetiver_api(v) %>% 
  pr_run()

pr() %>% 
  vetiver_api(v, type = "prob") %>% 
  pr_run()