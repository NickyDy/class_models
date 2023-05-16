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

set.seed(2019)
df_split <- initial_split(df, strata = subtype)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, strata = subtype)

# Recipe
mars_rec <- recipe(subtype ~., data = df_train) %>%
  step_zv(all_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  step_log(base = exp(1), all_predictors()) %>% 
  step_normalize(all_predictors())
mars_rec

mars_prep <- mars_rec %>% prep() %>% juice()
glimpse(mars_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(roc_auc, accuracy)

# Specify MARS-------------------------------------
mars_spec <- 
	mars(
		mode = "classification",
		engine = "earth",
		num_terms = tune(),
		prod_degree = tune(),
		prune_method = tune())

# Workflow
mars_wf <-
	workflow() %>%
	add_recipe(mars_rec) %>% 
	add_model(mars_spec)

# Grid
mars_grid <- grid_latin_hypercube(
	finalize(num_terms(), df_train),
	prod_degree(),
	prune_method(),
	size = 10)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
mars_tune <- mars_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = mars_grid)

# Select best metric
mars_best <- mars_tune %>% select_best(metric = "accuracy")
autoplot(mars_tune)

mars_best

# Train results
mars_train_results <- mars_tune %>%
	filter_parameters(parameters = mars_best) %>%
	collect_metrics()
mars_train_results

# Last fit
mars_test_results <- mars_wf %>% 
	finalize_workflow(mars_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

mars_results <- mars_test_results %>% collect_metrics()
mars_results

#MARS Results
mars_results%>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

#MARS
collect_predictions(mars_test_results) %>%
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
		title = "Confusion Matrix - MARS")

mars_test_results %>%
	collect_predictions() %>%
	roc_curve(churn, .pred_churn) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity)) +
	geom_line(size = 1.5, color = "midnightblue") +
	geom_abline(
		lty = 2, alpha = 0.5,
		color = "gray50",
		size = 1.2) + 
	labs(title = "MARS - ROC curve", subtitle = "AUC = ")

mars_test_results %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_churn, fill = churn), alpha = 0.5) +
	labs(title = "Density function from probabilities - MARS")

mars_test_results %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% 
	vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
	labs(title = "Variable Importance - MARS")

mars_preds <- collect_predictions(mars_test_results) %>%
	bind_cols(df_test %>% select(-churn))
mars_preds %>%
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
