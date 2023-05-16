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

# MLP--------------------------------------------------
mlp_spec <- 
	mlp(
		mode = "classification",
		engine = "nnet",
		hidden_units = tune(),
		penalty = tune(),
		epochs = tune())

# Workflow
mlp_wf <-
	workflow() %>%
	add_recipe(model_rec) %>% 
	add_model(mlp_spec) 

# Grid
mlp_grid <- grid_latin_hypercube(
	hidden_units(),
	penalty(),
	epochs(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
mlp_tune <- mlp_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = mlp_grid)

# Select best metric
mlp_best <- mlp_tune %>% select_best(metric = "accuracy")
autoplot(mlp_tune)

mlp_best

# Train results
mlp_train_results <- mlp_tune %>%
	filter_parameters(parameters = mlp_best) %>%
	collect_metrics()
mlp_train_results

# Last fit
mlp_test_results <- mlp_wf %>% 
	finalize_workflow(mlp_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

mlp_results <- mlp_test_results %>% collect_metrics()
mlp_results
#---------
mlp_fit <- mlp_wf %>%
	finalize_workflow(mlp_best) %>%
	fit(df_test)
mlp_fit

pred_mlp<-predict(mlp_fit, new_data = val, type = "prob")
pred_mlp

#MLP Results
mlp_results%>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

#MLP
collect_predictions(mlp_test_results) %>%
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
		title = "Confusion Matrix - MLP")

mlp_test_results %>%
	collect_predictions() %>%
	roc_curve(churn, .pred_churn) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity)) +
	geom_line(size = 1.5, color = "midnightblue") +
	geom_abline(
		lty = 2, alpha = 0.5,
		color = "gray50",
		size = 1.2) + 
	labs(title = "MLP - ROC curve", subtitle = "AUC = ")

mlp_test_results %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_churn, fill = churn), alpha = 0.5) +
	labs(title = "Density function from probabilities - MLP")

mlp_test_results %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% 
	vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
	labs(title = "Variable Importance - MLP")

mlp_preds <- collect_predictions(mlp_test_results) %>%
	bind_cols(df_test %>% select(-churn))
mlp_preds %>%
	select(.pred_churn, calls_outgoing_count, calls_outgoing_duration_max, 
				 last_100_calls_outgoing_duration, user_no_outgoing_activity_in_days, sms_outgoing_inactive_days) %>%
	rename(prob = .pred_churn) %>%
	mutate(prob = prob *100) %>%
	slice_max(prob, n = 10) %>%
	group_by(calls_outgoing_count) %>%
	arrange(desc(prob))
