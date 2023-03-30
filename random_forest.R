library(tidyverse)
library(tidymodels)

df <- read_csv("../../Downloads/mult.csv") %>% 
	janitor::clean_names() %>% select(1:3, 6:8, 10:14) %>% 
	mutate(y = as.factor(y))

df %>% count(y)
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
df_split <- initial_split(df, strata = y)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, strata = y)

# Recipe
y_rec <- recipe(y ~., data = df_train) %>%
	step_corr(all_numeric_predictors()) %>% 
	step_normalize(all_numeric_predictors())
y_rec
train_prep <- y_rec %>% prep() %>% juice()
glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(roc_auc, accuracy)

# Specify RF--------------------------------------
rf_spec <- 
	rand_forest(mtry = tune(),
							min_n = tune(),
							trees = tune()) %>%
	set_engine("ranger") %>% 
	set_mode("classification")

# Workflow
rf_wflow <-
	workflow() %>%
	add_recipe(y_rec) %>% 
	add_model(rf_spec)

# Grid
rf_grid <- grid_latin_hypercube(
	finalize(mtry(), df_train),
	trees(),
	min_n(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
rf_res <-
	rf_wflow %>% 
	tune_grid(
		resamples = folds, 
		metrics = model_metrics,
		control = model_control,
		grid = rf_grid)

# Select best metric
rf_best <- rf_res %>% select_best(metric = "accuracy")
autoplot(rf_res)

rf_best

# Train results
rf_train_results <- rf_res %>% 
	filter_parameters(parameters = rf_best) %>% 
	collect_metrics()
rf_train_results

# Last fit
rf_test_res <- rf_wflow %>% 
	finalize_workflow(rf_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

# Test Results
rf_results%>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

#RF
collect_predictions(rf_test_res) %>%
	conf_mat(y, .pred_class) %>%
	pluck(1) %>%
	as_tibble() %>%
	ggplot(aes(Prediction, Truth, alpha = n)) +
	geom_tile(show.legend = FALSE) +
	geom_text(aes(label = n), colour = "white", alpha = 1, size = 8) +
	labs(
		y = "Actual result",
		x = "Predicted result",
		fill = NULL,
		title = "Confusion Matrix - Random Forest")

rf_test_res %>%
	collect_predictions() %>%
	roc_curve(y, .pred_1:.pred_4) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity, color = .level)) +
	geom_line(size = 1.5) +
	geom_abline(
		lty = 2, alpha = 0.5,
		color = "gray50",
		linewidth = 1.2) + 
	labs(title = "Random Forest - ROC curve", subtitle = "AUC = ")

rf_test_res %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_1, fill = y), alpha = 0.5) +
	labs(title = "Density function from probabilities - RF")

rf_preds <- collect_predictions(rf_test_res) %>%
	bind_cols(df_test %>% select(-y))
rf_preds %>%
	select(.pred_1, mv012, mv152, mv133, mv104) %>%
	rename(prob = .pred_1) %>%
	mutate(prob = prob *100) %>%
	slice_max(prob, n = 10) %>%
	group_by(prob) %>%
	arrange(desc(prob))

library(vip)
rf_test_res %>%
	extract_fit_engine() %>%
	vi()

rf_test_res %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% 
	vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
	labs(title = "Variable Importance - Multinomial Regression")

library(vetiver)
v <- rf_test_res %>%
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

