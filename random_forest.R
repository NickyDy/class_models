library(tidyverse)
library(tidymodels)
library(themis)

df <- read_csv("data/diss.csv")

# df %>% count(y)
# skimr::skim(df)
# glimpse(df)
# diagnose_numeric(df) %>% flextable()
# plot_intro(df)
# plot_histogram(df)
# plot_correlate(df)
# df %>% plot_bar_category(top = 15)
# df %>% plot_bar(by  = "profit")
# df %>% select(profit, where(is.numeric)) %>% plot_boxplot(by = "profit")

#Modeling-------------------------------------------------
# Set seed for reproducibility
set.seed(2019)
df_split <- initial_split(df, prop = 0.8, strata = veg_type)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, v = 5, strata = veg_type)

# Recipe
rf_rec <- recipe(veg_type ~ ., data = df_train) %>%
  step_zv(all_numeric_predictors()) %>% 
  step_nzv(all_numeric_predictors()) %>% 
  step_corr(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(veg_type)
rf_rec

# train_prep <- rf_rec %>% prep() %>% juice()
# glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(roc_auc, accuracy)

# Specify RF--------------------------------------
rf_spec <- 
	rand_forest(mtry = tune(),
							min_n = tune(),
							trees = tune()) %>%
	set_engine("ranger", importance = "impurity") %>% 
	set_mode("classification")

# Workflow
rf_wflow <-
	workflow() %>%
	add_recipe(rf_rec) %>% 
	add_model(rf_spec)

# Grid
rf_grid <- grid_space_filling(
	finalize(mtry(), df_train),
	trees(),
	min_n(),
	size = 15)

# Tune
set.seed(1234)
rf_res <-
	rf_wflow %>% 
	tune_grid(
		resamples = folds, 
		metrics = model_metrics,
		control = model_control,
		grid = rf_grid)

# Select best metric
rf_best <- rf_res %>% 
  select_best(metric = "accuracy")
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

rf_results <- rf_test_res %>% collect_metrics()

rf_results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

#RF
collect_predictions(rf_test_res) %>%
	conf_mat(veg_type, .pred_class) %>%
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
	roc_curve(veg_type, .pred_f) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity)) +
	geom_line(linewidth = 0.5, color = "midnightblue") +
	geom_abline(lty = 2, alpha = 0.5,	color = "gray50",	linewidth = 0.5) + 
	labs(title = "Random Forest - ROC curve", 
	     subtitle = paste0("AUC = ", round(rf_results$.estimate[2], 3)))

rf_test_res %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_f, fill = veg_type), alpha = 0.5) +
	labs(title = "Density function from probabilities - RF")

rf_preds <- collect_predictions(rf_test_res) %>%
	bind_cols(df_test %>% select(-veg_type))
rf_preds %>%
	select(.pred_f, s_m2, herb_cover, e, elev) %>%
	rename(prob = .pred_f) %>%
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
	labs(title = "Variable Importance - Random Forest")

library(vetiver)
v <- rf_test_res %>%
	extract_workflow() %>%
	vetiver_model("veg_type")
v
augment(v, slice_sample(df_test, n = 10))

library(plumber)
pr() %>% 
	vetiver_api(v) %>% 
	pr_run()

pr() %>% 
	vetiver_api(v, type = "prob") %>% 
	pr_run()
