library(tidyverse)
library(tidymodels)
library(themis)

df <- read_csv("stroke.csv", col_types = "icicccccddcc") %>% 
  janitor::clean_names() %>% 
  mutate(hypertension = fct_recode(hypertension, "No" = "0", "Yes" = "1"),
         heart_disease = fct_recode(heart_disease, "No" = "0", "Yes" = "1"),
         stroke = fct_recode(stroke, "No" = "0", "Yes" = "1")) %>% 
  select(-id) %>% 
  na.omit()

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
df_split <- initial_split(df, strata = stroke)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, strata = stroke)

# Recipe
rf_rec <- recipe(stroke ~., data = df_train) %>% 
  step_downsample(stroke)
rf_rec

train_prep <- rf_rec %>% prep() %>% juice()
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
	add_recipe(rf_rec) %>% 
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

rf_results <- rf_test_res %>% collect_metrics()

rf_results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

#RF
collect_predictions(rf_test_res) %>%
	conf_mat(stroke, .pred_class) %>%
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
	roc_curve(stroke, .pred_No) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity)) +
	geom_line(linewidth = 0.5, color = "midnightblue") +
	geom_abline(lty = 2, alpha = 0.5,	color = "gray50",	linewidth = 0.5) + 
	labs(title = "Random Forest - ROC curve", subtitle = "AUC = 0.824")

rf_test_res %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_Yes, fill = stroke), alpha = 0.5) +
	labs(title = "Density function from probabilities - RF")

rf_preds <- collect_predictions(rf_test_res) %>%
	bind_cols(df_test %>% select(-stroke))
rf_preds %>%
	select(.pred_Yes, gender, age, hypertension, heart_disease) %>%
	rename(prob = .pred_Yes) %>%
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
augment(v, slice_sample(df_test, n = 10)) %>% select(12:14)

library(plumber)
pr() %>% 
	vetiver_api(v) %>% 
	pr_run()

pr() %>% 
	vetiver_api(v, type = "prob") %>% 
	pr_run()

