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

df %>% count(stroke)
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
folds <- vfold_cv(df_train, v = 5, strata = stroke)

# Recipe
svm_rec <- recipe(stroke ~., data = df_train) %>%
	step_normalize(all_numeric_predictors()) %>%
	step_dummy(all_nominal_predictors()) %>%
	step_downsample(stroke)
svm_rec
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
	add_recipe(svm_rec) %>% 
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

svm_results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

#SVM
collect_predictions(svm_test_results) %>%
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
		title = "Confusion Matrix - SVM")

svm_test_results %>%
	collect_predictions() %>%
	roc_curve(stroke, .pred_Yes) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity)) +
	geom_line(linewidth = 1.5, color = "midnightblue") +
	geom_abline(lty = 2, alpha = 0.5,	color = "gray50",	linewidth = 1.2) + 
	labs(title = "SVM - ROC curve", subtitle = "AUC = ")

svm_test_results %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_stroke, fill = stroke), alpha = 0.5) +
	labs(title = "Density function from probabilities - SVM")

svm_preds <- collect_predictions(svm_test_results) %>%
	bind_cols(df_test %>% select(-stroke))
svm_preds %>%
	select(.pred_Yes, gender, age, hypertension, heart_disease) %>%
	rename(prob = .pred_Yes) %>%
	mutate(prob = prob *100) %>%
	slice_max(prob, n = 10) %>%
	group_by(calls_outgoing_count) %>%
	arrange(desc(prob))

library(vip)
dt_test_results %>%
  extract_fit_engine() %>%
  vi()

dt_test_results %>% 
  pluck(".workflow", 1) %>%
  extract_fit_parsnip() %>% 
  vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
  labs(title = "Variable Importance - Logistic Regression")

library(vetiver)
v <- dt_test_results %>%
  extract_workflow() %>%
  vetiver_model("stroke")
v
augment(v, slice_sample(df_test, n = 10))

library(plumber)
pr() %>% 
  vetiver_api(v) %>% 
  pr_run()

pr() %>% 
  vetiver_api(v, type = "prob") %>% 
  pr_run()
