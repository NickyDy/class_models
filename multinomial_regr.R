library(tidyverse)
library(tidymodels)

df <- read_csv("../../Downloads/mult.csv") %>% 
	janitor::clean_names() %>% select(1:3, 6:8, 10:14) %>% 
	mutate(y = as.factor(y))

df %>% count(y)
glimpse(df)
diagnose_numeric(df) %>% flextable()
plot_intro(df)
plot_histogram(df)
plot_correlate(df)
df %>% plot_bar_category(top = 15)
df %>% plot_bar(by  = "profit")
df %>% select(profit, where(is.numeric)) %>% plot_boxplot(by = "profit")
df %>% map_dfr(~ sum(is.na(.))) %>% View()

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

# Specify Multinomial Regression--------------------------
mr_spec <- 
	multinom_reg(
		mode = "classification",
		engine = "nnet",
		penalty = tune(),
		mixture = 1)

# Workflow
mr_wf <-
	workflow() %>%
	add_recipe(y_rec) %>% 
	add_model(mr_spec) 

# Grid
mr_grid <- grid_latin_hypercube(
	penalty(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
mr_tune <- mr_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = mr_grid)

# Select best metric
mr_best <- mr_tune %>% select_best(metric = "accuracy")
autoplot(mr_tune)
mr_best

# Train results
mr_train_results <- mr_tune %>%
	filter_parameters(parameters = mr_best) %>%
	collect_metrics()
mr_train_results

# Last fit
mr_test_results <- mr_wf %>% 
	finalize_workflow(mr_best) %>%
	last_fit(split = df_split, metrics = model_metrics)
mr_results <- mr_test_results %>% collect_metrics()
mr_results

# Multinomial Regression Results
mr_results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

mr_test_results %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_class, fill = y), alpha = 0.5) +
	labs(title = "Density function from probabilities - Log Reg")

# Confusion matrix
collect_predictions(mr_test_results) %>%
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
		title = "Confusion Matrix - Multinomial regression") +
	theme(text = element_text(size = 16))

# ROC AUC
mr_test_results %>%
	collect_predictions() %>%
	roc_curve(y, .pred_1:.pred_4) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity, colour = .level)) +
	geom_line(linewidth = 0.5) +
	geom_abline(
		lty = 2, alpha = 0.5,
		color = "gray50",
		linewidth = 1) + 
	labs(title = "Multinomial Regression - ROC curve", subtitle = "AUC = 0.954")

library(vip)
mr_test_results %>%
	extract_fit_engine() %>%
	vi()

mr_test_results %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% 
	vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
	labs(title = "Variable Importance - Multinomial Regression")

library(vetiver)
v <- mr_test_results %>%
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
