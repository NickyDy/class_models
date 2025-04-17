library(tidyverse)
library(tidymodels)
library(themis)

df <- read_rds("subtype.rds")

df %>% count(subtype)
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
log_rec <- recipe(subtype ~., data = df_train) %>%
	step_zv(all_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_downsample(subtype)
log_rec
#train_prep <- model_rec %>% prep() %>% juice()
#glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(roc_auc, accuracy)

# Specify LOG REG---------------------------------------------
log_spec <- 
	logistic_reg(mixture = 1, penalty = tune()) %>% 
	set_engine("glmnet") %>% 
	set_mode("classification")

# Workflow
log_wf <- workflow() %>%
	add_recipe(log_rec) %>%
	add_model(log_spec)

# Grid
log_grid <- grid_latin_hypercube(
	penalty(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
log_tune <- log_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = log_grid)

# Select best metric
log_best <- log_tune %>% select_best(metric = "accuracy")
autoplot(log_tune)

log_best

# Train results
log_train_results <- log_tune %>%
	filter_parameters(parameters = log_best) %>%
	collect_metrics()
log_train_results

# Last fit
log_test_results <- log_wf %>% 
	finalize_workflow(log_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

log_results <- log_test_results %>% collect_metrics()
log_results

# Logistic Regression Results
log_results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

log_test_results %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_Default, fill = defaulted), alpha = 0.5) +
	labs(title = "Density function from probabilities - Log Reg")

# Coefficients
log_test_results %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% tidy(exponentiate = TRUE, conf.level = 0.95) %>% arrange(-estimate)

log_preds <- collect_predictions(log_test_results) %>%
	bind_cols(df_test %>% select(-defaulted))
log_preds %>%
	select(.pred_Default, bank_balance, annual_salary) %>%
	rename(prob = .pred_Default) %>%
	mutate(prob = prob *100) %>%
	slice_max(prob, n = 10) %>%
	group_by(bank_balance) %>%
	arrange(desc(prob))

# Confusion matrix
collect_predictions(log_test_results) %>%
	conf_mat(defaulted, .pred_class) %>%
	pluck(1) %>%
	as_tibble() %>%
	ggplot(aes(Prediction, Truth, alpha = n)) +
	geom_tile(show.legend = FALSE) +
	geom_text(aes(label = n), colour = "white", alpha = 1, size = 8) +
	labs(
		y = "Actual result",
		x = "Predicted result",
		fill = NULL,
		title = "Confusion Matrix - Logistic regression") +
	theme(text = element_text(size = 16))

# ROC AUC
log_test_results %>%
	collect_predictions() %>%
	roc_curve(defaulted, `.pred_No Default`) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(size = 0.5, color = "midnightblue") +
  geom_abline(lty = 2, color = "black", linewidth = 0.5) +
  theme(text = element_text(size = 16)) +
	labs(title = "Logistic Regression - ROC curve", 
	     subtitle = paste0("AUC = ", round(log_results$.estimate[4], 3)))

library(vip)
log_test_results %>%
	extract_fit_engine() %>%
	vi()

log_test_results %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% 
	vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  geom_text(aes(label = round(Importance, 3)), hjust = -0.2) +
  theme(text = element_text(size = 16)) +
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
