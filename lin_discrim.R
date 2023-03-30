library(tidyverse)
library(tidymodels)
library(themis)
library(discrim)

df <- read_csv("mdata.csv") %>% janitor::clean_names()

df <- df %>% pivot_longer(2:34, names_to = "locality", values_to = "absorbance") %>% 
	mutate(locality = str_replace_all(locality, "^local_[:digit:]+", "local"),
				 locality = str_replace_all(locality, "^nonlocal_[:digit:]+", "nonlocal")) %>% 
	relocate(locality, .after = absorbance)

df %>% count(defaulted)
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
df_split <- initial_split(df, strata = locality)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, v = 5, strata = locality)

# Recipe
rec <- recipe(locality ~., data = df_train) %>%
	step_normalize(all_numeric_predictors()) %>%
	step_smote(locality)
rec
#train_prep <- model_rec %>% prep() %>% juice()
#glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(roc_auc, accuracy)

# Specify LOG REG---------------------------------------------
spec <- 
	discrim_linear(
		mode = "classification",
		penalty = tune(),
		regularization_method = "diagonal",
		engine = "mda")

# Workflow
wf <- workflow() %>%
	add_recipe(rec) %>%
	add_model(spec)

# Grid
grid <- grid_latin_hypercube(
	penalty(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
tune <- wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = grid)

# Select best metric
best <- tune %>% select_best(metric = "accuracy")
autoplot(tune)

best

# Train results
train_results <- tune %>%
	filter_parameters(parameters = best) %>%
	collect_metrics()
train_results

# Last fit
test_results <- wf %>% 
	finalize_workflow(best) %>%
	last_fit(split = df_split, metrics = model_metrics)

results <- test_results %>% collect_metrics()
results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

test_results %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_local, fill = locality), alpha = 0.5) +
	labs(title = "Density function from probabilities - Discriminant Regression")

# Coefficients
test_results %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% tidy(exponentiate = TRUE, conf.level = 0.95) %>% arrange(-estimate)

preds <- collect_predictions(test_results) %>%
	bind_cols(df_test %>% select(-defaulted))
preds %>%
	select(.pred_Default, bank_balance, annual_salary) %>%
	rename(prob = .pred_Default) %>%
	mutate(prob = prob *100) %>%
	slice_max(prob, n = 10) %>%
	group_by(bank_balance) %>%
	arrange(desc(prob))

# Confusion matrix
collect_predictions(test_results) %>%
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
test_results %>%
	collect_predictions() %>%
	roc_curve(defaulted, `.pred_No Default`) %>%
	ggplot(aes(x = 1 - specificity, y = sensitivity)) +
	geom_line(size = 0.5, color = "midnightblue") +
	geom_abline(
		lty = 2, alpha = 0.5,
		color = "gray50",
		size = 1) + 
	labs(title = "Logistic Regression - ROC curve", subtitle = "AUC = 0.954")

library(vip)
test_results %>%
	extract_fit_engine() %>%
	vi()

test_results %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% 
	vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
	labs(title = "Variable Importance - Logistic Regression")

library(vetiver)
v <- test_results %>%
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
