library(tidyverse)
library(tidymodels)
library(themis)
library(rpart.plot)

df <- read_csv("stroke.csv", col_types = "icicccccddcc") %>% 
  janitor::clean_names() %>% 
  mutate(hypertension = fct_recode(hypertension, "No" = "0", "Yes" = "1"),
         heart_disease = fct_recode(heart_disease, "No" = "0", "Yes" = "1"),
         stroke = fct_recode(stroke, "No" = "0", "Yes" = "1")) %>% 
  select(-id) %>% 
  na.omit()

# df %>% count(stroke)
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
df_split <- initial_split(df, strata = purchase)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, strata = purchase)

# Recipe
dt_rec <- recipe(purchase ~ ., data = df_train) %>%
	step_downsample(purchase)
dt_rec
#train_prep <- model_rec %>% prep() %>% juice()
#glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(roc_auc, accuracy)

# Specify Decision Trees-----------------
dt_spec <- 
	decision_tree(
		mode = "classification",
		engine = "rpart",
		cost_complexity = tune(),
		tree_depth = tune(),
		min_n = tune())

# Workflow
dt_wf <-
	workflow() %>%
	add_recipe(dt_rec) %>% 
	add_model(dt_spec) 

# Grid
dt_grid <- grid_space_filling(
	cost_complexity(),
	tree_depth(),
	min_n(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
dt_tune <- dt_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = dt_grid)

# Select best metric
dt_best <- dt_tune %>% select_best(metric = "accuracy")
autoplot(dt_tune)
dt_best

# Train results
dt_train_results <- dt_tune %>%
	filter_parameters(parameters = dt_best) %>%
	collect_metrics()
dt_train_results

# Last fit
dt_test_results <- dt_wf %>% 
	finalize_workflow(dt_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

dt_results <- dt_test_results %>% collect_metrics()
dt_results%>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

#DT
collect_predictions(dt_test_results) %>%
	conf_mat(purchase, .pred_class) %>%
	pluck(1) %>%
	as_tibble() %>%
	ggplot(aes(Prediction, Truth, alpha = n)) +
	geom_tile(show.legend = FALSE) +
	geom_text(aes(label = n), colour = "white", alpha = 1, size = 8) +
	labs(y = "Actual result",	x = "Predicted result",	fill = NULL,
		title = "Confusion Matrix - DT") +
	theme(text = element_text(size = 16))

dt_test_results %>%
	collect_predictions() %>%
	roc_curve(purchase, .pred_No) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(linewidth = 0.5, color = "midnightblue") +
  geom_abline(lty = 2, color = "black", linewidth = 0.5) +
  theme(text = element_text(size = 16)) +
	labs(title = "DT - ROC curve", subtitle = "AUC = 0.731")

dt_test_results %>%  
	collect_predictions() %>% 
	ggplot() + geom_density(aes(x = .pred_Yes, fill = purchase), alpha = 0.5) +
	labs(title = "Density function from probabilities - DT")

dt_preds <- collect_predictions(dt_test_results) %>%
	bind_cols(df_test %>% select(-purchase))
dt_preds %>%
	select(prob = .pred_Yes, purchase, tv_total, you_tube_total, print_total,
	       pinterest, you_tube_mobile, flyers) %>%
	mutate(prob = prob * 100) %>%
	slice_max(prob, n = 10) %>%
	group_by(purchase) %>%
	arrange(-prob)

dt_test_results %>%
	extract_fit_engine() %>%
	rpart.plot(roundint = FALSE, cex = 0.9, type = 0)

library(vip)
dt_test_results %>%
	extract_fit_engine() %>%
	vi()

dt_test_results %>% 
  pluck(".workflow", 1) %>%
  extract_fit_parsnip() %>% 
  vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  geom_text(aes(label = round(Importance, 0)), hjust = -0.2) +
  theme(text = element_text(size = 16)) +
  labs(title = "Variable Importance - DT")

library(vetiver)
v <- dt_test_results %>%
	extract_workflow() %>%
	vetiver_model("purchase")
v
augment(v, slice_sample(df_test, n = 10))%>% 
  select(purchase, everything())

library(plumber)
pr() %>% 
	vetiver_api(v) %>% 
	pr_run()

pr() %>% 
	vetiver_api(v, type = "prob") %>% 
	pr_run()
