#🔢 1. Load Required Libraries

library(tidyverse)
library(tidymodels)
library(themis)

#Optional: Set a seed for reproducibility

set.seed(123)

#📦 2. Load and Explore the Data

data(iris)  # Example dataset
glimpse(df)
df <- read_csv("data/diss.csv") %>% select(2:27) %>% 
  mutate(across(is.character, as.factor))

#🧹 3. Split the Data

data_split <- initial_split(df, prop = 0.8, strata = veg_type)
train_data <- training(data_split)
test_data <- testing(data_split)

#🔧 4. Create a Recipe for Preprocessing

recipe <- recipe(veg_type ~ ., data = train_data) %>%
  step_zv(all_numeric_predictors()) %>% 
  step_nzv(all_numeric_predictors()) %>% 
  step_corr(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(veg_type)

#🧠 5. Specify a Model

rf_model <- rand_forest(mtry = 5, trees = 2000, min_n = 2) %>%
  set_engine("ranger") %>%
  set_mode("classification")

#🔗 6. Create a Workflow

workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe)

#🚂 7. Fit the Model

rf_fit <- fit(workflow, data = train_data)

#📊 8. Evaluate the Model

rf_preds <- predict(rf_fit, test_data) %>%
  bind_cols(test_data)

metrics(rf_preds, truth = veg_type, estimate = .pred_class)
conf_mat(rf_preds, truth = veg_type, estimate = .pred_class)
#----------------------------------------------------------
#🎯 9. Tune Hyperparameters (Optional but Recommended)

# Create cross-validation folds
folds <- vfold_cv(train_data, v = 5, strata = veg_type)

# Update model to allow tuning
rf_tune_model <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# New workflow
rf_tune_workflow <- workflow() %>%
  add_model(rf_tune_model) %>%
  add_recipe(iris_recipe)

# Define grid
rf_grid <- grid_regular(mtry(range = c(1, 4)),
                        min_n(range = c(2, 10)),
                        levels = 4)

# Tune
tune_results <- tune_grid(
  rf_tune_workflow,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(accuracy)
)

# Show best parameters
show_best(tune_results, "accuracy")

#✅ 10. Finalize the Workflow and Refit on Full Training Data

best_params <- select_best(tune_results, "accuracy")

final_rf <- finalize_workflow(rf_tune_workflow, best_params)
final_fit <- fit(final_rf, data = train_data)

#🧪 11. Test Final Model on Holdout Set

final_predictions <- predict(final_fit, test_data) %>%
  bind_cols(test_data)

metrics(final_predictions, truth = veg_type, estimate = .pred_class)