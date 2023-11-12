library(tidyverse)
library(tidymodels)
library(discrim)
library(vip)
library(vetiver)
library(plumber)

df <- read_csv("data/diss.csv") %>% 
  select(veg_type, slope, s_ha, n2, invasive, herb_cover, h)

set.seed(123)
df_split <-
  df |> 
  mutate(veg_type = as.factor(veg_type)) |> 
  initial_split(strata = veg_type)

df_train <- training(df_split)
df_test <- testing(df_split)

set.seed(234)
df_folds <- vfold_cv(df_train, strata = veg_type)
df_folds

nb_spec <- naive_Bayes()
nb_spec_tune <- naive_Bayes(smoothness = tune())
mars_spec <- mars() |> 
  set_mode("classification")
mars_spec_tune <- mars(num_terms = tune()) |> 
  set_mode("classification")
rf_spec <- rand_forest(trees = 1e3) |> 
  set_mode("classification")
rf_spec_tune <- rand_forest(trees = 1e3, mtry = tune(), min_n = tune()) |> 
  set_mode("classification")

df_models <-
  workflow_set(
    preproc = list(formula = veg_type ~ .),
    models = list(
      nb = nb_spec, 
      mars = mars_spec, 
      rf = rf_spec,
      nb_tune = nb_spec_tune, 
      mars_tune = mars_spec_tune, 
      rf_tune = rf_spec_tune))

df_models

set.seed(123)
doParallel::registerDoParallel()

df_res <-
  df_models |> 
  workflow_map(
    "tune_grid",
    resamples = df_folds,
    metrics = metric_set(accuracy, sensitivity, specificity))

autoplot(df_res)

rank_results(df_res, rank_metric = "accuracy")

df_wf <- workflow(
  veg_type ~ ., 
  rf_spec |> set_engine("ranger", importance = "impurity"))
df_fit <- last_fit(df_wf, df_split)
df_fit

collect_predictions(df_fit) |> 
  conf_mat(veg_type, .pred_class)

collect_predictions(df_fit) |> 
  roc_curve(veg_type, .pred_f) |> 
  autoplot()

extract_workflow(df_fit) |>
  extract_fit_parsnip() |>
  vip()

v <- extract_workflow(df_fit) |> 
  vetiver_model("df-rf")
v

pr() |> 
  vetiver_api(v) |> 
  pr_run()

