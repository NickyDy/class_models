library(tidyverse)
library(tidymodels)
library(themis)
library(FFTrees)

df <- read_csv("https://raw.githubusercontent.com/kitkat0891/debt/main/all.csv") %>% janitor::clean_names()
df <- df %>% select(-c(in2y, x1, iso, year, country, region, sub_region, crisis)) %>% 
  mutate(in1y = factor(in1y)) %>% 
  mutate(in1y = fct_recode(in1y, "TRUE" = "1", "FALSE" = "0"))

df %>% count(reer_percent_dev)
glimpse(df)
df %>% map_dfr(~ sum(is.na(.))) %>% View()

recipe <- recipe(in1y ~., data = df) %>%
  step_impute_knn(all_predictors()) %>% 
  step_smote(in1y)
recipe

df_ready <- recipe %>% prep() %>% juice()
View(df_ready)

set.seed(2022)
split <- initial_split(df, strata = in1y)
train <- training(split)
test <- testing(split)

train <- train %>% mutate(in1y = as.logical(in1y))
test <- test %>% mutate(in1y = as.logical(in1y))

fit <- FFTrees(
  formula = in1y ~ .,
  data = train,
  data.test = test,
  decision.labels = c("0", "1"),
  do.comp = T)
fit

plot(fit, data = "train", main = "In 1y")

plot(fit, data = "test", main = "In 1y")

predict(fit, newdata = new_data, type = "both")

plot(fit, what = "cues", data = "train")

fit$competition$test %>%
  mutate_if(is.numeric, round, 3)
