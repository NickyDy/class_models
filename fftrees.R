library(tidyverse)
library(tidymodels)
library(themis)
library(FFTrees)

df <- read_rds("cimp.rds") %>% 
  mutate(subtype = fct_recode(subtype, "TRUE" = "CIMP", "FALSE" = "noCIMP")) %>% 
  select(subtype, 1:101) %>% select(-rowname)

df %>% count(overall_survival)
df_ready %>% map_dfr(~ sum(is.na(.)))

recipe <- recipe(overall_survival ~., data = df) %>%
  step_zv(all_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_downsample(overall_survival)
recipe

df_ready <- recipe %>% prep() %>% juice()
glimpse(df_ready)

set.seed(2022)
split <- initial_split(df, strata = subtype)
train <- training(split)
test <- testing(split)

train_last <- train %>% mutate(subtype = as.logical(subtype))
test_last <- test %>% mutate(subtype = as.logical(subtype))

fit <- FFTrees(
  formula = subtype ~ .,
  data = train_last,
  data.test = test_last,
  decision.labels = c("No CIMP", "CIMP"),
  do.comp = T)
fit

plot(fit, data = "train", main = "Risk of CIMP")

plot(fit, data = "test", main = "Risk of CIMP")

#predict(fit, newdata = val)

plot(fit, what = "cues", data = "train")

library(gt)
fit$competition$test %>%
  mutate_if(is.numeric, round, 3) %>% 
  gt()
