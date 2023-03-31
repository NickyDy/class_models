library(tidyverse)
library(tidymodels)
library(themis)
library(FFTrees)

df <- read_rds("~/Desktop/R/quatro/lecture_statistics/cimp.rds") %>% relocate(subtype, .after = rowname)
df <- df %>% pivot_longer(-c(rowname, subtype)) %>% 
  group_by(name) %>% 
  mutate(v = var(value)) %>% 
  arrange(desc(v)) %>% 
  ungroup() %>% 
  select(-v) %>% 
  pivot_wider(names_from = "name", values_from = "value") %>% 
  select(1:2002)

df %>% 
  mutate(subtype = fct_recode(subtype, "TRUE" = "CIMP", "FALSE" = "noCIMP")) %>% 
  select(subtype, 1:101) %>% select(-rowname) %>% slice(1:170)

new_data <- read_rds("~/Desktop/R/quatro/lecture_statistics/cimp.rds") %>% 
  mutate(subtype = fct_recode(subtype, "TRUE" = "CIMP", "FALSE" = "noCIMP")) %>% 
  select(subtype, 1:101) %>% select(-rowname) %>% slice(171:184)

df %>% map_dfr(~ sum(is.na(.)))

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

predict(fit, newdata = new_data, type = "both")

plot(fit, what = "cues", data = "train")

library(gt)
fit$competition$test %>%
  mutate_if(is.numeric, round, 3) %>% 
  gt()
