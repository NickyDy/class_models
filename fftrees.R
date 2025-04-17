library(tidyverse)
library(tidymodels)
library(themis)
library(FFTrees)

df <- read_csv("purchase.csv", col_types = "ififffffddff") %>% janitor::clean_names()
df <- df %>% 
  mutate(purchase = fct_recode(purchase, "TRUE" = "Yes", "FALSE" = "No"),
         gender = factor(gender)) %>% drop_na()

df %>% count(purchase)
glimpse(df)
df %>% map_dfr(~ sum(is.na(.))) %>% View()

recipe <- recipe(purchase ~., data = df) %>%
  step_downsample(purchase)
recipe

df_ready <- recipe %>% prep() %>% juice()
View(df_ready)

set.seed(2022)
split <- initial_split(df, strata = purchase)
train <- training(split)
test <- testing(split)

train <- train %>% mutate(purchase = as.logical(purchase))
test <- test %>% mutate(purchase = as.logical(purchase))

fit <- FFTrees(
  formula = purchase ~ .,
  data = train,
  data.test = test,
  decision.labels = c("FALSE", "TRUE"),
  do.comp = T)
fit

plot(fit, data = "train", main = "purchase")

plot(fit, data = "test", main = "purchase")

new_data <- sample(test, size = 10)
predict(fit, newdata = new_data, type = "both")

plot(fit, what = "cues", data = "train")

fit$competition$test %>%
  mutate_if(is.numeric, round, 3)
