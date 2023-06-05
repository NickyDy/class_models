library(tidyverse)
library(tidymodels)
library(themis)
library(FFTrees)

df <- read_csv("stroke.csv", col_types = "ififffffddff") %>% janitor::clean_names()
df <- df %>% select(-c(id)) %>% 
  mutate(stroke = fct_recode(stroke, "TRUE" = "1", "FALSE" = "0")) %>% na.omit()

df %>% count(stroke)
glimpse(df)
df %>% map_dfr(~ sum(is.na(.))) %>% View()

recipe <- recipe(stroke ~., data = df) %>%
  step_smote(stroke)
recipe

df_ready <- recipe %>% prep() %>% juice()
View(df_ready)

set.seed(2022)
split <- initial_split(df, strata = stroke)
train <- training(split)
test <- testing(split)

train <- train %>% mutate(stroke = as.logical(stroke))
test <- test %>% mutate(stroke = as.logical(stroke))

fit <- FFTrees(
  formula = stroke ~ .,
  data = train,
  data.test = test,
  decision.labels = c("FALSE", "TRUE"),
  do.comp = T)
fit

plot(fit, data = "train", main = "Stroke")

plot(fit, data = "test", main = "Stroke")

new_data <- sample(test, size = 10)
predict(fit, newdata = new_data, type = "both")

plot(fit, what = "cues", data = "train")

fit$competition$test %>%
  mutate_if(is.numeric, round, 3)
