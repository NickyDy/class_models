library(tidyverse)
library(taylor)
glimpse(taylor_album_songs)

library(tidytext)

tidy_taylor <-
  taylor_album_songs |>
  unnest(lyrics) |> 
  unnest_tokens(word, lyric)

tidy_taylor

tidy_taylor |> 
  anti_join(get_stopwords()) |> 
  count(track_name, word, sort = TRUE)

lyrics_sparse <-
  tidy_taylor |> 
  count(track_name, word) |> 
  filter(n > 3) |> 
  cast_sparse(track_name, word, n)

dim(lyrics_sparse)

library(stm)
set.seed(123)
topic_model <- stm(lyrics_sparse, K = 13, verbose = FALSE)

summary(topic_model)

tidy(topic_model, matrix = "lift")

lyrics_gamma <- tidy(
  topic_model, 
  matrix = "gamma",
  document_names = rownames(lyrics_sparse))

lyrics_gamma |> 
  left_join(
    taylor_album_songs |> 
      select(album_name, document = track_name) |> 
      mutate(album_name = fct_inorder(album_name))
  ) |> 
  mutate(topic = factor(topic)) |> 
  ggplot(aes(gamma, topic, fill = topic)) +
  geom_boxplot(alpha = 0.7, show.legend = FALSE) +
  facet_wrap(vars(album_name)) +
  labs(x = expression(gamma))

set.seed(123)
effects <-
  estimateEffect(
    1:13 ~ album_name,
    topic_model,
    taylor_album_songs |> distinct(track_name, album_name) |> arrange(track_name))

tidy(effects) |> 
  filter(term != "(Intercept)", p.value < 0.05)

tidy(topic_model, matrix = "lift") |> 
  filter(topic == 13)