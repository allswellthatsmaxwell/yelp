library(prophet)
library(ggplot2)
library(dplyr)
library(magrittr)
library(purrr)
library(glue)
library(readr)
library(lubridate)

filter_biz <- function(dat, businesses_to_keep) {
  dplyr::inner_join(dat, businesses_to_keep, by = "business_id")
}

read_dat <- function(csv_basename) {
  readr::read_csv(glue("{DATA_DIR}/{csv_basename}"),
                  progress = FALSE)
}

## Elements stolen from prophet:::plot.prophet
copy_prophet_plot <- list(geom_ribbon(aes(ymin = yhat_lower,
                                          ymax = yhat_upper),
                                      alpha = 0.2,
                                      fill = "#0072B2",
                                      na.rm = TRUE),
                          geom_point(na.rm = TRUE),
                          geom_line(aes(y = yhat),
                                    color = "#0072B2",
                                    na.rm = TRUE),
                          theme(aspect.ratio = 3/5))

## Make the usual plot.prophet plot, but repeat in facets for every state.
plot_prophet_facets <- function(models, ylabel) {
  prophet_frames_by_state <-
    Map(function(state, model, fcast) {
          prophet:::df_for_plotting(model, fcast) %>% mutate(state = state)
        },
        models$state, models$model, models$forecast) %>%
    bind_rows()
  prophet_frames_by_state %>%
    ggplot(aes(x = as.Date(ds), y = y)) +
    facet_wrap(~state, scales = "free_y") +
    ##theme_bw() +
    ylab(ylabel) +
    xlab("date") +
    scale_x_date(date_breaks = "6 months",
                 labels = function(b) format(b, "%b %Y")) +
    theme(axis.text.x = element_text(angle = 90)) +
    copy_prophet_plot
}


DATA_DIR <- "../data"
MIN_REVIEWS_IN_2017 <- 300

businesses <- read_dat("yelp_business.csv")
businesses_to_keep <- businesses %>%
  select(business_id) %>%
  unique()

## checkins <- read_dat("yelp_checkin.csv") %>% filter_biz(businesses_to_keep)
reviews <- read_dat("yelp_review.csv") %>% filter_biz(businesses_to_keep)
## users <- read_dat("yelp_user.csv")

reviews %<>% select(-text)

businesses_and_reviews <- businesses %>%
  ## filter(review_count >= MIN_REVIEW_COUNT) %>%
  select(business_id, name, review_count, state, city) %>%
  inner_join(reviews, by = "business_id")

rm(reviews)
rm(businesses)

states_with_many_reviews <- businesses_and_reviews %>%
  group_by(state, year = year(date)) %>%
  summarize(reviews_in_year = n()) %>%
  filter(year == 2017, reviews_in_year >= MIN_REVIEWS_IN_2017)

state_review_values_by_date <- businesses_and_reviews %>%
  inner_join(states_with_many_reviews, by = "state") %>%
  group_by(state, date) %>%
  summarize(reviews = n(), mean_stars = mean(stars)) %>%
  group_by(state) %>%
  arrange(state, date)

daily_review_counts_plot <- state_review_values_by_date %>%
  ggplot(aes(x = date, y = reviews)) +
  geom_line() +
  facet_wrap(~state, scales = "free_y") +
  theme_bw()

models <- state_review_values_by_date %>%
  rename(ds = date, y = reviews) %>%
  select(-mean_stars) %>%
  group_by(state) %>%
  do(model = prophet(df = .))

models$future <- lapply(models$model, function(m) make_future_dataframe(m, 1000))
models$forecast <- Map(predict, models$model, models$future)

## The model is a little more hesitant about holiday dips than I'd like.
## I feel like the dips in AZ in particular can be more aggressive. Let's
## try adding in holidays to the model (can use just AZ for testing)
## to see if that ups the aggression.
prophet_facets <- plot_prophet_facets(models, ylab = "reviews posted on day")


daily_avg_stars_plot <- state_review_values_by_date %>%
  ggplot(aes(x = date, y = mean_stars)) +
  geom_line() +
  facet_wrap(~state) +
  theme_bw()







