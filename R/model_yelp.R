library(prophet)
library(ggplot2)
library(dplyr)
library(magrittr)
library(purrr)
library(glue)
library(readr)
library(lubridate)
library(tis)

#' Only keep the business IDs in businesses_to_use
filter_biz <- function(dat, businesses_to_use) {
  dplyr::inner_join(dat, businesses_to_use, by = "business_id")
}

#' Read csv_basename from elsewhere-defined DATA_DIR
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
                                    na.rm = TRUE))

#' Make the usual plot.prophet plot, but repeat in facets for every state.
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
    theme_bw() +
    ylab(ylabel) +
    xlab("date") +
    scale_x_date(date_breaks = "6 months",
                 labels = function(b) format(b, "%b %Y")) +
    theme(axis.text.x = element_text(angle = 90)) +
    copy_prophet_plot
}

#' Cast numeric date of the form yyyymmdd to a proper date.
yyyymmdd_to_date <- . %>% as.character() %>% as.Date(format = "%Y%m%d")

#' Get holidays.
get_holidays <- function(years) {
  hol_vec <- holidays(businessOnly = FALSE, board = TRUE, years = years)
  tibble(holiday = names(hol_vec), ds = yyyymmdd_to_date(hol_vec))
}

#' Construct daily-level fits, and forecast horizon days,
#' for each state in state_day_frame.
#' @param state_day_frame a dataframe with the columns
#' state, ds (a date), and y (response variable; numeric)
#' @param holiday_frame a dataframe with the columns "holiday" and "ds",
#' respectively the name of the holiday and the dates it fell on.
#' @param horizon scalar int; number of out-of-sample days in the future to
#' be forecasted
model_var_by_state <- function(state_day_frame,
                               holiday_frame,
                               horizon = 365 * 2) {
  models <- state_day_frame %>%
    select(state, ds, y) %>%
    group_by(state) %>%
    do(model = prophet(df = ., holidays = holiday_frame))
  models$future <- lapply(models$model,
                          function(m) make_future_dataframe(m, horizon))
  models$forecast <- Map(predict, models$model, models$future)
  models
}

#' Return only those states in dat that have at least min_reviews in year yyyy.
apply_reviews_in_year_criteria <- function(dat, min_reviews, yyyy = 2017) {
  dat %>%
    group_by(state, year = lubridate::year(date)) %>%
    summarize(reviews_in_year = n()) %>%
    filter(year == yyyy, reviews_in_year >= min_reviews) %>%
    select(state)
}

#' Common ggplot elements of simple raw timeseries plots.
facet_pt_state_date <- list(geom_point(aes(x = date)),
                            facet_wrap(~state, scales = "free_y"),
                            theme_bw())

#' Common states filter
additional_states_filter <- . %>% filter(TRUE) ##filter(state %in% c("AZ", "PA"))


DATA_DIR <- "../data"
MIN_REVIEWS_IN_2017 <- 300

businesses <- read_dat("yelp_business.csv")
businesses_to_use <- businesses %>% select(business_id) %>% unique() ## top_n

## checkins <- read_dat("yelp_checkin.csv") %>% filter_biz(businesses_to_use)
## users <- read_dat("yelp_user.csv")

reviews <- read_dat("yelp_review.csv") %>% filter_biz(businesses_to_use)

reviews %<>% select(-text)

businesses_and_reviews <- businesses %>%
  ## filter(review_count >= MIN_REVIEW_COUNT) %>%
  select(business_id, name, review_count, state, city) %>%
  inner_join(reviews, by = "business_id")

rm(reviews);rm(businesses);gc()

states_to_use <- businesses_and_reviews %>%
  apply_reviews_in_year_criteria(min_reviews = MIN_REVIEWS_IN_2017, yyyy = 2017)

state_review_values_by_date <- businesses_and_reviews %>%
  inner_join(states_to_use, by = "state") %>%
  group_by(state, date) %>%
  summarize(reviews = n(), mean_stars = mean(stars)) %>%
  arrange(state, date)

## Vanilla plots of daily values.
daily_avg_stars_plot <- state_review_values_by_date %>%
  ggplot(aes(y = mean_stars)) +
  facet_pt_state_date
daily_review_counts_plot <- state_review_values_by_date %>%
  ggplot(aes(y = reviews)) +
  facet_pt_state_date

## Modeling.
##
## Holiday drops are pretty drastic in a few states; adding in holidays
## induces prophet to drop its forecast more aggresively on holidays
## corresponding to those historical dips.
hols <- get_holidays(unique(year(state_review_values_by_date$date)))

reviews_models <- state_review_values_by_date %>%
  additional_states_filter() %>%
  rename(ds = date, y = reviews) %>%
  model_var_by_state(hols)

## Great!
reviews_facets <- plot_prophet_facets(reviews_models,
                                      ylab = "reviews posted on day")

stars_models <- state_review_values_by_date %>%
  additional_states_filter() %>%
  rename(ds = date, y = mean_stars) %>%
  model_var_by_state(hols)

## So there might be some seasonal components to rating values.
## How interesting are the sharp downward spikes around Christmas (?) in PA?
## Seems a little overzealous.
## Next, let's pull out each component and look at them.
stars_facets <- plot_prophet_facets(stars_models,
                                    ylab = "Mean stars of ratings posted on day")
