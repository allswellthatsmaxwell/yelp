library(prophet)
library(ggplot2)
library(dplyr)
library(magrittr)
library(purrr)
library(glue)
library(readr)
library(lubridate)
library(tis)
library(feather)
library(assertr)

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
                               holiday_frame = NULL,
                               horizon = 365 * 2) {
  input_groups_frame <- state_day_frame %>%
    select(state, ds, y) %>%
    group_by(state)
  prophet_call <- purrr::partial(prophet, df = input_groups_frame)
  if (!missing(holiday_frame))
    prophet_call <- purrr::partial(prophet_call, holidays = holiday_frame)
  models <- input_groups_frame %>% do(model = prophet_call())
  models$future <- lapply(models$model,
                          function(m) make_future_dataframe(m, horizon))
  models$forecast <- Map(predict, models$model, models$future)
  models
}

get_businesses_and_reviews_fpath <- function() {
  glue("{DATA_DIR}/businesses_and_reviews.feather")
}

#' Reads dataframe from the path get_businesses_and_reviews_fpath() if
#' that file exists. If it does not exist, reads and joins business and review
#' data sets, and writes out a file to get_businesses_and_reviews_fpath().
#' @return A tibble with one row per review, with business information attached.
prepare_businesses_and_reviews <- function() {
  path <- get_businesses_and_reviews_fpath()
  if (file.exists(path)) {
    read_feather(path)
  } else {
    businesses <- read_dat("yelp_business.csv")
    reviews <- read_dat("yelp_review.csv") %>% select(-text)
    businesses_and_reviews <- businesses %>%
      select(business_id, name, review_count, state, city) %>%
      inner_join(reviews, by = "business_id")
    write_feather(businesses_and_reviews, path)
    businesses_and_reviews
  }
}

.count_n_above <- function(tallied_dat, n_min) {
  tallied_dat %>% filter(n >= n_min) %>% nrow()
}

## State names for states that make it through the MIN_REVIEWS_IN_YEAR filter.
## Where state codes weren't clear, codes were manually converted by
## looking at cities in the dataset.
state_codes_map <- c("AZ" = "Arizona",
                     "BW" = "Baden-Wurttemberg",
                     "EDH" = "Edinburg area (Scotland)",
                     "IL" = "Illinois",
                     "NC" = "North Carolina",
                     "NV" = "Nevada",
                     "OH" = "Ohio",
                     "ON" = "Ontario",
                     "PA" = "Pennsylvania",
                     "QC" = "Quebec",
                     "SC" = "South Carolina",
                     "WI" = "Wisconsin") %>%
  tibble(state_code = names(.), state = .)


x_date_scale <- list(scale_x_date(date_breaks = "1 year",
                                  labels = function(b) format(b, "%Y")))

#' Common ggplot elements of simple raw timeseries plots.
facet_pt_state_date <- list(geom_point(aes(x = date)),
                            facet_wrap(~state, scales = "free_y"),
                            theme_bw(),
                            xlab("Date"))

#' Common states filter
additional_states_filter <- . %>%
  filter(state %in% c("Arizona", "Pennsylvania")) ## filter(TRUE)

DATA_DIR <- "../data"
MIN_REVIEWS_IN_YEAR <- 300
YEAR <- 2017

businesses_and_reviews <- prepare_businesses_and_reviews() %>%
  rename(state_code = state)

.min_date <- min(businesses_and_reviews$date)
.max_date <- max(businesses_and_reviews$date)
.number_of_states <- length(unique(businesses_and_reviews$state_code))
.all_time_reviews_by_state <- businesses_and_reviews %>%
  group_by(state_code) %>%
  tally()

states_year_review_counts <- businesses_and_reviews %>%
  group_by(state_code, year = lubridate::year(date)) %>%
  summarize(reviews_in_year = n())

states_to_use_counts <- states_year_review_counts %>%
  filter(year == YEAR, reviews_in_year >= MIN_REVIEWS_IN_YEAR) %>%
  left_join(state_codes_map, by = "state_code") %>%
  ## We wrote state names down (in state_codes_map) once we knew the output
  ## of the above filter.
  ## If we failed to write a name down for any state_code, throw an error:
  assertr::verify(!is.na(state)) %>%
  ungroup()

.states_to_use_counts <- states_to_use_counts %>%
  select(state, reviews_in_year) %>%
  arrange(desc(reviews_in_year)) %>%
  mutate(reviews_in_year = scales::comma(reviews_in_year)) %>%
  set_colnames(c("State", glue("Reviews in {YEAR}")))

state_review_values_by_date <- businesses_and_reviews %>%
  inner_join(states_to_use_counts, by = "state_code") %>%
  ## Now that we've filtered state codes down to ones that definitely
  ## exist and we have names for, we stop using state_code and start using
  ## the more descriptive state column.
  group_by(state, date) %>%
  summarize(reviews = n(), mean_stars = mean(stars)) %>%
  arrange(state, date)

## Vanilla plots of daily values.
.daily_avg_stars_plot <- state_review_values_by_date %>%
  ggplot(aes(y = mean_stars)) +
  facet_pt_state_date +
  x_date_scale
.daily_review_counts_plot <- state_review_values_by_date %>%
  ggplot(aes(y = reviews)) +
  facet_pt_state_date +
  x_date_scale +
  ylab("# reviews on day")

## Modeling.
##
## Holiday drops are pretty drastic in a few states; adding in holidays
## induces prophet to drop its forecast more aggresively on holidays
## corresponding to those historical dips.
hols <- get_holidays(unique(year(state_review_values_by_date$date)))

reviews_models <- state_review_values_by_date %>%
  additional_states_filter() %>%
  rename(ds = date, y = reviews) %>%
  model_var_by_state()

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
