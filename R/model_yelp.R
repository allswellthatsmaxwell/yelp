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
library(rowr)

#' Read csv_basename from elsewhere-defined DATA_DIR
read_dat <- function(csv_basename) {
  readr::read_csv(glue("{DATA_DIR}/{csv_basename}"),
                  progress = FALSE)
}

## Elements stolen from prophet:::plot.prophet
copy_prophet_plot <- list(geom_ribbon(aes(ymin = yhat_lower,
                                          ymax = yhat_upper),
                                      alpha = 0.2,
                                      fill = RIBBON_COLOR,
                                      na.rm = TRUE),
                          geom_point(na.rm = TRUE),
                          geom_line(aes(y = yhat),
                                    color = RIBBON_COLOR,
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
                 labels = function(d) format(d, "%b %Y")) +
    theme(axis.text.x = element_text(angle = 90)) +
    copy_prophet_plot
}

#' Cast numeric date of the form yyyymmdd to a proper date.
yyyymmdd_to_date <- function(dates) {
  datenames <- names(dates)
  dates %<>% as.character() %>% as.Date(format = "%Y%m%d")
  names(dates) <- datenames
  dates
}

#' Canada day is on July 1 every year. Return a vector with those dates,
#' with every element named CanadaDay.
get_canada_day <- function(years) {
  canada_days <- paste0(years, "0701")
  names(canada_days) <- rep("CanadaDay", length(canada_days))
  canada_days
}

#' Get holidays.
#' Pulls back holidays quite aggressively; takes each day
#' tis::holidays considers a holiday, plus the day before
#' and the day after (named the same as the source holiday).
get_holidays <- function(years) {
  hol_vec <- holidays(businessOnly = FALSE,
                      board = TRUE,
                      goodFriday = TRUE,
                      inaug = TRUE,
                      years = years) %>%
    c(get_canada_day(years)) %>%
    yyyymmdd_to_date()
  one_day_before <- hol_vec - 1
  one_day_after <- hol_vec + 1
  names(one_day_before) <- names(hol_vec)
  names(one_day_after)  <- names(hol_vec)
  tibble(holiday = c(names(one_day_before),
                     names(hol_vec),
                     names(one_day_after)),
         ds = c(one_day_before,
                hol_vec,
                one_day_after))
}

#' Construct daily-level fits, and forecast horizon days,
#' for each state in state_day_frame.
#' @param state_day_frame a dataframe with the columns
#' state, ds (a date), and y (response variable; numeric)
#' @param holiday_frame a dataframe with the columns "holiday" and "ds",
#' respectively the name of the holiday and the dates it fell on.
#' If not passed, models without holidays.
#' @param horizon scalar int; number of out-of-sample days in the future to
#' be forecasted
model_var_by_state <- function(state_day_frame,
                               holiday_frame = NULL,
                               horizon = DAYS_IN_YEAR) {
  input_groups_frame <- state_day_frame %>%
    select(state, ds, y) %>%
    group_by(state)
  args <- list(df = input_groups_frame)
  if (!missing(holiday_frame))
    args <- append(args, list(holidays = holiday_frame))
  models <- do(input_groups_frame,
               model = do.call(prophet, args))
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

#' Orders input nicely and gives it proper names.
.prepare_outlier_holidays_for_print <- function(outlier_holiday_dat) {
  outlier_holiday_dat %>%
    group_by(holiday) %>%
    mutate(total_appearances = sum(appearances_of_day_in_state)) %>%
    arrange(desc(total_appearances),
            desc(appearances_of_day_in_state),
            state) %>%
    select(-total_appearances) %>%
    rename("State" = state, "Holiday" = holiday,
           "Outliers (all-time)" = appearances_of_day_in_state)
}

#' sends input date yyyy-mm-dd to 1900-mm-dd
send_date_to_fixed_year <- function(date) {
  as.Date(glue("1900-{format(date, '%m-%d')}"))
}

#' returns the input vector of years (yyyy), and additionally all the years
#' between the final year in the input and the current year
extend_years_to_current <- function(year_vec)
  c((min(year_vec) + 1):lubridate::year(Sys.Date()))

#' @param dat either a dataframe representing a single group, or
#' a grouped dataframe, that has the column y
#' @param winsize size (days) of the rolling window to use to calculate outliers
#' @param sd_factor how many standard deviations away from the windowed
#' mean must a point be to qualify as an outlier?
get_outlier_dates <- function(dat, sd_factor) {
  dat %>%
    mutate(yearly_avg = rowr::rollApply(y, fun = mean, window = DAYS_IN_YEAR,
                                        align = "right")) %>%
    mutate(y_wo_year_trend = y - yearly_avg) %>%
    mutate(y_wo_year_trend_avg = mean(y_wo_year_trend),
           y_wo_year_trend_sd  = sd  (y_wo_year_trend)) %>%
    mutate(top = y_wo_year_trend_avg + sd_factor * y_wo_year_trend_sd,
           bot = y_wo_year_trend_avg - sd_factor * y_wo_year_trend_sd,
           outlier = !between(y_wo_year_trend, bot, top))
}

#' Plots year-detrended reviews per day, by state, identifying
#' outliers and standard-deviation regions by color
make_outlier_plot <- function(outlier_dat) {
  outlier_dat %>%
    filter(!is.na(outlier)) %>%
    ggplot(aes(x = date, y = y_wo_year_trend)) +
    geom_point(aes(color = outlier, shape = outlier)) +
    geom_line(aes(y = y_wo_year_trend_avg), color = "white") +
    facet_wrap(~state, scales = "free_y") +
    theme_bw() +
    scale_color_manual(values = c("black", "red")) +
    x_date_scale +
    ylab("# reviews on day") +
    geom_ribbon(aes(x = date, ymin = bot, ymax = top),
                alpha = 0.2,
                fill = RIBBON_COLOR,
                 na.rm = TRUE)
}

#' Makes a plot of the Germany and Scotland outliers for the
#' years 2008-2010. This plot should be displayed very small.
make_european_outliers_plot <- function(outlier_dat) {
  outlier_dat %>%
    filter(outlier,
           state %in% EUROPEAN_STATES,
           between(year(date), 2008, 2010)) %>%
    group_by(state, date) %>%
    tally() %>%
    arrange(desc(n)) %>%
    ggplot(aes(x = date, y = state)) +
    geom_point() +
    theme_bw() +
    theme(axis.title.x = element_blank()) +
    scale_x_date(date_breaks = "3 months",
                 labels = function(d) format(d, "%b '%y"))
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
DAYS_IN_YEAR <- 365
RIBBON_COLOR <- "#0072B2"
EUROPEAN_STATES <- c("Baden-Wurttemberg", "Edinburg area (Scotland)")

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

## Holiday drops are pretty drastic in a few states; adding in holidays
## induces prophet to drop its forecast more aggresively on holidays
## corresponding to those historical dips.
hols <- state_review_values_by_date$date %>%
  lubridate::year() %>%
  unique() %>%
  extend_years_to_current() %>%
  get_holidays()

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

## Outlier investigation. ######################################################
SD_FACTOR <- 4

## Same number of rows as the input frame, but with a bunch of
## columns containing the year-detrended series, standard deviation,
## outlier-interval, and with days marked with whether they're outliers.
outlier_days <- state_review_values_by_date %>%
  mutate(y = reviews) %>%
  group_by(state) %>%
  get_outlier_dates(sd_factor = SD_FACTOR) %>%
  select(state, date,
         y_wo_year_trend, y_wo_year_trend_avg, y_wo_year_trend_sd,
         bot, top,
         outlier)

state_review_values_w_outliers <- outlier_days %>%
  inner_join(state_review_values_by_date, by = c("state", "date"))

## Holidays that coincided with outliers.
outlier_holidays <- state_review_values_w_outliers %>%
  filter(outlier,
         !(state %in% EUROPEAN_STATES)) %>%
  inner_join(hols, by = c("date" = "ds")) %>%
  group_by(state, holiday) %>%
  summarize(appearances_of_day_in_state = n())


.daily_review_counts_plot_w_outliers <- state_review_values_w_outliers %>%
  make_outlier_plot()
.european_outliers_plot <- state_review_values_w_outliers %>%
  make_european_outliers_plot()
.outlier_holidays <- outlier_holidays %>% .prepare_outlier_holidays_for_print()


## Modeling and forecasting. ###################################################
##

reviews_models <- state_review_values_by_date %>%
  additional_states_filter() %>%
  rename(ds = date, y = reviews) %>%
  model_var_by_state()

reviews_models_hols <- state_review_values_by_date %>%
  additional_states_filter() %>%
  rename(ds = date, y = reviews) %>%
  model_var_by_state(holiday_frame = hols)

## Great!
.reviews_facets <- plot_prophet_facets(reviews_models,
                                       ylab = "reviews posted on day")

.reviews_facets_hols <- plot_prophet_facets(reviews_models_hols,
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
