
#' Read csv_basename from elsewhere-defined DATA_DIR
read_dat <- function(csv_basename) {
  readr::read_csv(glue("{DATA_DIR}/{csv_basename}"),
                  progress = FALSE)
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
                one_day_after)) %>%
    group_by(ds) %>%
    ## MLK and Inauguration can fall on same day; take it just once
    dplyr::slice(1) %>%
    ungroup()
}

#' Construct daily-level fits, and forecast horizon days,
#' for each group in day_frame (if it is grouped).
#' @param day_frame a dataframe with the columns
#' ds (a date), and y (response variable; numeric),
#' and possibly more grouping columns (preserved)
#' @param holiday_frame a dataframe with the columns "holiday" and "ds",
#' respectively the name of the holiday and the dates it fell on.
#' If not passed, models without holidays.
#' @param horizon scalar int; number of out-of-sample days in the future to
#' be forecasted
#' @param include_history argument to prophet::make_future_dataframe;
#' should in-sample data be forecasted?
model_ts <- function(day_frame,
                     holiday_frame = NULL,
                     horizon = HORIZON,
                     include_history = TRUE) {
  prophet_fn <-
    if (!missing(holiday_frame)) {
      purrr::partial(prophet, holidays = holiday_frame)
    } else {
      prophet
    }

  models <- do(day_frame, model = prophet_fn(df = .)) %>% ungroup()
  models <- day_frame %>% do(model = prophet_fn(df = .)) %>% ungroup()
  models$future <-
    lapply(models$model,
           function(m) make_future_dataframe(m, periods = horizon,
                                             include_history = include_history))
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
  businesses <- read_dat("yelp_business.csv")
  reviews <- read_dat("yelp_review.csv") %>% select(-text)
  businesses %>%
    select(business_id, name, review_count, state, city) %>%
    inner_join(reviews, by = "business_id")
}


#' sends input date yyyy-mm-dd to 1900-mm-dd
send_date_to_fixed_year <- function(date) {
  as.Date(glue("1900-{format(date, '%m-%d')}"))
}

#' returns the input vector of years (yyyy), and additionally all the years
#' between the final year in the input and the year of the target date
extend_years_to_target <- function(year_vec, target_date)
  c((min(year_vec) + 1):lubridate::year(target_date))

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

#' pulls state and forecast out of dat into a single data.frame
#' @param dat a rowwise_df with the columns state and forecast,
#' where state is a scalar character and forecast is a dataframe
#' @return a dataframe with all forecasts rowbinded, with state colbinded.
pull_out_forecast <- function(dat) {
  Map(function(state, fcast_dat) mutate(fcast_dat, state = state),
      dat$state,
      dat$forecast) %>%
    bind_rows() %>%
    mutate(ds = as.Date(ds))
}

#' return rows for those days in dat where ds is a member of hols$ds
take_holiday_days <- function(dat, hols) {
  dat %>% filter(as.character(ds) %in% as.character(hols$ds))
}

#' Filter, rename and group dat to prepare for per-state
#' prophet modeling.
prepare_for_state_modeling <- function(dat) {
  dat %>%
    additional_states_filter() %>%
    rename(ds = date, y = reviews) %>%
    select(state, ds,  y) %>%
    group_by(state)
}

#' Joins dat to the forecast data in models_frame by state and ds (day),
#' and marks it by adding the column group_name.
join_to_state_day_fcast <- function(dat, models_frame, group_name) {
  dat %>%
    inner_join(pull_out_forecast(models_frame), by = c("state", "ds")) %>%
    mutate(group = group_name)
}

#' Assigns the result of expr to name, and also saves the result
#' of expr to ../Robj/name.rds. If ../Robj/name.rds exists,
#' skips the computation of expr and just assigns the contents
#' of the rds to name instead.
`%<-%` <- function(name, expr) {
  name_text <- deparse(substitute(name))
  fname <- glue("../Robj/{name_text}.rds")
  result <- tryCatch(
    expr = {readRDS(fname)},
    error = function(e) {
      val <- eval(expr)
      saveRDS(val, fname)
      val
    })
  assign(name_text, result, envir = parent.frame())
}

#' mean absolute percent error of actuals y and predictions yhat
mape <- function(y, yhat) {
  n <- length(y)
  if (n != length(yhat)) stop("y and yhat are of different lengths")
  sum(abs((y - yhat) / y)) / n
}

#' ratio difference between actuals y and predictions yhat
ratio_diff <- function(y, yhat) (y - yhat) / y


#' Plots year-detrended reviews per day, by state, identifying
#' outliers and standard-deviation regions by color
.make_outlier_plot <- function(outlier_dat) {
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

#' Make the usual plot.prophet plot, but repeat in facets for every state.
.plot_prophet_facets <- function(models, ylabel) {
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
    scale_x_date(date_breaks = "1 year",
                 labels = function(d) format(d, "%b %Y"),
                 minor_breaks = NULL) +
    theme(axis.text.x = element_text(angle = 90)) +
    ## Elements stolen from prophet::plot.prophet
    geom_ribbon(aes(ymin = yhat_lower, ymax = yhat_upper),
                alpha = 0.2, fill = RIBBON_COLOR, na.rm = TRUE) +
    geom_point(na.rm = TRUE, alpha = 0.5, color = "gray") +
    geom_line(aes(y = yhat), color = RIBBON_COLOR, na.rm = TRUE)
}

#' returns a per-state plot with one line per level of stacked_forecast_make_outlier_plotframe,
#' and one point per date in holiday_frame, for the subset of rows with
#' year(stacked_input_frame$ds) equal to YEAR.
.make_model_comparison_plot <- function(stacked_forecast_frame, holiday_frame,
                                       year = YEAR) {
  fcast_year <- stacked_forecast_frame %>% filter(year(ds) == YEAR) %>%
    mutate(ds = as.Date(ds))
  holiday_data <- take_holiday_days(fcast_year, holiday_frame)
  fcast_year %>%
    ggplot(aes(x = ds, y = yhat)) +
    geom_line(aes(color = group), alpha = 0.8) +
    geom_point(data = holiday_data, aes(x = ds, y = yhat),
               alpha = 0.5, size = 1.6) +
    facet_wrap(~state, scales = "free_y") +
    theme_bw() +
    one_year_settings +
    labs(title = glue("{YEAR} in-sample fit: two different models"),
         y = paste("in-sample prediction for", DAILY_REVIEWS_YLAB),
         x = "Date")
}


#' Makes a plot of the Germany and Scotland outliers for the
#' years 2008-2010. This plot should be displayed very small.
.make_european_outliers_plot <- function(outlier_dat) {
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

#' return the number of rows in tallied_dat where n >= n_min
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
library(purrr)
library(tidyr)

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

one_year_settings <-
  list(theme(legend.title = element_blank()),
       scale_x_date(date_breaks = "1 month",
                    labels = function(d) format(d, "%b")))


#' Common states filter
additional_states_filter <- . %>% filter(TRUE)
## filter(state %in% c("Arizona", "Pennsylvania"))

DATA_DIR <- "../../data"
MIN_REVIEWS_IN_YEAR <- 300
YEAR <- 2017
DAYS_IN_YEAR <- 365
HORIZON <- DAYS_IN_YEAR
RIBBON_COLOR <- "#0072B2"
EUROPEAN_STATES <- c("Baden-Wurttemberg", "Edinburg area (Scotland)")
DAILY_REVIEWS_YLAB <- "reviews posted on day"
WO_HOLS_NAME <- "Without holidays"
WITH_HOLS_NAME <- "With holidays"

businesses_and_reviews %<-% {prepare_businesses_and_reviews() %>%
  rename(state_code = state)}

min_date <- min(businesses_and_reviews$date)
max_date <- max(businesses_and_reviews$date)
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
  extend_years_to_target(max_date + HORIZON) %>%
  get_holidays()

## Vanilla plots of daily values.
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
  .make_outlier_plot()
.european_outliers_plot <- state_review_values_w_outliers %>%
  make_european_outliers_plot()
.outlier_holidays <- outlier_holidays %>% .prepare_outlier_holidays_for_print()


## Modeling and forecasting. ###################################################
##


model_input <- state_review_values_by_date %>% prepare_for_state_modeling()

## Construct models without and with holidays marked.
## This takes forever, so if we've done this before,
## just read the tables from the stored rds.
reviews_models %<-% {model_input %>% model_ts()}
reviews_models_hols %<-% {model_input %>% model_ts(holiday_frame = hols)}

stacked_forecasts <-
  bind_rows(pull_out_forecast(reviews_models) %>%
              mutate(group = WO_HOLS_NAME),
            pull_out_forecast(reviews_models_hols) %>%
              mutate(group = WITH_HOLS_NAME))


## Great!
.reviews_facets <- reviews_models %>%
  .plot_prophet_facets(ylab = DAILY_REVIEWS_YLAB)

.reviews_facets_hols <- reviews_models_hols %>%
  .plot_prophet_facets(ylab = DAILY_REVIEWS_YLAB)

.holiday_vs_non_forecast_plot <- stacked_forecasts %>%
  .make_model_comparison_plot(hols, YEAR)

## Accuracy. ###################################################################

train_end <- as.Date("2016-06-30")
horizon <- as.integer(max_date - train_end)
## Construct the same models, but only up to train_end, then forecast
## HORIZON days.
reviews_models_trn %<-% {model_input %>%
                           filter(ds <= train_end) %>%
                           model_ts(horizon = horizon,
                                    include_history = FALSE)}

reviews_models_hols_trn %<-% {model_input %>%
                                filter(ds <= train_end) %>%
                                model_ts(horizon = horizon,
                                         holiday_frame = hols,
                                         include_history = FALSE)}

stacked_accuracies <-
  bind_rows(join_to_state_day_fcast(model_input, reviews_models_trn,
                                    WO_HOLS_NAME),
            join_to_state_day_fcast(model_input, reviews_models_hols_trn,
                                    WITH_HOLS_NAME))

.out_of_sample_year_plot <- stacked_accuracies %>%
  ggplot(aes(x = ds)) +
  geom_point(aes(y = y), color = "gray") +
  geom_line(aes(y = yhat, color = group)) +
  facet_wrap(~state, scales = "free_y") +
  theme_bw() +
  one_year_settings +
  labs(x = "Date", y = DAILY_REVIEWS_YLAB)

mapes <- stacked_accuracies %>%
  filter(group == WITH_HOLS_NAME) %>%
  group_by(state) %>%
  summarize(mape = mape(y, yhat))

year_errors <- stacked_accuracies %>%
  filter(group == WITH_HOLS_NAME) %>%
  group_by(state) %>%
  summarize(y_year = sum(y),
            yhat_year = sum(yhat),
            ratio_diff = ratio_diff(y_year, yhat_year)) %>%
  dplyr::select(state, ratio_diff)

.error_table <- inner_join(mapes, year_errors, by = "state") %>%
  mutate(mape = scales::percent(mape),
         `Sum-of-year % error` = scales::percent(ratio_diff)) %>%
  dplyr::select(-ratio_diff) %>%
  rename(State = state, "Avg daily % error" = mape)


## Stars. Incomplete. ##########################################################
##

stars_models <- state_review_values_by_date %>%
  additional_states_filter() %>%
  rename(ds = date, y = mean_stars) %>%
  model_var_by_state(hols)

.daily_avg_stars_plot <- state_review_values_by_date %>%
  ggplot(aes(y = mean_stars)) +
  facet_pt_state_date +
  x_date_scale

## So there might be some seasonal components to rating values.
## How interesting are the sharp downward spikes around Christmas (?) in PA?
## Seems a little overzealous.
## Next, let's pull out each component and look at them.
stars_facets <- plot_prophet_facets(stars_models,
                                    ylab = "Mean stars of ratings posted on day")
