
library(xgboost)
library(magrittr)

#' get n lags from point t (n <= t) in y
#' (i.e. each of y[t-1], y[t-2], ..., y[n+1], y[n])
get_lags <- function(y, t, n) {
  if (n > t) stop (glue("must have n <= t (got n = {n}, t = {t})"))
  lags <- sapply(1:n, function(i) y[t - i])
  matrix(lags, nrow = 1)
}

#' expands y into a (length(y) - n)-by-(n + 1) matrix,
#' where for j == 1, m[i,j] is y[i] and for j > 1, m[i, j] is
#' the jth lag of y[i].
#' @param y a timeseries
#' @param n a scalar int: number of lags to capture
get_n_lag_matrix <- function(y, n) {
  ylen <- length(y)
  lapply((n + 1):ylen,
         function(t) cbind(y[t], get_lags(y, t, n))) %>%
    do.call(rbind, .)
}

drop_first_n_rows <- function(mat, n) {
  mat[(n + 1):nrow(mat),]
}

#' make lag matrix from timeseries, with optional external regressors.
#' @param timeseries a vector, *already ordered by time (ascending)*
#' @param nlags number of lags to use as features
#' @param external_regressors a matrix of other features to cbind onto
#' the lag features. If null, the returned matrix only has the lag columns.
#' The returned matrix has length(timeseries) - nlags rows
#' and nlags + ncol(external_regressors) columns.
make_autoreg_matrices <- function(timeseries, nlags,
                                  external_regressors = NULL) {
  mat <- get_n_lag_matrix(timeseries, nlags)
  X <- mat[,-1]
  y <- mat[,1]
  if (!missing(external_regressors)) {
    stopifnot(length(timeseries) == nrow(external_regressors))
    X <- cbind(X, drop_first_n_rows(external_regressors, nlags))
  }
  list(X = X, y = y)
}

#' Predict using lags as predictors. Where known lags don't exist because
#' it is the future, predicted values are used instead.
#' This is accomplished by using the ith prediction as the
#' 1st lag on the (i + 1)th prediction step, the (i-1)th prediction
#' as the second lag, and so on.
#' @param model a model with a predict function that was
#' trained on nlags columns (one column per lag) (currently
#' maybe only xgboost works)
#' @param y the same y used to train the model
#' @param horizon the number of period to predict
#' @param nlags the number of lags used to train the model
walk_prediction <- function(model, y, horizon, nlags,
                            external_regressors = NULL) {
  ## The first prediction uses nlag true values from y;
  ## each subsequent prediction step throws one true value out
  ## from early in y and tacks the predicted value onto the
  ## end, and this happens horizon times. so we'll eventually be
  ## using nlags + horizon values from this vector (but only nlags
  ## at a time).
  using_external <- missing(external_regressors)
  if (using_external) stopifnot(horizon == nrow(external_regressors))
  short_y <- rep(as.numeric(NA), nlags + horizon)
  short_y[1:nlags] <- take_last_n(y, nlags)
  for (i in 1:horizon) {
    t_i <- nlags + i
    lag_predictors <- get_lags(short_y, t_i, nlags)
    X <-
      if (using_external) {
        cbind(lag_predictors, external_regressors[i, , drop = FALSE])
      } else {
        lag_predictors
      }
    short_y[t_i] <- predict(model, newdata = X)
  }
  take_last_n(short_y, horizon)
}


#' generate dates from test_start to horizon days after test_start
generate_test_ds <- function(test_start, horizon) {
  seq(from = test_start,
      to = (test_start + horizon - 1),
      by = 1)
}

#' get daily errors between true and predicted values
#' @param true_dat, pred_dat dataframes with the columns ds and y
#' @return a dataframe with one record per day and
#' the columns ds and y_minus_yhat
get_daily_errors <- function(true_dat, pred_dat) {
  inner_join(select(true_dat, ds, y),
             select(pred_dat, ds, y),
             by = "ds",
             suffix = c("", "hat")) %>%
    mutate(y_minus_yhat = y - yhat)
}


#' get the in-sample fit for y from model
get_fit <- function(model, y, nlags) {
  trn_lag_mat <- get_n_lag_matrix(y, nlags)
  trn_response <- trn_lag_mat[,1]
  trn_predictors <- trn_lag_mat[,-1]
  yfit <- predict(model, trn_predictors)
  tibble(ds = seq(from = train_start + nlags,
                  to = train_end,
                  by = 1),
         y = yfit)
}

#' return final n elements of vec
take_last_n <- function(vec, n) vec[(length(vec) - n + 1):length(vec)]

#' roll training and prediction periods over a long timespan to get
#' long-term idea of shorter-horizon accuracy
roll_validation <- function(dat, earliest_date, horizon) {

}

#' plot comparison between xgboost and prophet accuracy
.plot_xg_prophet_comparison <- function(trn, tst,
                                        xg_preds_frame,
                                        prophet_preds_frame,
                                        test_start,
                                        horizon) {
  bind_rows(trn,
            tst,
            xg_preds_frame,
            prophet_preds_frame) %>%
    filter(ds <= test_start + horizon - 1) %>%
    filter(ds > test_start - 0 * horizon) %>%
    ggplot(aes(x = ds, y = y, color = type)) +
    geom_line() +
    theme_bw() +
    test_set_x_scale +
    labs(x = "Date", y = DAILY_REVIEWS_YLAB) +
    theme(legend.title = element_blank(),
          aspect.ratio = 2/5)
}

#' plot in-sample fit
.plot_xg_fit <- function(trn, fit_frame) {
  fit_frame %>%
    mutate(type = XGLABEL) %>%
    bind_rows(trn) %>%
    ggplot(aes(x = ds, y = y, color = type)) +
    geom_point(alpha = 0.3, size = 4) +
    theme_bw() +
    scale_x_date(date_breaks = "1 year", labels = function(d) format(d, "%Y")) +
    theme(legend.title = element_blank(),
          legend.text = element_text(size = 30),
          aspect.ratio = 2/5) +
    labs(x = "Date", y = DAILY_REVIEWS_YLAB)
}

#' get day level comparisons between methods from two test set prediction
#' frames and mark by day which did better and by how much
get_error_comparison <- function(tst, xg_preds_frame, prophet_preds_frame) {
  xg_errors <- get_daily_errors(tst, xg_preds_frame) %>%
    mutate(type = XGLABEL)
  ph_errors <- get_daily_errors(tst, prophet_preds_frame) %>%
    mutate(type = PHLABEL)

  inner_join(xg_errors, ph_errors, by = "ds", suffix = c("_xg", "_ph")) %>%
    mutate(better = case_when(abs(y_minus_yhat_xg) < abs(y_minus_yhat_ph) ~
                                XGLABEL,
                              abs(y_minus_yhat_xg) > abs(y_minus_yhat_ph) ~
                                PHLABEL,
                              TRUE ~ "Same"),
           error_difference = abs(y_minus_yhat_ph) - abs(y_minus_yhat_xg),
           cumu_error_diff = cumsum(error_difference) / 1:n())
}

#' plot daily which method is better from a day-level dataframe of
#' error comparisons
.plot_better_by_day <- function(better_by_day_frame, nlags) {
  better_by_day_frame %>%
    ggplot(aes(x = ds)) +
    geom_point(aes(y = error_difference, color = better), size = 5, alpha = 0.9) +
    geom_line(aes(y = cumu_error_diff), size = 1.2) +
    geom_hline(yintercept = 0, color = "purple") +
    theme_bw() +
    labs(x = "Date",
         y = "Prophet error - xgboost error (in reviews per day)",
         title = glue("Difference in error by day between xgboost with {nlags} lags, and Prophet"),
         subtitle = glue("Days are colored by which model has the smaller absolute error.
The black line is the cumulative better-ness of xgboost.")) +
  theme(legend.title = element_text(size = 30),
        aspect.ratio = 2/5) +
  test_set_x_scale
}


#' Plot ds vs. y from dat
.plot_series <- function(dat) {
  dat %>%
    ggplot(aes(x = ds, y = y)) +
    geom_line() +
    theme(aspect.ratio = 2/5) +
    theme_bw()
}

#' build prophet model from trn and use it to forecast horizon days
#' forward from test_start
get_prophet_prediction <- function(trn, test_start, horizon) {
  proph <- prophet(trn)
  prophet_preds_frame <-
    predict(proph, tibble(ds = generate_test_ds(test_start, horizon))) %>%
    mutate(ds = as.Date(ds), type = PHLABEL) %>%
    rename(y = yhat)
    ##select(ds, y, type)
  prophet_preds_frame
}

#' Do the whole xgboost pipeline (model, predict on in-sample, predict on
#' out-of-sample horizon)
do_xg_steps <- function(trn, horizon, nlags,
                        external_regressors = NULL,
                        nrounds = 50) {
  inputs <- make_autoreg_matrices(trn$y, nlags, external_regressors)
  xg <- with(inputs, xgboost(X, y, nrounds = 50))
  xg_preds <- walk_prediction(xg, trn$y, horizon, nlags)
  xg_preds_frame <- tibble(ds = generate_test_ds(test_start, horizon),
                           y = xg_preds,
                           type = XGLABEL)
  xg_fit_frame <- get_fit(xg, trn$y, nlags)
  list(xg = xg,
       xg_preds = xg_preds,
       xg_preds_frame = xg_preds_frame,
       xg_fit_frame = xg_fit_frame)
}

#' xgboost modeling, prediction, and comparison with prophet.
#' @param full_dat a day-level (ds) dataframe with true y values (y)
#' and marked periods (period %in% c("train", "test"))
#' @param horizon number of dats after test_start to forecast
#' @param nlags number of lags to use as features
#' @return a list of named plots regarding test-set predictions and
#' model accuracy
model_predict_compare <- function(full_dat, test_start, horizon, nlags) {
  trn <- full_dat %>% filter(period == "train")
  tst <- full_dat %>% filter(period == "test")

  y_trn <- trn %>% arrange(ds) %$% y
  xg_list <- do_xg_steps(trn, horizon, nlags)

  ## Single-state prophet model to compare with
  prophet_preds_frame <- get_prophet_prediction(trn, test_start, horizon)

  ## By-day comparison of the two prediction sets.
  better_by_day <- get_error_comparison(tst,
                                        xg_list$xg_preds_frame,
                                        prophet_preds_frame)

  .better_by_day_plot <- .plot_better_by_day(better_by_day, nlags)
  .xg_fit_plot <- .plot_xg_fit(trn, xg_list$xg_fit_frame)
  .xg_prophet_comparison_plot <-
    .plot_xg_prophet_comparison(trn, tst,
                                xg_list$xg_preds_frame, prophet_preds_frame,
                                test_start, horizon)

  importance <- xgb.importance(model = xg_list$xg) %>%
    mutate(lag_number = as.integer(Feature) + 1L) %>%
    select(lag_number, Gain) %>%
    arrange(desc(Gain))

  list(xg_model = xg_list$xg,
       xg_preds_frame = xg_list$xg_preds_frame,
       xg_fit_frame = xg_list$xg_fit_frame,
       better_by_day = better_by_day,
       .better_by_day_plot = .better_by_day_plot,
       .xg_fit_plot = .xg_fit_plot,
       .xg_prophet_comparison_plot = .xg_prophet_comparison_plot,
       importance = importance)
}

#' return a matrix M where M[i, j] == 1 if the ith day is day j, else 0,
#' for dates between start and end (inclusive)
get_day_mat <- function(start, end) {
  day_vec <- weekdays(seq.Date(from = start, to = end, by = 1))
  table(1:length(day_vec), day_vec)
}


test_set_x_scale <- list(scale_x_date(date_breaks = "2 weeks",
                                      labels = function(d) format(d, "%d %b %Y")))

STATE <- "Arizona"
PHLABEL <- "prophet"
XGLABEL <- "xgboost"

#####
## Prepare single-state dataframe for modeling. ################################
#####

one_state_dat <- state_review_values_by_date %>%
  prepare_for_state_modeling() %>%
  filter(state == STATE) %>%
  ungroup() %>%
  select(-state)

train_start <- min(one_state_dat$ds)
train_end <- as.Date("2016-06-30")
test_start <- train_end + 1
test_end <- max(one_state_dat$ds)

ds_all <-
  tibble(ds = seq(from = train_start, to = test_end, by = 1),
         period = case_when(between(ds, train_start, train_end) ~ "train",
                            between(ds, test_start, test_end) ~ "test"))

one_state_dat_complete <- one_state_dat %>%
  right_join(ds_all, by = "ds") %>%
  ## The missing dates are all early and in the time of low activity,
  ## and it seems like Yelp has provided all non-zero days,
  ## so assume missing days are zeros.
  mutate(y = ifelse(is.na(y), 0, y)) %>%
  mutate(type = "actual")


######
## Forecasting with ensemble model. ############################################
######

one_state_dat_unmodified <- one_state_dat_complete

one_state_dat_stationary <- one_state_dat_complete %>% mutate(y = y - lag(y))

.stationary_plot <- one_state_dat_stationary %>%
  .plot_series() +
  labs(title = glue("{STATE} reviews-per-day made stationary"),
       x = "Date",
       y = DAILY_REVIEWS_YLAB)


horizon <- 365
NLAGS <- 365

series_result <- one_state_dat_unmodified %>%
  model_predict_compare(test_start, horizon, NLAGS)
stationary_series_result <- one_state_dat_stationary %>%
  model_predict_compare(test_start, horizon, NLAGS)

## Give this xgboost stuff a real run.
full_dat <- one_state_dat_unmodified
trn <- full_dat %>% filter(period == "train")
tst <- full_dat %>% filter(period == "test")

trn_day_mat <- get_day_mat(train_start, train_end)

inputs <- make_autoreg_matrices(trn$y, NLAGS, trn_day_mat)
xg <- with(inputs, xgboost(X, y, nrounds = 50))

tst_day_mat <- get_day_mat(test_start, test_start + horizon)
preds <- walk_prediction(xg, trn$y, horizon, NLAGS,
                         external_regressors = tst_day_mat)