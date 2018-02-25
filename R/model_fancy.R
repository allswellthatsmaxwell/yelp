
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


#' get the in-sample fit for y from model
get_fit <- function(model, y, nlags, external_regressors = NULL) {
  X <- make_autoreg_matrices(y, nlags, external_regressors) %$% X
  yfit <- predict(model, X)
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

#' plot out-of-sample day-level predictions from
#' different models (marked by the column "type")
#' in preds_frame
.plot_preds_frame <- function(trn,
                              tst,
                              preds_frame,
                              test_start,
                              horizon) {
  bind_rows(trn,
            tst,
            preds_frame) %>%
    filter(ds <= test_start + horizon - 1) %>%
    filter(ds > test_start - 0 * horizon) %>%
    ggplot(aes(x = ds, y = y, color = type)) +
    geom_line() +
    theme_bw() +
    test_set_x_scale +
    labs(x = "Date", y = DAILY_REVIEWS_YLAB) +
    theme(legend.title = element_blank())##, aspect.ratio = 2/5)
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

#' get daily errors between true and predicted values
#' @param true_dat, pred_dat dataframes with the columns ds and y
#' @return a dataframe with one record per day and
#' the columns ds and y_minus_yhat
get_daily_errors <- function(true_dat, pred_dat) {
  inner_join(select(true_dat, state, ds, y),
             select(pred_dat, state, ds, y),
             by = c("ds", "state"),
             suffix = c("", "hat")) %>%
    mutate(y_minus_yhat = y - yhat)
}

#' returns the only unique element in vec, or throws an error
#' if vec contains multiple unique values
only <- function(vec) {
  el <- unique(vec)
  stopifnot(length(el) == 1)
  el
}

p_ <- function(s) paste0("_", s)

#' get day level comparisons between methods from two test set prediction
#' frames and mark by day which did better and by how much
#' @param preds_frame_1, preds_frame_2 dataframe with the columns ds, y,
#' and type
get_error_comparison <- function(tst, preds_frame_1, preds_frame_2) {
  e1 <- get_daily_errors(tst, preds_frame_1)
  e2 <- get_daily_errors(tst, preds_frame_2)
  type1 <- only(preds_frame_1$type)
  type2 <- only(preds_frame_2$type)
  suffix1 <- p_(type1)
  suffix2 <- p_(type2)
  name1 <- paste0("y_minus_yhat", suffix1)
  name2 <- paste0("y_minus_yhat", suffix2)

  compare_frame <- inner_join(e1, e2,
                              by = c("ds", "state"),
                              suffix = c(suffix1, suffix2)) %>%
    mutate(type1 = type1, type2 = type2)
  wrapr::let(list(E1 = name1, E2 = name2),
             compare_frame %>%
               mutate(better = case_when(abs(E1) < abs(E2) ~ type1,
                                         abs(E1) > abs(E2) ~ type2,
                                         TRUE ~ "Same"),
                      error_difference = abs(E1) - abs(E2),
                      cumu_error_diff = cumsum(error_difference) / 1:n()))
}

#' plot daily which method is better from a day-level dataframe of
#' error comparisons
.plot_better_by_day <- function(better_by_day_frame, pt_size = 5) {
  t1 <- only(better_by_day_frame$type1)
  t2 <- only(better_by_day_frame$type2)
  better_by_day_frame %>%
    ggplot(aes(x = ds)) +
    geom_point(aes(y = error_difference, color = better), size = pt_size, alpha = 0.9) +
    geom_line(aes(y = cumu_error_diff), size = 1.2) +
    geom_hline(yintercept = 0, color = "purple") +
    theme_bw() +
    labs(x = "Date",
         y = "{t2} error - {t1} error",
         title = glue("Difference in error by day between {t1} and {t2}."),
         subtitle = glue("Days are colored by which model has the smaller absolute error.
The black line is the cumulative better-ness of {t1}.")) +
  theme(legend.title = element_text(size = 30)) +
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


get_seasonal_naive <- function(trn, test_start, horizon, season_len = 365) {
  stopifnot(horizon <= season_len)
  ds <- generate_test_ds(test_start, horizon)
  pred <- take_last_n(trn$y, season_len)[1:horizon]
  tibble(y = pred, type = glue("seasonal_naive_{season_len}"), ds = ds)
}

#' do default auto.arima fit without external regressors
do_arima <- function(trn, test_start, horizon) {
  model <- forecast::auto.arima(trn$y)
  fcast <- forecast::forecast(model, h = horizon)
  preds_frame <- tibble(y = fcast$mean, type = "arima",
                        ds = generate_test_ds(test_start, horizon))
  list(model = model, forecast_obj = fcast, preds_frame = preds_frame)
}

#' Do the whole xgboost pipeline (model, predict on in-sample, predict on
#' out-of-sample horizon)
do_xg_steps <- function(trn, horizon, nlags,
                        trn_external_regressors = NULL,
                        tst_external_regressors = NULL,
                        nrounds = 50) {
  inputs <- make_autoreg_matrices(trn$y, nlags, trn_external_regressors)
  xg <- with(inputs, xgboost(X, y, nrounds = nrounds))
  xg_preds <- walk_prediction(xg, trn$y, horizon, nlags, tst_external_regressors)
  xg_preds_frame <- tibble(ds = generate_test_ds(test_start, horizon),
                           y = xg_preds,
                           type = XGLABEL)
  xg_fit_frame <- get_fit(xg, trn$y, nlags, trn_external_regressors)
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

  .better_by_day_plot <- .plot_better_by_day(better_by_day, pt_size = 5) +
    theme(aspect.ratio = 2/5)
  .xg_fit_plot <- .plot_xg_fit(trn, xg_list$xg_fit_frame)
  .xg_prophet_comparison_plot <-
    .plot_preds_frame(trn, tst,
                      bind_rows(xg_list$xg_preds_frame,
                                prophet_preds_frame),
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

cross_join <- function(d1, d2) {
  inner_join(d1 %>% mutate(abcdefg = TRUE),
             d2 %>% mutate(abcdefg = TRUE),
             by = "abcdefg") %>%
    select(-abcdefg)
}

#' for each dataframe in list_of_df, add the column "state", holding
#' the corresponding name (i.e. names(list_of_df)[[i]] for list_of_df[[i]]).
#' Then rowbinds all these together.
add_state_combine <- function(list_of_df) {
  Map(function(dat, state) mutate(dat, state = state),
      list_of_df, names(list_of_df)) %>%
  bind_rows()
}

test_set_x_scale <- list(scale_x_date(date_breaks = "2 weeks",
                                      labels = function(d) format(d, "%d %b %Y")))

corr_plot_elements <- list(geom_point(),
                           theme_bw(),
                           facet_wrap(~state, scales = "free"),
                           geom_vline(xintercept = 0, color = "purple"),
                           geom_hline(yintercept = 0, color = "purple"))

STATE <- "Arizona"
PHLABEL <- "prophet"
XGLABEL <- "xgboost"

#####
## Prepare single-state dataframe for modeling. ################################
#####

train_start <- min(model_input$ds)
train_end <- as.Date("2016-06-30")
test_start <- train_end + 1
test_end <- max(model_input$ds)

ds_all <-
  tibble(ds = seq(from = train_start, to = test_end, by = 1),
         period = case_when(between(ds, train_start, train_end) ~ "train",
                            between(ds, test_start, test_end) ~ "test"))
states_all <- tibble(state = unique(state_review_values_by_date$state))
ds_states_all <- cross_join(ds_all, states_all)


states_days_complete <- state_review_values_by_date %>%
  prepare_for_state_modeling() %>%
  right_join(ds_states_all, by = c("state", "ds")) %>%
  ## The missing dates are all early and in the time of low activity,
  ## and it seems like Yelp has provided all non-zero days,
  ## so assume missing days are zeros.
  mutate(y = ifelse(is.na(y), 0, y)) %>%
  mutate(type = "actual")

one_state_dat_complete <- states_days_complete %>% filter(state == STATE)



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
NLAGS <- 1000

series_result <- one_state_dat_unmodified %>%
  model_predict_compare(test_start, horizon, NLAGS)
stationary_series_result <- one_state_dat_stationary %>%
  model_predict_compare(test_start, horizon, NLAGS)

## Give this xgboost stuff a real run. Various external regressors,
## comparisons to various other techniques, for all 12 states.
full_dat <- states_days_complete
trn <- full_dat %>% filter(period == "train")
tst <- full_dat %>% filter(period == "test")

trn_day_mat <- get_day_mat(train_start, train_end)
tst_day_mat <- get_day_mat(test_start, test_start + horizon)

hols_frame <- get_holidays(year(train_start):year(test_end)) %>%
  right_join(ds_all, by = "ds") %>%
  mutate(holiday = if_else(is.na(holiday), "None", holiday)) %>%
  filter(between(ds, train_start, test_end))

trn_hols_mat <- hols_frame %>% filter(period == "train") %>%
  {table(1:length(.$holiday), .$holiday)}
tst_hols_mat <- hols_frame %>% filter(period == "test") %>%
  {table(1:length(.$holiday), .$holiday)}

trns <- trn %>% split(.[["state"]])

xg_lists <- trns %>%
  lapply(. %>% do_xg_steps(horizon,
                           NLAGS,
                           trn_hols_mat,
                           tst_hols_mat,
                           nrounds = 50))
pr_lists <- trns %>% lapply(. %>% get_prophet_prediction(test_start, horizon))
sn_lists <- trns %>% lapply(. %>% get_seasonal_naive(test_start, horizon))

xgs_preds_frame <- xg_lists %>% lapply(. %$% xg_preds_frame) %>% add_state_combine()
prs_preds_frame <- pr_lists %>% add_state_combine()
sns_preds_frame <- sn_lists %>% add_state_combine()

xg_sn_comparison <- get_error_comparison(tst, xgs_preds_frame, sns_preds_frame)
xg_pr_comparison <- get_error_comparison(tst, xgs_preds_frame, prs_preds_frame)


.plot_better_by_day(xg_sn_comparison, pt_size = 1) +
  facet_wrap(~state, scales = "free_y")

all_preds <- bind_rows(xgs_preds_frame, prs_preds_frame, sns_preds_frame)

MID_HORIZON <- 200
.all_states_preds_plot <-
  .plot_preds_frame(trn,
                    tst,
                    all_preds,
                    test_start,
                    horizon = MID_HORIZON) +
  facet_wrap(~state, scales = "free_y", ncol = 3)

## xgboost and prophet errors are quite correlated
.xg_pr_error_corr_plot <- xg_pr_comparison %>%
  ggplot(aes(x = y_minus_yhat_xgboost,
             y = y_minus_yhat_prophet)) +
  corr_plot_elements

## xgboost and seasonal-naive errors are not terribly correlated,
## but also not uncorrelated enough to feel good at a glance.
## Let's see what averaging them gets us, though.
.xg_sn_error_corr_plot <- xg_sn_comparison %>%
  ggplot(aes(x = y_minus_yhat_xgboost,
             y = y_minus_yhat_seasonal_naive_365)) +
  corr_plot_elements
