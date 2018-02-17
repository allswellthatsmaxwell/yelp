
library(forecast)
library(magrittr)

combine_ts_train_test <- function(y_trn, y_tst, yhat,
                                  trn_start, trn_end,
                                  tst_start, tst_end) {
  tibble(ds = trn_start:test_end,
         y = c(y_trn, y_tst, yhat),
         type = c(rep("actual", length(y_trn) + length(y_tst)),
                  rep("fcast", length(yhat))))
}

STATE <- "Arizona"

one_state_dat <- state_review_values_by_date %>%
  prepare_for_state_modeling() %>%
  filter(state == STATE) %>%
  ungroup() %>%
  select(-state)

train_start <- min(one_state_dat$ds)
train_end <- as.Date("2016-07-01")
test_start <- train_end + 1
test_end <- max(one_state_dat$ds)

ds_all <-
  tibble(ds = seq.Date(from = train_start, to = test_end, by = 1),
         period = case_when(between(ds, train_start, train_end) ~ "train",
                            between(ds, test_start, test_end) ~ "test"))

one_state_dat_complete <- one_state_dat %>%
  right_join(ds_all, by = "ds") %>%
  ## The missing dates are all early and in the time of low activity,
  ## and it seems like Yelp has provided all non-zero days,
  ## so assume missing days are zeros.
  mutate(y = ifelse(is.na(y), 0, y)) %>%
  mutate(type = "actual")

trn <- one_state_dat_complete %>% filter(period == "train")
tst <- one_state_dat_complete %>% filter(period == "test")
y_trn <- trn %>% arrange(ds) %$% ts(y)
y_tst <- tst %>% arrange(ds) %$% ts(y)

model <- nnetar(y_trn)

fit_dat <- tibble(ds = ds_all %>% filter(period == "train") %$% ds,
                  y = model$fitted,
                  type = "fitted")

fcast <- forecast(model, h = ds_all %>% filter(period == "test") %>% nrow)
fcast_dat <- tibble(ds = ds_all %>% filter(period == "test") %$% ds,
                    y = fcast$mean,
                    type = "predicted")

prophet_result <- stacked_accuracies %>%
  filter(state == STATE) %>%
  ungroup() %>%
  select(ds, yhat, group) %>%
  rename(type = group, y = yhat) %>%
  filter(type == WITH_HOLS_NAME)

fcast_dat %>% bind_rows(fit_dat, one_state_dat_complete, prophet_result) %>%
  filter(year(ds) >= 2015) %>%
  ggplot(aes(x = ds, y = y, color = type)) +
  geom_point(alpha = 0.2) +
  theme_bw()


## Forecasting with ensemble model.

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


NLAGS <- 100
trn_lag_mat <- get_n_lag_matrix(y_trn, NLAGS)
trn_response <- trn_lag_mat[,1]
trn_predictors <- trn_lag_mat[,-1]

tst_lag_mat <- get_n_lag_matrix(y_tst, NLAGS)
tst_response <- tst_lag_mat[,1]
tst_predictors <- tst_lag_mat[,-1]

xg_model <- xgboost(trn_predictors, trn_response, nrounds = 200)
xg_fit <- predict(xg_model, newdata = trn_lag_mat)
xg_fcast <- predict(xg_model, newdata = tst_lag_mat)

xg_trn_ds <- seq.Date(train_start + NLAGS, train_end, by = 1)
xg_tst_ds <- seq.Date(test_start + NLAGS, test_end, by = 1)

xg_fit_frame <- tibble(ds = xg_trn_ds, y = xg_fit,
                       type = glue("xgboost ({NLAGS})")) %>%
  bind_rows(tibble(ds = xg_trn_ds, y = trn_response,
                   type = "actual"))

xg_fcast_frame <- tibble(ds = xg_tst_ds, y = xg_fcast,
                         type = glue("xgboost ({NLAGS})")) %>%
  bind_rows(tibble(ds = xg_tst_ds, y = tst_response,
                   type = "actual"))

xg_fit_frame %>% ggplot(aes(x = ds, y = y, color = type)) +
  geom_line() +
  theme_bw()

xg_fcast_frame %>% ggplot(aes(x = ds, y = y, color = type)) +
  geom_line() +
  theme_bw()