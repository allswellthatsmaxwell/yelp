CSV_PATTERN <- "(.*)\\.csv$"

make_friend_graph <- function(users_frame) {
  friends <- lapply(users_frame$friends,
                    function(comma_friends) comma_friends %>%
                                            strsplit(",") %>%
                                            .[[1]] %>%
                                            trimws())
  edges <- Map(function(user, friend_vec) {
                 tibble(user = user, friend = friend_vec)
               },
               users_frame$user_id,
               friends) %>%
    bind_rows()
  as_tbl_graph(edges, directed = TRUE)
}


#' extract name from a file matching the pattern name.csv
#' Errors if filename doesn't match the pattern.
extract_csv_name <- function(csv_filename) {
  matches <- stringr::str_match(basename(csv_filename), CSV_PATTERN)
  if (all(is.na(matches))) stop(glue("{csv_filename} is not a valid csv name"))
  matches[,2]
}

.error_by_day_plot <- stacked_accuracies %>%
  mutate(error = y - yhat) %>%
  ggplot(aes(x = ds, y = error, color = group)) +
  geom_line() +
  facet_wrap(~state)





## a mess follows ##############################################################

trn_fair_horizon <- trn %>% filter(ds < max(trn$ds) - horizon)

xg_y_trn <- trn %>% arrange(ds) %$% y
## We can use the lags from the training set to predict out-of-sample
## values. This isn't cheating.
xg_y_tst <- tst %>% arrange(ds) %$% y %>% {c(take_last_n(xg_y_trn, NLAGS), .)}

trn_lag_mat <- get_n_lag_matrix(xg_y_trn, NLAGS)
trn_response <- trn_lag_mat[,1]
trn_predictors <- trn_lag_mat[,-1]

tst_lag_mat <- get_n_lag_matrix(xg_y_tst, NLAGS)
tst_response <- tst_lag_mat[,1]
tst_predictors <- tst_lag_mat[,-1]

xg_model <- xgboost(trn_predictors, trn_response, nrounds = 100)
xg_fit <- predict(xg_model, newdata = trn_lag_mat)
xg_fcast <- predict(xg_model, newdata = tst_lag_mat)

xg_trn_ds <- seq.Date(train_start + NLAGS, train_end, by = 1)
xg_tst_ds <- seq.Date(test_start, test_end, by = 1)

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

multiple_models_frame <- xg_fcast_frame %>%
  bind_rows(mutate(prophet_result, type = "prophet"))

multiple_models_frame %>%
  ggplot(aes(x = ds, y = y, color = type)) +
  geom_line() +
  theme_bw()


####
## Neural net (FF, single hidden layer... poor default performance) ############
####
library(forecast)

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

fcast_dat %>%
  bind_rows(fit_dat, one_state_dat_complete, prophet_result) %>%
  filter(year(ds) >= 2015) %>%
  ggplot(aes(x = ds, y = y, color = type)) +
  geom_point(alpha = 0.2) +
  theme_bw()
