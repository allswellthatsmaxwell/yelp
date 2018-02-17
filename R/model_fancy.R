
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
