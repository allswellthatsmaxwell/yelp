library(prophet)
library(ggplot2)
library(data.table)
library(dplyr)
library(magrittr)
library(purrr)
library(glue)
library(readr)
library(lubridate)

#' extract name from a file matching the pattern name.csv
#' Errors if filename doesn't match the pattern.
extract_csv_name <- function(csv_filename) {
  matches <- stringr::str_match(basename(csv_filename), CSV_PATTERN)
  if (all(is.na(matches))) stop(glue("{csv_filename} is not a valid csv name"))
  matches[,2]
}

filter_biz <- function(dat, businesses_to_keep) {
  dplyr::inner_join(dat, businesses_to_keep, by = "business_id")
}

read_dat <- function(csv_basename) {
  readr::read_csv(glue("{DATA_DIR}/{csv_basename}"),
                  progress = FALSE)
}


DATA_DIR <- "../data"
N_BUSINESSES <- Inf
CSV_PATTERN <- "(.*)\\.csv$"
MIN_REVIEWS_IN_2017 <- 300

businesses <- read_dat("yelp_business.csv")
businesses_to_keep <- businesses %>%
  select(business_id) %>%
  unique() ## %>% top_n(N_BUSINESSES)

checkins <- read_dat("yelp_checkin.csv") %>% filter_biz(businesses_to_keep)
reviews <- read_dat("yelp_review.csv") %>%
  filter_biz(businesses_to_keep)

reviews %<>% select(-text)

businesses_and_reviews <- businesses %>%
  ## filter(review_count >= MIN_REVIEW_COUNT) %>%
  select(business_id, name, review_count, state, city) %>%
  inner_join(reviews, by = "business_id")

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
  facet_wrap(~state, scales = "free") +
  theme_bw()

daily_avg_stars_plot <- state_review_values_by_date %>%
  ggplot(aes(x = date, y = mean_stars)) +
  geom_line() +
  facet_wrap(~state) +
  theme_bw()







