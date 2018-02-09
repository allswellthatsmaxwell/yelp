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
