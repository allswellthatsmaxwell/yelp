fmt_text_date <- function(dt) format(dt, "%B %d, %Y")


pdf_gg_facet_theme <- list(theme(strip.text.x = element_text(size = 30),
                                 axis.text = element_text(size = 26),
                                 axis.text.x = element_text(angle = 90),
                                 axis.title = element_text(size = 34),
                                 legend.text = element_text(size = 32),
                                 title = element_text(size = 42)))

pdf_gg_single_theme <- list(theme(axis.text = element_text(size = 30),
                                  axis.text.x = element_text(angle = 90),
                                  axis.title = element_text(size = 34),
                                  legend.text = element_text(size = 27),
				  plot.title = element_text(size = 40),
                                  plot.subtitle = element_text(size = 34)))
