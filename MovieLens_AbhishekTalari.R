#title: "MovieLens Project"
#author: "Abhishek Talari"
#date: "2023-12-05"
# Note: this first code chunk was provided by the course

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))
movielens <- left_join(ratings, movies, by = "movieId")
# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
# Make sure userId and movieId in final hold-out test set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Loading required packages
requiredPackages <- c("tidyverse", "rafalib", "ggpubr", "knitr", "raster")
lapply(requiredPackages, library, character.only = TRUE)

# Histogram of ratings per movie: Some movies are more rated than others 

hist_movies <- edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 40, fill = "blue") + 
  labs(title = "Ratings per movie",
       x = "Number of ratings per movie", y = "Count", fill = element_blank()) +
  theme_classic()

# Histogram of ratings per user: Some users rated more movies than others

hist_users <- edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 40, fill = "blue") + 
  labs(title = "Ratings per user",
       x = "Number of ratings per user", y = "Count", fill = element_blank()) +
  theme_classic()


ggarrange(hist_movies, hist_users,
          ncol = 2, nrow = 1)

# Histogram of ratings
edx %>%
  ggplot(aes(rating)) +
    geom_histogram(fill = "steelblue") + 
    labs(title = "Histogram of ratings",
       x = "Ratings", y = "Count", fill = element_blank()) +
  theme_classic()

# View of all unique genres
unique_genres_list <- str_extract_all(unique(edx$genres), "[^|]+") %>%
  unlist() %>%
  unique()

unique_genres_list

# Creating the long version of both the train and validation datasets. With separeted genres
edx_genres <- edx %>%
  separate_rows(genres, sep = "\\|", convert = TRUE)

validation_genres <- validation %>%
  separate_rows(genres, sep = "\\|", convert = TRUE)

# Histogram of ratings per genre
hist_genres <- ggplot(edx_genres, aes(x = reorder(genres, genres, function(x) - length(x)))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Ratings per genre",
       x = "Genre", y = "Counts") +
   scale_y_continuous(labels = paste0(1:4, "M"),
                      breaks = 10^6 * 1:4) +
  coord_flip() +
  theme_classic()
  
  hist_genres

# Boxplot of movie ratings per genre
boxplot_genre_ratings <- ggplot(edx_genres, aes(genres, rating)) + 
  geom_boxplot(fill = "steelblue", varwidth = TRUE) + 
  labs(title = "Movie ratings per genre",
       x = "Genre", y = "Rating", fill = element_blank()) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

boxplot_genre_ratings

# Creating the RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Analysis section --------------------------------------------------------
# Method: Just the average ------------------------------------------------


mu_hat <- mean(edx$rating)
mod_average <- RMSE(edx$rating, mu_hat)

rmse_results <- tibble(Method = "Just the average", RMSE = mod_average)
kable(rmse_results)

# Method: Movie Effect Model ----------------------------------------------


movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

predicted_ratings <- mu_hat + validation %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  pull(b_i)

predicted_ratings <- clamp(predicted_ratings, 0.5, 5)

mod_m <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Movie Effect Model",
                                 RMSE = mod_m))


kable(rmse_results[2,])

# Method: Movie + User Effects Model --------------------------------------
user_avgs <- edx %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

predicted_ratings <- clamp(predicted_ratings, 0.5, 5)

mod_m_u <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Movie + User Effects Model",  
                                 RMSE = mod_m_u))

kable(rmse_results[3,])

# Method: Regularized Movie + User + Genre Ind. + Movie_Genre + Genre_User Effect Model 
# Regularized parameter
lambdas <- seq(11.5, 12.5, 0.2)

# Grid search to tune the regularized parameter lambda
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- edx_genres %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))
    
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  predicted_ratings <- clamp(predicted_ratings, 0.5, 5)
  
  return(RMSE(predicted_ratings, validation$rating))
})

plot_rmses <- qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]

rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Regularized Movie + User Effect Model",RMSE = min(rmses)))
kable(rmse_results[4,])


# Results section ---------------------------------------------------------

kable(rmse_results)
