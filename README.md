# Project Description
This project represents my first attempt at building a relatively large-scale content recommender model with a focus on building the following skills:
1. Running a parallelized model on a large, partitioned dataset in a cluster environment (Hadoop)
2. Implementing and tuning a latent factor model (Alternating Least Squares) as the recommender model algorithm
3. Implementing a "content cold start" model extension to recommend new content with no prior user interactions
4. Data preprocessing strategies for recommender systems
5. Evaluation strategies for recommender systems

This project was completed for one of my courses in NYU's Masters in Data Science program, and leveraged NYU's cluster environment for data storage and model execution. All code included in this repository was initially written by myself, with some updates and contributions from teammates.

# Overview

Below is a description of the input data, model implementation, and final results.

## Data

For this project, we used the [MovieLens](https://grouplens.org/datasets/movielens/latest/) datasets collected by 
> F. Maxwell Harper and Joseph A. Konstan. 2015. 
> The MovieLens Datasets: History and Context. 
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

The MovieLens dataset contains ratings from about 280,000 users on over 58,000 unique movies. Additionally, we used data contains content-based features for 10,000 movies in the MovieLens dataset in the form of "tag relevancy scores", collected by [GroupLens](https://grouplens.org/datasets/movielens/tag-genome/). Both datasets are hosted in NYU's HPC environment. Timestamped ratings data served as the main data source for this project, as they provided the user-item interactions data required to produce vectorized user / item representations for our recommender model.

**Note on train-validation-test splits:** When splitting the data into train-validation-test partitions, we separated the interactions (rating) data temporally (earlier interaction timestamps in training, later interaction timestamps in validation / test, as would be the case in production). Additionally, we performed splits along interaction data for each user to ensure users in the evaluation sets were trained on in the training data and removed users / items for which interactions data was too limited (below some established threshold). This effectively meant we chose to evaluate our system on users and items above a certain "maturity" level.


## Model Implementation
We built three different versions of a movie recommender system for evaluation and comparison, varying strategies for data preparation, feature engineering, and model implementation for each. In each approach, the goal was the same: **produce 100 recommendations for each user in the dataset that are as close to their actual top 100 rated movies as possible.** All three versions are outlined below.

### Version 1: Popularity Baseline Model
Highly simplistic model with no personalization used as a baseline to evaluate against versions 2 and 3. In this approach, we identified the top 100 most popular movies across all users and recommended this set to each user, irrespective of their individual interactions history. Specifically, we computed average ratings for each movie, divided by a damping factor inversely related to the number of ratings attributed to that movie. In this manner, we assigned a "popularity score" to each movie that increased with rating scores and number of reviews, which we then used to find the top 100 most "popular" movies to recommend to each user. This model was optimized by tuning the weight of the damping factor, which controlled how important the number of reviews was to a product's popularity score.

### Version 2: Latent Factor Model
Collaborative Filtering model that used the Alternating Least Squares (ALS) algorithm to learn and compute latent factors (d-dimensional vectors that each represent individual users and items). Briefly, ALS aims to factorize a ratings matrix R into a user matrix U and an item matrix V (where each row in U represents a user and each column in V represents an item) such that UV approximates R as closely as possible. The method is termed "Alterning Least Squares" because it treats this problem as a minimization of sum of squares of the difference between elements in R and in UV, which it accomplishes by "alternating" between adjusting U and V using gradient descent until convergence conditions. Note that there is also often a regularization parameter added to the sum of squares in the ALS objective function which penalizes user / item vectors with high L2 norms.

To implement the ALS model on our ratings dataset, we used PySpark's ALS implementation in the [pyspark.ml.recommendation module](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html), which integrated nicely with PySpark dataframes for parallelization on NYU's High Performance Cluster during model training. While the ALS algorithm has a number of important hyperparameters for tuning, we focused on the following three to optimize:
- **Rank**: Dimension of the latent user / item factors. Higher rank vectors encode more information but are more costly to compute
- **RegParam**: Weight assigned to regularization term in the ALS objective function to prevent overfitting
- **MaxIter**: Maximum number of ALS training iterations used to compute optimal U and V matrices

At model inference time, we used scikit-learn's Nearest Neighbors search implementation to identify the top 100 item factors for each user factor by increasing cosine distance. The intuition behind using this approach was that since the ratings matrix R contains ratings (user-item affinity scores), and we assume UV approximates R reasonably well, the dot product of row U_i and column V_i should represent user i's affinity for item i, and cosine distance between vectors is inversely related to their dot product.

### Version 3: Latent Factor Model with Content Cold Start
Version 3 builds upon Version 2 and attempts to address the problem of content cold start that arises in production recommender systems: when new items are added to a catalogue, a pure collaborative filtering approach fails for that item due to lack of prior interaction data for the product. Here we explore an approach to produce and evaluate content-based item representations to remedy this issue. To obtain content data for this approach, we used the "tag relevancy scores" dataset from GroupLens (see the **Data** section for reference), which provides a list of relevancy scores for different content-based tags (think tags that frequently associate with Netflix shows like "funny", "dark", "sophisticated") for 10,000 movies that appear in the original MovieLens ratings dataset.

The modified approach to generating and evaluating user / item representations using ALS with Content Cold Start proceeded as follows:
1. Generate d-dimensional user / item factors from all interactions data using pure collaborative filtering (ALS)
2. For movies with GroupLens tag data, train a multivariate multiple regression model (with L2 regularization) to learn the movie's d-dimensional item factors generated in step 1 from content information (tag scores).
3. Remove all interaction data for 10% of movies from the original MovieLens rating dataset to simulate cold start conditions, and generate a new set of user / item factors with the remaining interactions data using ALS
4. Use the regression model from step 2 to compute item factors for the 10% of held out items from step 1 and append those factors to the items matrix generated in step 3.
5. Generate top 100 movie recommendations for each user using the same Nearest Neighbor search approach in Version 2: Latent Factor Model.


## Evaluation, Results, and Discussion
Models were evaluated on how closely the top 100 recommendations for each user matched their top 100-ranked movies by rating on average, taking order into account. To do so, we computed ranking *Mean Average Precision* (MAP) for each model using PySpark's implementation in their [mllib-evaluation-metrics module](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems). 

MAP results for each of the 3 model versions were as follows:
- Version 1 (Baseline): 0.017
- Version 2 (Latent Factor): 0.028
- Version 3 (Latent Factor with Cold Start): 0.011

It appears that using collaborative filtering for personalization of recommendations drove a significant improvement in performance over the baseline, while the first iteration of our content cold-start approach underperformed relative to baseline. Overall, it's worth noting that all three MAP scores appear quite low, which may have been due in large part to penalization of incorrectly ordered movie recommendations. In future work building recommendation systems, it may be worth exploring alternative evaluation metrics, such as Precision @ k, that don't take ordering into account to get a more holistic profile of model performance.


## Credits: 
- Thank you to Adi Srikanth and Jin Ishizuka as contributors on this project, who helped build a baseline popularity model for comparison against the ALS recommender model and fix / tune the ALS model as needed
- Thank you to Professor Brian McFee for guidance and the NYU High Performance Computing group for allowing use of their cluster resources throughout this project
