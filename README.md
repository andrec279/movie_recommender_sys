# Project Description
This project represents my first attempt at building a relatively large-scale content recommender model with a focus on building the following skills:
1. Running a parallelized model on a large, partitioned dataset in a cluster environment (Hadoop)
2. Implementing and tuning a latent factor model (Alternating Least Squares) as the recommender model algorithm
3. Implementing a "content cold start" model extension to recommend new content with no prior user interactions
4. Data preprocessing strategies for recommender systems
5. Evaluation strategies for recommender systems

This project was completed as a final course project at NYU, and leveraged NYU's cluster environment for data storage and model execution.

# Overview

Below is a description of the input data, model implementation, and final results.

## The data set

For this project, we used the [MovieLens](https://grouplens.org/datasets/movielens/latest/) datasets collected by 
> F. Maxwell Harper and Joseph A. Konstan. 2015. 
> The MovieLens Datasets: History and Context. 
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

The MovieLens dataset contains ratings from about 280,000 users on over 58,000 unique movies. Additionally, we used data contains content-based features for 10,000 movies in the MovieLens dataset in the form of "tag relevancy scores", collected by [GroupLens](https://grouplens.org/datasets/movielens/tag-genome/). Both datasets are hosted in NYU's HPC environment.


## Basic recommender system [80% of grade]

1.  As a first step, you will need to partition the rating data into training, validation, and test samples as discussed in lecture.
    I recommend writing a script do this in advance, and saving the partitioned data for future use.
    This will reduce the complexity of your experiment code down the line, and make it easier to generate alternative splits if you want to measure the stability of your
    implementation.

2.  Before implementing a sophisticated model, you should begin with a popularity baseline model as discussed in class.
    This should be simple enough to implement with some basic dataframe computations.
    Evaluate your popularity baseline (see below) before moving on to the enxt step.

3.  Your recommendation model should use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.
    Be sure to thoroughly read through the documentation on the [pyspark.ml.recommendation module](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html) before getting started.
    This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably: 
      - the *rank* (dimension) of the latent factors, and
      - the regularization parameter.

### Evaluation

Once you are able to make predictions—either from the popularity baseline or the latent factor model—you will need to evaluate accuracy on the validation and test data.
Scores for validation and test should both be reported in your write-up.
Evaluations should be based on predictions of the top 100 items for each user, and report the ranking metrics provided by spark.
Refer to the [ranking metrics](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems) section of the Spark documentation for more details.

The choice of evaluation criteria for hyper-parameter tuning is up to you, as is the range of hyper-parameters you consider, but be sure to document your choices in the final report.
As a general rule, you should explore ranges of each hyper-parameter that are sufficiently large to produce observable differences in your evaluation score.

  - *Cold-start*: using supplementary metadata (tags, genres, etc), build a model that can map observable data to the learned latent factor representation for items.  To evaluate its accuracy, simulate a cold-start scenario by holding out a subset of items during training (of the recommender model), and compare its performance to a full collaborative filter model.  *Hint:* you may want to use dask for this.

## Credits: 
- Thank you to Adi Srikanth and Jin Ishizuka as contributors on this project, who helped build a baseline popularity model for comparison against the ALS recommender model and fix / tune the ALS model as needed
- Thank you to Professor Brian McFee for guidance and the NYU High Performance Computing group for allowing use of their cluster resources throughout this project
