# Zillow_Home_Value_Prediction
Predicting the log-error between 'Zestimate' and actual home value

## Introduction
This project aims to predict the log(error) of Zillow’s estimate ('Zestimate') and the actual sale prices in California for a specified number of months in 2016 and 2017. According to the National Association of Realtors, more than 5 million units of existing homes were sold in the US in 2020. Home purchase signifies a significant expense for any individual, therefore, a lot of research goes into buying a home. Home price estimates would give the seller and the buyer a reference point, which would thus reduce the time and effort of both parities involved. As a result, a good home price estimate would reduce a lot of unnecessary cost, and would help both the buyers and sellers.

In this project, I use the dataset provided by Zillow for a Kaggle competition, and apply linear and gradient boosting regression to predict the log(error) of Zillow’s estimate. I use the Mean Absolute Error to evaluate the model as it is mathematically easy to define and we can figure out the difference in the price error of the estimate.

![](images/ZillowKaggle.jpg)

## Data
The data for this project comes from Zillow for a Kaggle competition. The dataset 60 features that described the location, year built, square footage, number of bedrooms and bathrooms, tax amount, among many others. The model aims to predict the log(error), which is described as:

logerror = log(Zestimate) - log(SalePrice)

## Exploratory Data Analysis
To save on memory, I explored the number of properties in my dataset that did not have a matching train record. I found that almost 3 million properties did not have corresponding target data (those properties did not sell, therefore, we do not know the sale price), so those entries were separated out. 

One thing I did during the initial analysist was to check the range for the different transaction dates and it is visualized below:

![](images/transactiondatehist.png)

I also found that some of the homes were sold twice during the period that the training data was captured which left multiple records for a particular parcelId. 
Because only ~100 of the ~90,000 properties had multiple records, we chose to just take a random sales record for each parcel with more than one sales record (as opposed to... always using the most recent record, always using the oldest record, using both records, or engineering a feature from the info).

When exploring Unique Values in the dataset a few things that I looked for are:
* Are any of the features all of the same value? These won't be useful in a model, so I discarded them
* Are any features discrete with high cardinality? These wouldn't hold very much information because there would only be a few records for each "level" of the categorical feature

I also found it interesting to explore the Target Variable, shown below:

![](images/targetdistribution.png)

I keep this in mind when choosing Machine Learning Algorithms and corresponding loss functions. When I plotted the density on a log-scale, the target looks closer to a Normal Distribution:

![](images/targetlogscale.png)

## Preprocessing & Feature Engineering
By creating a preprocessing tool instead of script, preprocessed data file, I was able to change things on the fly during the model build process and speed up iterations.
In the `02_Preprocessing` Notebook, I did some feature engineering and missing value imputation. Including:
* Filter features
* Encoding missing values with a unique value (for that column)
* Encoding categorical features for us in common Machine Learning algorithms

One example that required `encoding` was the datetime column. For this, I encoded the datetime variable as month and year (disregard day because it won't be included in the data we'll be scoring).

Because most Machine Learning Algorithms do not handle categorical features well, I encoded them as a real number. To be flexible during modeling, I encoded a set of categorical variables with a set of binary features (`dummy encoding`). 

## Modeling
#### Defining success
The final model parameters was based on the evaluation criteria used by the Kaggle competition, Mean Squared Error (MAE).
#### Tune tree depth
`learning_rate` and `max_estimators` are indirectly-correlated; the higher the learning rate the fewer trees are needed. Increasing the learning rate will make the models fit faster. This comes with a bit of a hit to accuracy though, so a popular technique is to tune the tree depth with a higher learning rate and then tune the number of trees with a lower learning rate.

While training the model, I looked at the learning curve and compared the MAE of models with a max depth of 2, 3, and 4.

***** training for max depth = 2 *****
optimal number of trees: 538
best MAE: 0.0670813421874875
baseline MAE: 0.06803804369747765

![](images/maxdepth2.png)

***** training for max depth = 3 *****
optimal number of trees: 435
best MAE: 0.06709945059252274
baseline MAE: 0.06803804369747765

![](images/maxdepth3.png)

***** training for max depth = 4 *****
optimal number of trees: 182
best MAE: 0.06711932431156699
baseline MAE: 0.06803804369747765

![](images/maxdepth4.png)

The final parameters settled on were:
* n_estimators = 1000
* learning_rate = 0.1
* max_depth = 2
* loss = 'lad'
* subsample = 0.5

* optimal number of trees: 560
* best MAE: 0.06712509883803039
* baseline MAE: 0.06803804369747765

![](images/finalparameters.png)

This shows that the model has true predictive power because the MAE (0.0671) is lower than the baseline MAE (0.0680).

#### Create model object
I used a Scikit-Learn Pipeline to combine the preprocessing and GBM steps in one package. The pipeline object called `my_model` implements `fit` and `predict` methods (among others). When we call the `fit` method, the pipeline will execute `preprocessor.fit_transform()` on the data we pass in the arguments, then pass the results to `GradientBoostingRegressor.fit()`. Similarly, when we call the `predict` method, it will execute `preprocessor.transform()` and then `GradientBoostingRegressor.predict()`.

#### Fit final model
I then fit the final model using the training data and saved the results in a folder titled `models`.

## Results
#### Score test dataset
I only did this once to avoid overfitting the test set. This test dataset used was released after the Kaggle competition was over, but is not the same format as what is scored as part of the competition. The final evaluation typically would be making a Kaggle submission (optional) however I used this test set to visualize the model results on an out-of-time test set, meaning this data is from another period as the data I trained on the model.

#### Model evaluation
I evaluated the model using the `naive median prediction method` and resulted in a Mean Absolute Error of 0.0699. This naive forecast involves using the previous observation directly as the forecast without any change.

When evaluating using the actual model, I got a Mean Absolute Error of 0.692. Because the MAE went down, this means that the model does have predictive power. THIS IS A GOOD SIGN. 
* Note that the MAE does not decrease by that much, but that is because I am trying to predict residuals of an existing model

#### Plot model results 

##### Actuals vs. Predictions

![](images/actualsVprediction.png)

This plot shows exactly where the lack of predictive power is come from - The model is predicting every really close to the average or median (probably median given the choice of loss function) instead of making more useful predictions. Notice that we are predicting everything about zero. This is not a great sign and speaks to why do not see a huge difference in the error between predicting the median and the model predictions.

This is one place where I would be able to show some of the value of the model if I worked for Zillow and knew more about the business model I could know how much the reduction of error nets the company $X amount in revenue.

##### Distribution of Actuals and Predictions

![](images/PredictionsActualsHistogram.png)

Note that the model predictions have far less variance than our true response values.

##### Checking importance of features

![](images/FeatureImportance.png)

You can see that tax amount, latitude and logitude (location), square footage, and so on, have the highest importance. These feature importances are on the model for predicting the error between the Zestimate and the actual sale price, so we can conclude that the features are the ones Zillow's model aren't fully capturing signal from.

## Appendix

##### Python Notebooks:
* 01_Zillow_Home_Value_Prediction_EDA.ipynb 
* 02_Preprocessing.ipynb
* 03_ModelTuning&Fitting.ipynb

##### Folders:
* images - holds images and visuals
* modules - helper and prerocessor scripts
* models - holds model

Kaggle: [https://www.kaggle.com/c/zillow-prize-1/overview]
