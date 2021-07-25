# 1. Data Science Process - Define the Problem
- We are a Data Science team in Propnex Pte Ltd, a real estate company. 
- We have been with tasked with analysing the Ames Housing Dataset that is available on [Kaggle](https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge) and finding the key predictors that influence the Sales Price of houses in Ames city. 
- We will be creating a regression model that will help us make predictions with the best R<sup>2</sup> score.
- Through this analysis, we will make recommendations to homeowners on improving their property’s value and to home buyers on which features will command high prices.


# 2. Data Science Process - Gather data (Cleaning the data)

- Large amount of missing data is attributed to "NA" which means "Not Applicable" being mis-interpreted by the default CSV file reading process to mean "nan" (missing).
We imputed "NA" to reflect its original value. 

- Lot Frontage (Continuous): Linear feet of street connected to property has many missing values which may mean there are no street connected to property. 
However, this cannot be verified unless we have access to the data source or data collector. Given that large percentage (16%) of test data have missing values, 
and a suitable replacement cannot be found to be imputed, we will drop this column - "Lot Frontage"

- Garage area = 0 actually means there are no garage. We identify other garage related features whose garage area = 0 and have missing values in other garage related features and impute "NA" as there are no garage for this group.

- Basement area = 0 actually means there are no basement. We identify other basement related features whose basement area = 0 and have missing values in other basement related features and impute "NA" as there are no basement for this group.

- For other features with a small number of missing values, we impute "UnKn" or "-1.0"



# 3. Data Science Process - Explore data (Exploratory Data Analysis, EDA)
### To determine which features exhibit a correlation with the target "SalePrice".

- Each of the features are plotted against SalePrice to look for trends of correlation.

- Features which are continuous are plotted using scatter plot and correlation is examined.

- For rest, we use box plot to examine the median values of each category and how much they vary against SalePrice.

- Outliers are kept and not discarded because they are valid transactional data points and there are potentially important reasons why their data is outside the normal range. 

- Distribution of SalePrice is plotted and we see that it is negatively skewed (left skew).
To correct this, we plot Log Distribution of SalePrice and this improved the negative skewness.
This may potentially be an important factor in modelling as Log of SalePrice may need to be used as the Dependent variable instead of SalePrice itself.

- From the EDA, we see continuous variables with high correlation like Ground Living Area and ordinal variables with high correlation like Overall Quality and these features do prove that they are important determinants in the SalePrice of houses during the modelling stage.


# 4. Data Science Process - Model with Data
Preprocessing and Modeling

- After the train dataset from train.csv and test dataset from test.csv are both cleaned 
and EDA is carried out on the train dataset, we concatenate both together by stacking them vertically.

- Categorical variables are generated for all features classified as ordinal or nominal 
and they are added to the concatenated train and test datasets, called train_test_dummies.

- Feature engineering is carried out to add additional features to the original features.
This is because the interplay of features may have a multiplier effect to generating higher R Square score.
Choice of additional features added may be due to its high absolute coefficients OR features which we 
think will have a big impact on SalePrice.

- The train_test_dummies datasets with 4 additional features are then split into train datasets and test datasets.

- As the test datasets do not contain SalePrice, we are unable to evaluate R Square score on the test datasets.

- We need to compare R Square on both train datasets and test datasets to check for improvement (higher) in R Square score in test datasets. 
An improvement in R Square score in Test dataset will mean there is lower probability of overfitting in the model, so we need to find another way to do the train test split.

- We can do train test split in the train datasets as the train datasets contain SalePrice which allows us to calculate R Square Score.
Inside the train datasets, we divide the rows into X_train_valid (looks like our usual X_train) and X_valid (looks like our usual X_test).
Corresponding, their dependent variables will be called y_train_valid (looks like our usual y_train) and y_valid (looks like our usual y_test).

- For completeness sake, the test datasets will be called X_test and y_test (as mentioned earlier, y_test refers to SalePrice in test datasets and this data is not given).

- Scaling is done to scale all independent variables as they have different units which if not done, will have unequal impact onto the SalePrice in the model.
After scaling, X_train_valid will become X_train_valid_scal, X_valid will become X_valid_scal and X_test will become X_test_scal.


- The baseline score refers to the mean of the SalePrice when we do not consider the independent variables.

#### Linear Regression
- Cross validation for Linear Regression is done and cross_val_score is -1.0875970239468627e+24.
The mean R^2 from cross_val_score for Linear Regression is extremely negative. 
The Linear Regression is performing far worse than baseline on the train sets.
It is probably dramatically overfitting and the redundant variables are affecting the coefficients in weird ways.


#### Ridge Regression
- Cross validation for Ridge Regression is done and cross_val_score is 0.8512446841393162.
We first find the optimal value for Ridge regression alpha using RidgeCV and use it to find Cross Val Score for Ridge regression.

- R Square score for Ridge regression is 0.9421745080125637 for train datasets (X_train_valid_scal, y_train_valid).

- R Square score for Ridge regression is 0.9684873883445125 for validation datasets (X_valid_scal, y_valid).
We can see that using Ridge regression model on the train and validation datasets, as the R Square score has improved, there is lower probability of overfitting in the Ridge model.
Also, the mean R^2 from cross_val_score for Ridge Regression is vastly better than the Linear Regression. 
There is likely so much multicollinearity in the data that "vanilla" regression overfits and has bogus coefficients on predictors. 
Ridge is able to manage the multicollinearity and get a good out-of-sample result

- RMSE for validation dataset on Ridge regression is 13910

### - IN SUMMARY, FOR RIDGE REGRESSION, CROSS VAL IS 0.8512, R SQUARE FOR TRAIN IS 0.9422, R SQUARE FOR VALIDATION IS 0.9685.

#### WE DEFINE A GOOD MODEL TO BE ONE WHEREBY ALL 3 SCORES ARE APPROXIMATELY THE SAME. IN THIS CASE, THERE IS STILL ROOM FOR IMPROVEMENT IN THE RIDGE MODEL.


#### Lasso Regression
- Cross validation for Lasso Regression is done and cross_val_score is 0.8538532168890323.
We first find the optimal value for Lasso regression alpha using LassoCV and use it to find Cross Val Score for Lasso regression.

- R Square score for Lasso regression is 0.9349393939806893 for train datasets (X_train_valid_scal, y_train_valid).

- R Square score for Lasso regression is 0.9650167426506199 for validation datasets (X_valid_scal, y_valid).
We can see that using Lasso regression model on the train and validation datasets, as the R Square score has improved, there is little probability of overfitting in the Lasso model.
The Lasso performs slightly better than the Ridge in cross val score, but similarly. 
Lasso deals primarily with the feature selection of valuable variables, eliminating ones that are not useful. 
This also takes care of multicollinearity, but in a different way: it will choose the "best" of the correlated variables and zero-out the other redundant ones. 
There may also be useless variables in the data which it is simply getting rid of entirely.

- RMSE for validation dataset on Lasso regression is 14656

### - IN SUMMARY, FOR LASSO REGRESSION, CROSS VAL IS 0.8539, R SQUARE FOR TRAIN IS 0.9349, R SQUARE FOR VALIDATION IS 0.9650.

#### WE DEFINE A GOOD MODEL TO BE ONE WHEREBY ALL 3 SCORES ARE APPROXIMATELY THE SAME. IN THIS CASE, THERE IS STILL ROOM FOR IMPROVEMENT IN THE LASSO MODEL.



# 5. Data Science Process - Evaluate Model
#### Evaluation and Conceptual Understanding

#### Predicting with the test set

*Kaggle scores using Lasso Regression*
* Private: 19644.84070
* Public: 23085.24706

*Kaggle scores using Ridge Regression*
* Private: 20401.80596
* Public: 25146.93359

Lasso Regression performed consistently better in both Private and Public scores. This will be the preferred model to be used in predicting "SalePrice".

- After completing Ridge and Lasso Regression, we have dropped out the coefficients with coefficient = 0 from Lasso and run Lasso and Ridge regression again.
As the results are marginally worse off, we will take the first round of Lasso and Ridge Regression results as final for submission.

- To improve the model further such that cross val score may be closer to R Square score for train and validation datasets, we may look into plotting Log of SalePrice against features.


# 6. Data Science Process - Answer Problem
### Conclusion and Recommendations

Based on the Lasso coefficients, these are the top 10 features that I have determined to be useful. They have the strongest positive coefficient score.

* Gr Liv Area * Total Bsmt SF /// (Gr Liv Area (Continuous): Above grade (ground) living area square feet * Total Bsmt SF (Continuous): Total square feet of basement area)
* Gr Liv Area * Year Built /// (Gr Liv Area (Continuous): Above grade (ground) living area square feet * Year Built (Discrete): Original construction date)
* Year Built /// (Year Built (Discrete): Original construction date)
* Overall Qual_9 /// (Overall Qual (Ordinal): Rates the overall material and finish of the house / 9 - Excellent)
* Gr Liv Area * BsmtFin SF 1 /// (Gr Liv Area (Continuous): Above grade (ground) living area square feet * BsmtFin SF 1 (Continuous): Type 1 finished square feet)
* Overall Qual_10 /// (Overall Qual (Ordinal): Rates the overall material and finish of the house / 10 - Very Excellent)
* Overall Qual_8 /// (Overall Qual (Ordinal): Rates the overall material and finish of the house / 8 - Very Good)
* Garage Area /// (Garage Area (Continuous): Size of garage in square feet)
* Neighborhood_NridgHt /// (Neighborhood (Nominal): Physical locations within Ames city limits (map available) / NridgHt - Northridge Heights)
* Fireplaces /// (Fireplaces (Discrete): Number of fireplaces)

The 5 features that have the strongest negative coefficient scores are:

* Bsmt Qual_Gd /// (Bsmt Qual (Ordinal): Evaluates the height of the basement / Gd - Good (90-99 inches))
* Pool QC_NA /// (Pool QC (Ordinal): Pool quality /  NA - No Pool)
* Bldg Type_TwnhsE /// (Bldg Type (Nominal): Type of dwelling/TwnhsE- Townhouse End Unit)
* Kitchen Qual_Gd /// (KitchenQual (Ordinal): Kitchen quality /  Gd - Good)
* Bedroom AbvGr /// (Bedroom (Discrete): Bedrooms above grade (does NOT include basement bedrooms))

For homeowners and home buyers, I would recommend:
* (Buyers) Investing in houses with a large gross living area, large basement area and large garage area that are built recently and has more fireplaces may likely fetch a higher SalePrice later as these features command good premium.
* (Homeowners) Ensuring houses to be of good overall quality (interiors and exteriors), or spending on renovation will increase SalePrice of the house.
* (Homeowners) To ensure that the kitchen is inviting and must be in Excellent condition, - after all, cooking/entertaining is very trendy now
* (Buyers) Investing in Northridge Heights neighborhoods may likely fetch a higher SalePrice later as these features command good premium.