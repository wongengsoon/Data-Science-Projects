# CAPSTONE DATA SCIENCE PROJECT
# How to use banking transactions data to do liquidity forecasting, client segmentation and loan default prediction?


## Problem Statement

We are the Data Scientists in AI Lab in ABC Bank that specialises in exploratory data analysis and modelling for the bank.

We will be using banking transactions data to give insights on liquidity forecasting, customer segmentation and loan default prediction.

On liquidity forecasting, we will be using banking transaction data to forecast the amount of liquidity which the bank needs to hold to satisfy the withdrawals required by its borrowers.
We will be measure accuracy of the SARIMA time-based modelling via the mean squared error.

On Customer Segmentation, we aim to generate leads and propose recommendations to increase sales and revenue for the bank. 
We will be using K-means clustering to segmentise the customers and using the silhouette score to obtain the optimal number of clusters.

On Loan Default Prediction, we will be using banking transactions data, together with client demographics data, to enrich the loans data.
We will be using various classification models to do the prediction modelling and using Accuracy metrics and ROC AUC to score the models.

The analysis and findings will provide valuable insights for Senior Bank Management to aid them in their decision making processes.

---

## Data Collection

1999 Czech Financial Dataset - Real Anonymized Transactions by Liz Petrocelli

The dataset is a collection of financial information from a Czech bank that deals with over 5,300 bank clients with approximately 1,000,000 transactions. 

Additionally, the bank represented in the dataset has extended close to 700 loans and issued nearly 900 credit cards, all of which are represented in the data. 

Source: 
1. https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions<br>
2. https://webpages.uncc.edu/mirsad/itcs6265/group1/index.html

---

## Data Cleaning & EDA

**Data Cleaning**

1) Conversion of data column fields from Czechoslovak language to English <br>
2) Dates correction<br>
3) Separation of Birth Number into Birthday and Gender

**EDA**

1) On Transactions data, by Transaction Types (Withdrawal/Credit):<br>
Number of counts of Transactions Types for Withdrawal is 200k more than Credit.					
In terms of Transactions Amount, we see more Withdrawals with large transactional amounts than Credit.		
Histogram of Transaction Amounts shows Withdrawals have MORE counts of Withdrawals at almost every Transaction Amounts than Credit.
    
2) On Transactions data, by Transaction Operations:<br>
Transactions Operations shows Cash Withdrawals having largest % 

3) On Transactions data, by Transaction Characteristics:<br>
Transactions by Characteristics shows Unknown category are most highly skewed in their Transactions Amounts

4) On Loans data, Loan Default is affected by big monthly loan payment, long loan duration and big loan amount

5) On Transaction Amount, daily liquidity shortfall reaches (80,000) at its peak and this occurs during the mid-year period from 1993 to 1998.

6) On Transaction Amount, as time period increases from Weekly to Yearly, the Average Transaction Amount averages to zero

---

## Modeling, Evaluation, Conclusions and Recommendations

**Modeling**

**1. Time Series Forecasting using Transactions Amounts Data:**

On Liquidity forecasting, SARIMA model with parameters Weekly Transactions Amount with SARIMA (3,0,0) x (2,0,1,52) is able to predict transaction amounts (green) closely with test data (yellow) with minimum mean squared error of 490k.

Future scope for enhancement may include including exogenous variables like economic growth, national stock market index or strength of national currency as macroeconomic factors may play a part in the transactions amount fund flows.

**2. Client Segmentation using K-means clustering:**

Using k = 10 clusters with silhouette score of 0.6227 to cluster the clients from the Transactions Data gives an optimal number of clusters AND also sufficient number of clients per cluster with adequate Transaction Amounts and Balances.

Cluster 4 having high transaction balance and amount has transaction operation from Collection From Other Bank and Credit In Cash.
<br>Product Recommendations for Cluster 4 may include Investments, Loan and Insurance Product Solutioning.

Using time based modelling on Cluster 4 (1998 Transactions Data), we are able to forecast transaction amounts (green) closely with test data (yellow) with mean squared error of 13m.
<br>With Leads Generation and Recommendations taken to cross-sell Banking Products, we expect Transactions Amount to be higher than below forecast.

Future scope for enhancement may include client risk rating data so that recommendations for products  may be better tailored for clients' appetite for risk.

**3. Loans Default Prediction Classicfication:**

Gradient Booasting Classifier, Random Forest Classifier and Decision Tree Classifier all achieve perfect/almost perfect score of 1 in the Model Accuracy Score AND ROC Score for both train and test dataset.

It is worth noting that Monthly loan payment, loan amount, loan duration and balance has consistently appeared in all 3 models for Gradient Booasting, Random Forest and Decision Tree Classifier as the top 4 most important features.

Future Scope for enhancement may include adding in macroeconomic factors like economic growth, national stock market index, Central Bank benchmark interest rates and strength of national currency as these may also influence an individual's ability to service the loan.





















