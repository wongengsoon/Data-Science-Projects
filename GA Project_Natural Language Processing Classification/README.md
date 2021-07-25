### The Data Science Process

**1. Problem Statement**

We are a financial technology start up called Crypto Go that sells crypto range of products.

We are actively expanding advertising efforts to capture more market share.

However, with a limited budget, we need to maximize every advertising dollar with gaining good quality prospects who will be interested in our products.

Using Natural Language Processing, we have trained a classifier on user postings in crypto subreddit, together with another subreddit, with the Random Forest Classifier, MultinomialNB and Support Vector Classifier with a high degree of accuracy.

This is a binary classification problem - Whether a posting Belongs to Crypto subreddit OR Belongs to Other subreddit.

Our objective is to use the model to look for postings in other social media channels (e.g. Facebook, Youtube) to target our advertising of crypto products in these channels so that we can expand the market penetration of our crypto products to potential prospects. 

Success of the model will be evaluated by the Sensitivity score (or recall).

This project will be critical for our fintech startup as we will need to identify accurately channel postings in other social media channels (e.g. Facebook, Youtube) of users who have a deep interest in Crypto products 
so that we can expand our market penetration with a bigger prospects base.

The project will also be useful to Crypto lovers from other social media channels because they will be able to see the Crypto products solutioning that our fintech start up has to offer to meet their investment needs. 

The 2 subreddits from which we will be classifying the posts will be:

a. https://www.reddit.com/r/stocks/

b. https://www.reddit.com/r/CryptoCurrency/


### The Data Science Process

**2. Data Collection**

1. Identified the URL (CryptoCurrency + stocks)

url = 'https://api.pushshift.io/reddit/search/submission'

subreddit : CryptoCurrency , stocks

2. Called the API

res = requests.get(url, params)

3. Converted the API output to .json

data = res.json()

3 thousand posts of data from subreddit in CryptoCurrency and another 3 thousand from stocks were collected as train data.

### The Data Science Process

**3. Data Cleaning and EDA**

The following steps were performed to clean our data to get it ready for modeling:

1. Removed posts which were removed by moderators or posts with no selftext (null posts)

2. Tokenize the posts - Break sentences into words

3. Lemmatize the text - Break each word down to its root word - “Playing to play”

4. Removes stopwords - Remove the most common words to get rid of the ‘noise’

#### EDA:

Barchart and Word Cloud are plotted to show the most common words in CryptoCurrency and stocks:

Top 5 words in CryptoCurrency are:
1. Crypto
2. Bitcoin
3. Just
4. Like
5. Market

Top 5 words in stocks are:
1. Stocks
2. Amp
3. Market
4. Https
5. com

### The Data Science Process

**4. Preprocessing and Modeling**

Baseline Model

Pipeline created to do Gridsearch on:

1. Choosing CountVectorizer OR TfidfVectorizer and best params

2. Choosing best params for MultinomialNB, Random Forest Classifier and Support Vector Classifier

Modelling:

Either CountVectorizer OR TfidfVectorizer using best params

Model using MultinomialNB, Random Forest Classifier and Support Vector Classifier using best params



### The Data Science Process

**5. Evaluation and Conceptual Understanding**

Use accuracy score, ROC AUC score and sensitivity (recall) score (from confusion matrix) to evaluate 3 models

Calculate the best model using sensitivity (recall) score from the confusion matrix.

### The Data Science Process

**6. Conclusion and Recommendations**

Background:
We have indicated in the Problem Statement that our objective is to use the model to look for postings in other social media channels (e.g. Facebook, Youtube) to target our advertising of crypto products in these channels so that we can expand the market penetration of our crypto products to potential prospects.

1. Advertise to posts that our model tagged as interested in CryptoCurrency.

Spend the advertising dollars on True Positive-675 posts (True Crypto and classified correctly as Crypto) + False Negatives-74 posts (True Crypto but classified wrongly as stocks) = 749 posts

Instead of advertising to whole population of 675 (True Positives) + 74 (False Negatives) + 104 (False Positive) + 643 (True Negatives) = 1496 posts


#### Future Steps

1. Train on more subreddits to make the model more robust to different unknown datasets.

2. Train our model on different social media sites e.g. Youtube comments

3. Train on CryptoCurrency news articles so as to ennrich the model with CryptoCurrency keywords.

4. Get more data