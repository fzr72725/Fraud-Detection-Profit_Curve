# Fraud-Detection-Profit_Curve
Over 200,000 transactions made by credit cards in September 2013 by european cardholders were provided
in the data file. The goal is to generate a model to predict which transactions are frauds.

## Motivation
Although incidences of credit card fraud are limited to about 0.1% of all card transactions, they have resulted in huge financial losses as the fraudulent transactions have been large value transactions. In 1999, out of 12 billion transactions made annually, approximately 10 million—or one out of every 1200 transactions—turned out to be fraudulent.

## Data Set
### General Description
- Data Set:
284807 rows, 31 columns

- NaN values: No

- Classes extremely imbalanced, only 0.17% positive labels

### Data Dictionary

|Variable  |  Definition  |  Key|
| --- | --- | --- |
|Time  |  Seconds elapsed between each transaction and the first transaction  |  |
|V1  |  PCA transformation  |  |
|...  | ...   |   |
|V28 |  PCA transformation    |   |
|Amount |  Transaction Amount   |   |
|Class |  Response variable   | 0 = Non-Fraud, 1 = Fraud  |

## Process
### Pre-Processing
- Build pipeline for upcoming modeling
- Scale all features to have the same measurement
- Treat imbalanced classes issue to bring balance

### Model Development
- LogisticRegression
- Random Forest
- GridSearch to tune hyperparameters
- Confusion matrix
- Profit Curves

### Conclusion
| TP  | FP |
| --- | ---|
| 1998|-2  |
| FN  | TN |
| --- | ---|
| 0   | 0  |

With the cost-benefit matrix above, it's a good choice to keep the threshold high so that only the top 25% of the targeted "fraud" transactions are being reacted on.
