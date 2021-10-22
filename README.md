# Supervised Machine Learning - Predicting Credit Risk

The task was to build a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. 

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

I used this data to create machine learning models to classify the risk level of given loans. Specifically, I compared the Logistic Regression model and Random Forest Classifier.

## Retrieve the data

In the `Generator` folder in `Resources`, there is a [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) notebook that will download data from LendingClub and output two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

I used an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric

I created a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, I created a testing set from the 2020 loans, also using `pd.get_dummies()`. 
Note, there were categories in the 2019 loans that do not exist in the testing set (if you fit a model to the training set and try to score it on the testing set as is, you will get an error. I used code to fill in the missing categories in the testing set). 

## Consider the models

I created and compared two models on this data: a logistic regression, and a random forests classifier. Before I created, fitted, and scored the models, I made a prediction as to which model would perform better (I described why I thought one model would be better than other in the [Credit Risk Evaluator Notebook](/CreditRiskEvaluator.ipynb)). 
## Fit a LogisticRegression model and RandomForestClassifier model

I created a LogisticRegression model, fitted it to the data, and printed the model's score. Then, did the same for a RandomForestClassifier.

## Revisit the Preprocessing: Scale the data

The data going into these models was never scaled, an important step in preprocessing. I used `StandardScaler` to scale the training and testing sets. Before re-fitting the LogisticRegression and RandomForestClassifier models on the scaled data, I made another prediction about how I thought scaling would affect the accuracy of the models. 

Fitted and scored the LogisticRegression and RandomForestClassifier models on the scaled data. How did the model scores compare to each other, and to the previous results on unscaled data? How did this compare to my prediction? You can find my thoughts in the [Credit Risk Evaluator Notebook](/CreditRiskEvaluator.ipynb).

### References

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/](https://resources.lendingclub.com/)