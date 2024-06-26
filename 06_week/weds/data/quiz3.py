# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read in data
df = pd.read_csv('./data/nba_rookies.csv')

# Set the index as the name 
df.set_index('Name', inplace = True)

# Convert the 'TARGET_5Yrs' column to 0/1 using 0 for 'No' and 1 for 'Yes'
df['TARGET_5Yrs'] = df['TARGET_5Yrs'].map({'No': 0, 'Yes':1})

# Use ALL columns (except your target) as your X matrix
X = df.drop(columns = 'TARGET_5Yrs')
y = df['TARGET_5Yrs']

# Use a random state of 42 when splitting your data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# Fit a classification 
logreg = LogisticRegression(penalty=None, max_iter=1000)
logreg.fit(X_train, y_train)

# Generate predictions on your test data
preds = logreg.predict(X_test)

# Have one column called 'predictions' 
# which is the predictions from your model on your test data
pred_df = pd.DataFrame(preds, y_test.index)

# in the data folder in this repository
pred_df.to_csv('./data/predictions.csv')