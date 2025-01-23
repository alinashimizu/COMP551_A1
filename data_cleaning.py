import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

#TASK 1 - Acquire, Preprocess and Analyze Data
#Dataset 1 heart disease (binary)

#fetch dataset
heart_disease = fetch_ucirepo(id=45)

#data as pd data frame
X = heart_disease.data.features
y = heart_disease.data.targets

#metadata
#print(heart_disease.metadata)

#variable information
#print(heart_disease.variables)

#clean data - check for missing features

#print("Missing feature values:")
#print(X.isna().sum())

#print("Missing target values")
#print(y.isna().sum())

#Option 1 - Drop missing values
X_clean = X.dropna() # we could also do average here and compare
y_clean = y.loc[X_clean.index]
#print(X_clean.shape)
#print(y_clean.shape)

#Option 2 - Fill with average
X_clean2 = X.fillna(X.mean())
#print(X_clean)
y_clean2 = y
#y can stay the same because there are no missing target values


#other cleaning processes to be completed: remove duplicates, outliers, invalid values
print("Check for outliers")
#for some reason i couldnt include ca, but i got the other int features, also do we need to remove outliers?
X_clean.boxplot(column=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
#plt.show()

#categorical variables
#for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
    #print(f"{col} unique values: {X_clean[col].unique()}")

#duplicate check
duplicates = X_clean.duplicated().sum()
#print(duplicates)

#Compute basic statistics
pos_indices = y_clean[y_clean['num'].isin([1, 2, 3, 4])].index
#print(pos_indices)
X_pos = X_clean.loc[pos_indices]
mean_pos = X_pos.mean()
#print(f"Means of features for people with heart disease\n{mean_pos}")
X_pos.describe()

neg_indices = y_clean[y_clean['num'] == 0].index
#print(neg_indices)
X_neg = X_clean.loc[neg_indices]
mean_neg = X_neg.mean()
#print(f"Means of features for people without heart disease\n{mean_neg}")
#X_neg.describe()

#Rank mean squared difference
mean_sqd_diff = (mean_pos - mean_neg) ** 2
msd_ranked = mean_sqd_diff.sort_values(ascending=False)
#print("Mean squared differnce of features for people with and without heart disease")
#print(msd_ranked)

#***DATASET 2 - Penguin Dataset for Multi-Class***
data = pd.read_csv('/Users/alinashimizu/Downloads/penguins_size.csv')
X_penguin = data.drop(columns=['species', 'island'])
y_penguin = data['species']

#Option 1 - Drop na rows 
X_penguin_clean = X_penguin.dropna() 
y_penguin_clean = y.loc[X_clean.index]
#print(y_penguin_clean.head())
#print(X_penguin_clean.head())

#Option 2 - Fill NA rows with average 
X_penguin_clean2 = X_penguin.copy()
numeric_columns = X_penguin.select_dtypes(include=['float64', 'int64']).columns
X_penguin_clean2[numeric_columns] = X_penguin[numeric_columns].fillna(X_penguin[numeric_columns].mean())
#handle sex column differently 
X_penguin_clean2['sex'] = X_penguin['sex'].fillna(X_penguin['sex'].mode()[0])

y_penguin_clean2 = y_penguin
#y can stay the same because there are no missing target values
print(y_penguin_clean2.head())
print(X_penguin_clean2.head())

#other cleaning processes to be completed: remove duplicates, outliers, invalid values
