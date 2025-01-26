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

#Option 1 - Drop missing values
X_clean = X.copy().dropna() # we could also do average here and compare
y_clean = y.copy().loc[X_clean.index]

#Option 2 - Fill with average
X_clean2 = X.copy().fillna(X.mean())
y_clean2 = y.copy() #y can stay the same because there are no missing target values


#other cleaning processes to be completed: remove duplicates, outliers, invalid values
#for some reason i couldnt include ca, but i got the other int features, also do we need to remove outliers?
#X_clean.boxplot(column=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
#plt.show()

#categorical variables
#for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
    #print(f"{col} unique values: {X_clean[col].unique()}")

#duplicate check
duplicates = X_clean.duplicated().sum()
#print(duplicates)

#Compute basic statistics
pos_indices = y_clean[y_clean['num'].isin([1, 2, 3, 4])].index
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

#Remove unecessary features 

#***DATASET 2 - Penguin Dataset for Multi-Class***
data = pd.read_csv('/Users/alinashimizu/Downloads/penguins_size.csv')


#Mapping here to change sex into a numerical variable 
X_penguin = data.drop(columns=['species', 'island'])
X_penguin['sex'] = X_penguin['sex'].map({'MALE': 1, 'FEMALE': 0})
y_penguin = data['species']


#Option 1 - Drop na rows 
X_penguin_clean = X_penguin.copy().dropna() 
y_penguin_clean = y_penguin.copy().loc[X_penguin_clean.index]
#print(y_penguin_clean.head())
#print(X_penguin_clean.head())

#Option 2 - Fill NA rows with average 
X_penguin_clean2 = X_penguin.copy()
numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
X_penguin_clean2[numeric_columns] = X_penguin_clean2[numeric_columns].fillna(X_penguin_clean2[numeric_columns].mean())
#handle sex column differently 
X_penguin_clean2['sex'] = X_penguin_clean2['sex'].fillna(X_penguin_clean2['sex'].mode().iloc[0])

y_penguin_clean2 = y_penguin.copy()
#y can stay the same because there are no missing target values

#other cleaning processes to be completed: remove duplicates, outliers, invalid values

#X_penguin_clean2.boxplot(column=numeric_columns)
#plt.show()
#this was hard to visualize since the numbers are on very different scales, but could go back and normalize 

#checked for duplicates, all False and all values valid 

#Compute means of each feature in each species group: Need to concat back into one df first 
cleaned_penguin_data = pd.concat([X_penguin_clean2, y_penguin_clean2], axis=1)

#helper function for multi-class stats 
def penguin_stats(group):
    all_stats = group.describe(include='all')
    numeric_all = all_stats.loc[['mean', 'std', 'min', 'max'], numeric_columns]
    sex_mode = group['sex'].mode()
    sex_mode = sex_mode.iloc[0]
    stats = numeric_all.transpose()  #needed to transpose to get columns as stats 
    stats['sex_mode'] = sex_mode
    return stats

grouped_stats = cleaned_penguin_data.groupby('species').apply(penguin_stats)

#compute all the squared differences 
squared_differences = {}

#sum of squared differences, should also maybe do pairwise differences
#NORMALIZE
means = grouped_stats.reset_index().pivot(index='species', columns='level_1', values='mean')

for col in numeric_columns:
    diff_sum = 0
    group_means = means[col] 
    for i in range(len(group_means)):
        for j in range(i + 1, len(group_means)):
            diff_sum += (group_means.iloc[i] - group_means.iloc[j]) ** 2
    squared_differences[col] = diff_sum  

rank_features = sorted(squared_differences.items(), key=lambda x: x[1], reverse=True)

for feature, score in rank_features:
    print(f"{feature}: {score}")

#Drop unecessary features 
