import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
