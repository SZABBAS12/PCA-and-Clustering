# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:11:43 2019

@author: zehra
"""

###############################################################################
#Importing Libraries
###############################################################################

import os
os.chdir('D:\Machine Learning\Final Exam')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis
from sklearn.cluster import KMeans

###############################################################################
#Importing Dataset
###############################################################################

file = 'final.xlsx'
final = pd.read_excel(file)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

###############################################################################
# Fundamental Dataset Exploration
###############################################################################

# Column names

final.columns

# Dimensions of the DataFrame

final.shape

# Information about each variable

final.info()

# Descriptive statistics

desc = final.describe().round(2)

print(desc)

# Viewing the first few rows of the data #

final.head(n = 5)

# To look for missing values #

final.isnull().sum().sum()

""" No missing values
    No negative values
    88 variables of type int64 """
    
###############################################################################
#Renaming column of data frame to match with the survey
###############################################################################

col_dict = {'q1': 'age', 'q2r1': 'iphone', 'q2r2': 'ipod', 'q2r3': 'android',
            'q2r4': 'blackberry', 'q2r5': 'nokia', 'q2r6': 'window', 'q2r7': 
            'hp', 'q2r8': 'tablet', 'q2r9': 'other_smart', 'q2r10': 'no_phone',
            'q4r1': 'music_app', 'q4r2': 'tvcheck_app', 'q4r3': 'entertain_app'
            , 'q4r4': 'tvshow_app', 'q4r5': 'game_app', 'q4r6' : 'social_app',
            'q4r7': 'general_app', 'q4r8': 'shopping_app', 'q4r9':
            'spenew_app', 'q4r10': 'other_app', 'q4r11': 'no_app', 'q11':
            'num_app', 'q12': 'per_free_down', 'q13r1': 'week_facebook',
            'q13r2': 'week_twitter', 'q13r3': 'weeke_myspace', 'q13r4':
            'week_pandora', 'q13r5': 'week_vevo', 'q13r6': 'week_youtube',
            'q13r7': 'week_aol', 'q13r8': 'weel_fm', 'q13r9': 'week_yahoo',
            'q13r10': 'week_imdb', 'q13r11': 'week_linkedin', 'q13r12':
            'week_netfix', 'q48' : 'high_educ', 'q49': 'marital', 'q50r1':
            'no_child', 'q50r2': 'under6_child', 'q50r3': 'btw6_12_child',
            'q50r4': 'btw13_17_child', 'q50r5': 'older18', 'q54' : 'race',
            'q55': 'ethinicity', 'q56': 'annual_income', 'q57': 'gender',
            'q24r1': 'tech_dev', 'q24r2': 'advice_peo', 'q24r3': 'enjoy_pur',
            'q24r4': 'much_tech', 'q24r5': 'enjoy_tech', 'q24r6': 'save_time',
            'q24r7': 'music_per', 'q24r8': 'fav_tvshow', 'q24r9': 'inform',
            'q24r10': 'friend_per', 'q24r11': 'touch_friend', 'q24r12':
            'less_speak', 'q25r1': 'opi_lead', 'q25r2': 'standout', 'q25r3':
            'advice', 'q25r4': 'dec_maker', 'q25r5': 'new_things', 'q25r6':
            'told', 'q25r7': 'control', 'q25r8': 'risk', 'q25r9': 'creative',
            'q25r10': 'optimistic', 'q25r11': 'active', 'q25r12': 'stretched',
            'q26r18': 'luxury', 'q26r3': 'discount', 'q26r4': 'shop', 'q26r5':
            'package_deal', 'q26r6': 'online_shop', 'q26r7': 'design', 'q26r8'
            : 'not_en_app', 'q26r9': 'cool_per', 'q26r10': 'showoff', 'q26r11'
            : 'child_impact', 'q26r12': 'extra_app_fea', 'q26r13': 'spender',
            'q26r14': 'hot', 'q26r15': 'style_brand', 'q26r16': 'impulse',
            'q26r17': 'entertainer_phone'} 

final.columns = [col_dict.get(x, x) for x in final.columns]

###############################################################################
# Step 1: Remove demographic information
###############################################################################

column = ['caseID', 'age', 'high_educ', 'marital', 'no_child', 'under6_child', 
          'btw6_12_child', 'btw13_17_child','older18','race', 'ethinicity',
          'annual_income', 'gender']

final_reduced = final.drop(column, axis=1)

###############################################################################
# Step 2: Scale to get equal variance
###############################################################################

scaler = StandardScaler()


scaler.fit(final_reduced)


X_scaled_reduced = scaler.transform(final_reduced)

###############################################################################
# Step 3: Run PCA without limiting the number of components
###############################################################################

final_pca_reduced = PCA(n_components = None,
                           random_state = 508)

final_pca_reduced.fit(X_scaled_reduced)

X_pca_reduced = final_pca_reduced.transform(X_scaled_reduced)

###############################################################################
# Step 4: Analyze the scree plot to determine how many components to retain
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(final_pca_reduced.n_components_) 


plt.plot(features,
         final_pca_reduced.explained_variance_ratio_, 
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')

plt.title('Reduced final Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()

###############################################################################
# Step 5: Run PCA again based on the desired number of components
###############################################################################

final_pca_reduced = PCA(n_components = 3,
                           random_state = 508)


final_pca_reduced.fit(X_scaled_reduced)

final_pca_dataset = final_pca_reduced.transform(X_scaled_reduced)

# Checking variance ratio of principal components #

ex_variance=np.var(final_pca_dataset,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)

print (ex_variance_ratio) 

# Creating heatmap to see which features mixed up to create components #

plt.matshow(final_pca_reduced.components_,cmap='viridis')
plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize = 10)
plt.colorbar()
plt.xticks(range(len(final_reduced.columns)),
           final_reduced.columns,rotation = 65,ha = 'left')
plt.tight_layout()
plt.show()

###############################################################################
# Step 6: Analyze factor loadings to understand principal components
###############################################################################

factor_loadings_df = pd.DataFrame(pd.np.transpose
                                  (final_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(final_reduced.columns[:])


print(factor_loadings_df.round(2))

factor_loadings_df.columns = ['crazy_app_per', 'chill_pill_per', 'least_int']

factor_loadings_df.to_excel('final_factor_loadings.xlsx')

###############################################################################
# Step 7: Analyze factor strengths per customer
###############################################################################

final_pca_dataset = final_pca_reduced.transform(X_scaled_reduced)

X_pca_df = pd.DataFrame(final_pca_dataset) # This goes into clustering

# Rename factors #

X_pca_df.columns = ['crazy_app_per', 'chill_pill_per', 'least_int']

###############################################################################
# Combining PCA and Clustering!!! (Model Code)
###############################################################################

###############################################################################
# Step 1: Take your transformed dataframe
###############################################################################

print(X_pca_df.head(n = 5))

print(pd.np.var(X_pca_df))

###############################################################################
# Step 2: Scale to get equal variance
###############################################################################

scaler = StandardScaler()

scaler.fit(X_pca_df)

X_pca_clust = scaler.transform(X_pca_df)

X_pca_clust_df = pd.DataFrame(X_pca_clust)

print(pd.np.var(X_pca_clust_df))

X_pca_clust_df.columns = X_pca_df.columns

###############################################################################
# Step 3: Experiment with different numbers of clusters
###############################################################################

customers_k_pca = KMeans(n_clusters = 3,
                         random_state = 508)

customers_k_pca.fit(X_pca_clust_df)

customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})

print(customers_kmeans_pca.iloc[: , 0].value_counts())

###############################################################################
# Step 4: Analyze cluster centers
###############################################################################

centroids_pca = customers_k_pca.cluster_centers_

centroids_pca_df = pd.DataFrame(centroids_pca)

centroids_pca_df.columns = ['crazy_app_per', 'chill_pill_per', 'oldies']


print(centroids_pca_df)

# Sending data to Excel

centroids_pca_df.to_excel('final_pca_centriods.xlsx')

###############################################################################
# Step 5: Analyze cluster memberships
###############################################################################

clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)

###############################################################################
# Step 6: Reattach demographic information
###############################################################################

final_pca_clust_df = pd.concat([final.loc[ : , ['caseID', 'age', 'high_educ', 
                                                'marital', 'no_child', 
                                                'under6_child', 'btw6_12_child'
                                                , 'btw13_17_child','older18',
                                                'race', 'ethinicity',
                                                'annual_income', 'gender']],
                                clst_pca_df],
                                axis = 1)


print(final_pca_clust_df.head(n = 5))

###############################################################################
# Demographic Engineering
###############################################################################

# Creating categories for age #

age_group = {1  : 'under30',
             2  : 'under30',
             3  : 'under30',
             4  : 'under40',
             5  : 'under40',
             6  : 'under40',
             7  : 'under40',
             8  : 'under60',
             9  : 'under60',
             10 : 'above60',
             11 : 'above60'
             }

final_pca_clust_df['age'].replace(age_group, inplace = True)

# Creating categories for annual income #

income_group = {1  : 'under10000',
                2  : 'under30000',
                3  : 'under30000',
                4  : 'under30000',
                5  : 'under70000',
                6  : 'under70000',
                7  : 'under70000',
                8  : 'under70000',
                9  : 'under100000',
                10 : 'under100000',
                11 : 'under100000',
                12 : 'under150000',
                13 : 'under150000',
                14 : 'equal_above150000'
             }

final_pca_clust_df['annual_income'].replace(income_group, inplace = True)

# Creating categories for education status #

edu_group = {   1  : 'school',
                2  : 'school',
                3  : 'college',
                4  : 'college',
                5  : 'post_graduate',
                6  : 'post_graduate'
                }

final_pca_clust_df['high_educ'].replace(edu_group, inplace = True)        

###############################################################################
#Boxplots
###############################################################################

# Factor 'crazy_app_per' with age #

fig, ax = plt.subplots(figsize = (8, 4))

sns.boxplot(x = 'age', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)

""" To see count for each category of age for factor 'crazy_app_per' 
    with respect to each cluster """

pd.pivot_table(final_pca_clust_df, index = 'age', columns = 'cluster',
               values = 'crazy_app_per' , aggfunc = np.count_nonzero)

""" Cluster 1 has most count for under 30 """

# Factor 'crazy_app_per' with gender #

fig, ax = plt.subplots(figsize = (8, 4))

sns.boxplot(x = 'gender', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)

""" To see count for each category of gender for factor 'crazy_app_per' 
    with respect to each cluster """

pd.pivot_table(final_pca_clust_df, index = 'gender', columns = 'cluster',
               values = 'crazy_app_per' , aggfunc = np.count_nonzero)

# Factor 'crazy_app_per' with marital status #

fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'marital', #Demographic
            y = 'crazy_app_per' , #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)

""" To see count for each category of marital status for factor 'crazy_app_per' 
    with respect to each cluster """

pd.pivot_table(final_pca_clust_df, index = 'marital', columns = 'cluster',
               values = 'crazy_app_per' , aggfunc = np.count_nonzero)

# Factor 'crazy_app_per' with education status #

fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'high_educ', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)

""" To see count for each category of education status for factor 
    'crazy_app_per' with respect to each cluster """

pd.pivot_table(final_pca_clust_df, index = 'high_educ', columns = 'cluster',
               values = 'crazy_app_per' , aggfunc = np.count_nonzero)

""" Cluster 1 has most count for college graduates """

# Factor 'crazy_app_per' with race #

fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'race', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)

""" To see count for each category of race for factor 
    'crazy_app_per' with respect to each cluster """

pd.pivot_table(final_pca_clust_df, index = 'race', columns = 'cluster',
               values = 'crazy_app_per' , aggfunc = np.count_nonzero)

# Factor 'crazy_app_per' with ethinicity #

fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'ethinicity', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)

""" To see count for each category of ethinicity for factor 
    'crazy_app_per' with respect to each cluster """

pd.pivot_table(final_pca_clust_df, index = 'ethinicity', columns = 'cluster',
               values = 'crazy_app_per' , aggfunc = np.count_nonzero)

# Factor 'crazy_app_per' with annual income #

fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'annual_income', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)

""" To see count for each category of annual_income for factor 
    'crazy_app_per' with respect to each cluster """

pd.pivot_table(final_pca_clust_df, index = 'annual_income', columns = 'cluster'
               ,values = 'crazy_app_per' , aggfunc = np.count_nonzero)

# Factor 'crazy_app_per' with child status #

fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'no_child', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)

""" To see count for each category of no child for factor 
    'crazy_app_per' with respect to each cluster """

pd.pivot_table(final_pca_clust_df, index = 'no_child', columns = 'cluster'
               ,values = 'crazy_app_per' , aggfunc = np.count_nonzero)

""" Cluster 1 has has no children mostly """


# Other less important Box Plots # 

fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'under6_child', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)

""" To see count for each category of under 6yrs child for factor 
    'crazy_app_per' with respect to each cluster """

pd.pivot_table(final_pca_clust_df, index = 'under6_child', columns = 'cluster'
               ,values = 'crazy_app_per' , aggfunc = np.count_nonzero)


fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'btw6_12_child', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)


fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'btw13_17_child', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)


fig, ax = plt.subplots(figsize = (9, 9))

sns.boxplot(x = 'older18', #Demographic
            y = 'crazy_app_per', #Principal component
            hue = 'cluster',
            data = final_pca_clust_df)



####################################The End####################################
