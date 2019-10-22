import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pylab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

os.getcwd()
#set path
path  = '/Users/janmichaelaustria/Documents/UNH Analytics Summer 2019/DATA 800/Final Case Analysis'
os.chdir(path)

#import the file
movies = pd.read_excel('movies_final.xlxs.xlsx')

movies.head()

movies.info()

#grab the column namese
names = movies.columns

#potential columns to drop are release_date, title, original_title

#created functions for helping with EDA

#get value counts panda series of categorical variable:
def value_counts(df,column_name):
    count_df = df[column_name].value_counts()
    return(count_df)

#examine spoken lanuage,grab top30
count_spoken = movies['Spoken_Language'].value_counts()
num_factors = len(count_spoken)

#plot distribution of spoken language
plt.figure(figsize=(14,8))
a =sns.countplot(x = 'Spoken_Language',data=movies)
a.set_xticklabels(a.get_xticklabels(), rotation=45)
plt.show()
#there are a lot of categories in the spoken_lang column

#grab the column namese
names = movies.columns

#examine spoken language
count_production = movies['Production_Country'].value_counts()
num_production = len(count_spoken)
plt.figure(figsize=(14,8))
b =sns.countplot(x = 'Production_Country',data=movies)
b.set_xticklabels(b.get_xticklabels(), rotation=25)
plt.show()
#there are a shit to in this category.....


#xamine production company
count_production_compnay = movies['Production_Company'].value_counts()
num_production = len(count_spoken)
#there are way too many levels for this category

#examine genre
count_genre = movies['Genre'].value_counts()
#there are 18 levels for this category, can we reduce to 3?
#i could do a chi squre on this one.....

#examine adule
count_adult = movies['adult'].value_counts()
#binary, good to go
oringal_language = movies['original_language'].value_counts()
#84 levels most of them are english

#no need to examine titles
orig_tit = movies['original_title'].value_counts()
#there are a lot of inidivudal titles, could drop this column

#examine overview
count_overview = movies['overview'].value_counts()
#these arejust the synopsis

#examine likeability
likeability = movies['Likeability'].value_counts()
sns.distplot(movies['Likeability'])
stats.probplot(movies['Likeability'], dist="norm",plot=pylab)
pylab.show()
#likeability is skewed, need to transform
sns.distplot(np.log(movies['Likeability']))
stats.probplot(np.log(movies['Likeability']), dist="norm",plot=pylab)
#take the natural log, but we'd incorrectly predict outliers

#examin vote average
sns.distplot(movies['vote_average'])
stats.probplot(movies['vote_average'], dist="norm",plot=pylab)
pylab.show()
#no need to transform

#examine vote count
sns.distplot(np.log(movies['vote_count']))
stats.probplot(np.log(movies['vote_count']), dist="norm",plot=pylab)
pylab.show()
#log the data

#can't do anything with date
#do year instead, convert to int
year = movies[


sns.scatterplot(x="Likeability", y="revenue", data=movies)
sns.scatterplot(x=movies['Likeability'],y=np.log(movies['revenue']+1))


#columns to use a predictors
columns_to_use = ['Genre','adult','budget','Likeability','vote_average','vote_count','Year',
                  'revenue','Popular','runtime','video']

movies_final = movies[columns_to_use]

#get dummy variables for Genre
dummy_genre = pd.get_dummies(movies['Genre'])

#merge dummy genres with movies_final
movies_final_1 = pd.concat([movies_final,dummy_genre],axis=1)
#drop the main genre column
movies_final = movies_final_1.drop(['War'],axis=1)
######################################################################
#create a linear estimator to predict budget values



#columns with zeros are budget, revenue
#use the other variable to make an estimator that will precit the missing values

#need to drop a genre column to eliminate colinearity, drop war
moives_final = movies_final.drop(['War'],axis=1)

#histogram of budget without 0 values
budget_no_zeros = 

movies_final = movies[columns_to_use]
#transform revenue
movies_final['log_revenue'] = np.log(movies_final['revenue'])




















