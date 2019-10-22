import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pylab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

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
    
#columns to use a predictors
columns_to_use = ['Genre','adult','budget','Likeability','vote_average','vote_count','Year',
                  'revenue','Popular','runtime','video']

movies_final = movies[columns_to_use]

#get dummy variables for Genre
dummy_genre = pd.get_dummies(movies['Genre'])

#merge dummy genres with movies_final
movies_final_1 = pd.concat([movies_final,dummy_genre],axis=1)



#transform revenue, log(value +1)
movies_final_1['log_revenue+1'] = np.log(movies_final_1['revenue']+1)
movies_final_1['log_budget+1'] = np.log(movies_final_1['budget']+1)

movies_corr = movies_final.corr()

movies_corr
plt.show()

#tranform likeability 

############################################################
#dont transform any of the variables, don't include genre
#predict likeability
predictor_variables = ['adult','budget','vote_average','vote_count','Year',
                  'revenue','Popular','runtime','video']
response_variable = ['Likeability']

X = movies_final_1[predictor_variables]
y = movies_final_1[response_variable]


#map Popular variable
pop_mapping = {"Other": False, "Popular": True}
X['Popular'] = X['Popular'].map(pop_mapping)


X.isna().any()
X.isnull().any()
#there is an na in runtime
sum(X['runtime'].isna())
#there are 2174 na values in runteim
proportion_missing = 2174 /34896
#missing data only represents 6.2%
X[X['runtime'].isna()]

#plotting distribution of runtime after dropping NaN values
sns.distplot(X[X['runtime'].notnull()]['runtime'])

#need to take care of missing values in the runtime column
#impute random values from non missing values
X["runtime"].fillna(np.random.choice(X[X['runtime'].notnull()]['runtime']), inplace =True)



#split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#create linear model
lm = LinearRegression()
lm.fit(X_train,y_train)

#get the coefficients and intercept
inntercept = lm.intercept_
coeffecients = pd.DataFrame(np.transpose(lm.coef_),X.columns)
coeffecients.columns = ['Hyperparameters']
coeffecients
intercept = 

#get predictions
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

sns.distplot(y_test-predictions)

plt.scatter(x=y_test,y=predictions)

#get score
lm.score(X_test,y_test)

metrics.r2_score(y_test, predictions)



