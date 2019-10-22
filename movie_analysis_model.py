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
from sklearn.model_selection import KFold
import statsmodels.api as sm

os.getcwd()
#set path
path  = '/Users/janmichaelaustria/Documents/UNH Analytics Summer 2019/DATA 800/Final Case Analysis'
os.chdir(path)

#import the file
movies = pd.read_excel('movies_final.xlxs.xlsx')

#predict likeability
predictor_variables = ['adult','budget','vote_average','vote_count','Year',
                  'revenue','Popular','runtime','video']
response_variable = ['Likeability']

#get matrices
X = movies[predictor_variables]
y = movies[response_variable]

#map Popular variable
pop_mapping = {"Other": False, "Popular": True}
X['Popular'] = X['Popular'].map(pop_mapping)

#need to take care of missing values in the runtime column
#impute random values from non missing values
X["runtime"].fillna(np.random.choice(X[X['runtime'].notnull()]['runtime']), inplace =True)

######indices for where revenue is 0
#rev0_ind = X['revenue'] != 0
#X = X[rev0_ind]
#y = y[rev0_ind]

#runtime remoive outliers
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
sns.distplot(X[(X['runtime'].notnull()) & (X['runtime']<300)]['runtime'])
plt.xlabel('Runtime in min')
plt.ylabel('proportion')
plt.title('Runtimes less than 300 mins')
plt.savefig('runtime_300.pdf',format='pdf')

#check likeability


#create linear model
lm = LinearRegression()

#fit and get stats


#cross validation within set to get the best hyperparatmers, 10 fold
cv = KFold(n_splits=10, random_state=42, shuffle=False)
set_of_hyperparameters = []
set_of_intercetps = []
scores_for_model = []
#create iterobject for cv training
for train_index, test_index in cv.split(X):
    #assign training and testing matrices
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    #fit the model
    lm.fit(X_train,y_train)
    #get the hyperparameters, and add to set
    set_of_hyperparameters.append(lm.coef_)
    #get intercet and append
    set_of_intercetps.append(lm.intercept_)
    #get_predictions
    y_pred = lm.predict(X_test)
    #get r2
    r2 = metrics.r2_score(y_test, y_pred)
    #get scores of model and append
    scores_for_model.append(r2)
    
#get average r^2 for the  
np.mean(scores_for_model)
#after cross validatoin the, about 51% of the variability in the predicted
# values is exaplined by the actual values, our model is incorect 51% of the time
#but a lot of data was 0, and random imputation on runtime had to be done

#create model that fits to the whole dataset
lm_likeability = LinearRegression()

#fit the data
lm_likeability.fit(X,y)


#grab coefs and intercepts
intercept = lm_likeability.intercept_
coeffecients = pd.DataFrame(np.transpose(lm_likeability.coef_),X.columns)
coeffecients.columns = ['Hyperparameters']
coeffecients

#import test data
new_movies = pd.read_csv("new_movies.csv")

#grab the right columns from new movies
new_movies_predictors =  new_movies[predictor_variables]
new_movies_Likeability = new_movies['Likeability']

#because sck learn doesn't have a summary out put im being stupid and writing it out!!
lm_likeability.fit(X,y)
params = np.append(lm_likeability.intercept_,lm.coef_)
predictions = lm_likeability.predict(X)

newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
MSE = (np.sum((y-predictions)**2))/(len(newX)-len(newX.columns))

var_b = MSE.values*np.array(np.linalg.inv(np.matrix(np.dot(newX.T,newX),dtype="float")).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]


########get non zero revenue observations
#rev_0_test = new_movies['revenue'] != 0 
#new_movies_predictors =  new_movies_predictors[rev_0_test]
#new_movies_Likeability =  new_movies_Likeability[rev_0_test]

#map popular again
new_movies_predictors['Popular'] = new_movies_predictors['Popular'].map(pop_mapping)

#again there are 7 missing values in runtime
sum(new_movies_predictors['runtime'].isna())
#apply random imputation to NaN
new_movies_predictors["runtime"].fillna(np.random.choice(new_movies_predictors[new_movies_predictors['runtime'].notnull()]['runtime']), inplace =True)

#predict likeability
pred_likeability = lm_likeability.predict(new_movies_predictors)
pred_likeability = pred_likeability[:,0]
df_pred_likeability = pd.DataFrame(pred_likeability)
df_pred_likeability.to_csv('pred_likeability.csv',index=True)

actual_likeability = new_movies_Likeability.values
df_actual_likeability = pd.DataFrame(actual_likeability)
df_actual_likeability.to_csv('actual_likeability.csv',index=True)

#actual_like ability vs pred_likeability 
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
sns.scatterplot(x=actual_likeability,y=pred_likeability)
plt.plot(np.arange(26),'r--')
plt.xlabel('Actual Likeability')
plt.ylabel('Predicted Likeability')
plt.title('Actual Likeability vs Predicted Likeability')
plt.savefig('Likeability_model_scatter.pdf',format='pdf')

#distrbution of residuals
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
sns.distplot(actual_likeability-pred_likeability)
plt.xlabel('Likeability residuals')
plt.ylabel('Proportion')
plt.title('Distribution of Likeability Residuals')
plt.savefig('Likeability_dist_resid.pdf',format='pdf')
#residualt pot
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
sns.scatterplot(x=actual_likeability,y=actual_likeability-pred_likeability)
plt.plot([0,25], [0,0], 'r--')
plt.xlabel('Actual Likeability')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.title('Likeability Residual Plot')
plt.savefig('Likeability_Resid_plot.pdf',format='pdf')

#rsquared, given the data he gave us, this model is ok for predicting likeability less than
metrics.r2_score(actual_likeability,pred_likeability)

print('MAE:', metrics.mean_absolute_error(actual_likeability, pred_likeability))
print('MSE:', metrics.mean_squared_error(actual_likeability, pred_likeability))
print('RMSE:', np.sqrt(metrics.mean_squared_error(actual_likeability, pred_likeability)))


########################################################################
#now model revenue
#import the file
movies = pd.read_excel('movies_final.xlxs.xlsx')

#predict likeability
predictor_variables = ['adult','budget','vote_average','vote_count','Year',
                  'Likeability','Popular','runtime','video']
response_variable = ['revenue']

#get matrices
X = movies[predictor_variables]
y = movies[response_variable]

#map Popular variable
pop_mapping = {"Other": False, "Popular": True}
X['Popular'] = X['Popular'].map(pop_mapping)

#need to take care of missing values in the runtime column
#impute random values from non missing values
X["runtime"].fillna(np.random.choice(X[X['runtime'].notnull()]['runtime']), inplace =True)

lm = LinearRegression()



#cross validation within set to get the best hyperparatmers, 10 fold
cv = KFold(n_splits=10, random_state=42, shuffle=False)
set_of_hyperparameters = []
set_of_intercetps = []
scores_for_model = []
#create iterobject for cv training
for train_index, test_index in cv.split(X):
    #assign training and testing matrices
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    #fit the model
    lm.fit(X_train,y_train)
    #get the hyperparameters, and add to set
    set_of_hyperparameters.append(lm.coef_)
    #get intercet and append
    set_of_intercetps.append(lm.intercept_)
    #get_predictions
    y_pred = lm.predict(X_test)
    #get r2
    r2 = metrics.r2_score(y_test, y_pred)
    #get scores of model and append
    scores_for_model.append(r2)
    
#get average r^2 for the
#the nodel for revenue looks so much better than the model for revenue
np.mean(scores_for_model)

#create model that fits to the whole dataset
lm_revenue = LinearRegression()

#fit the data
lm_revenue.fit(X,y)

#grab coefs and intercepts
intercept = lm_revenue.intercept_
coeffecients = pd.DataFrame(np.transpose(lm_revenue.coef_),X.columns)
coeffecients.columns = ['Hyperparameters']
coeffecients

lm_revenue.fit(X,y)
params = np.append(lm_revenue.intercept_,lm.coef_)
predictions = lm_revenue.predict(X)

newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
MSE = (np.sum((y-predictions)**2))/(len(newX)-len(newX.columns))

var_b = MSE.values*np.array(np.linalg.inv(np.matrix(np.dot(newX.T,newX),dtype="float")).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

#import test data
new_movies = pd.read_csv("new_movies.csv")

#grab the right columns from new movies
new_movies_predictors =  new_movies[predictor_variables]
new_movies_revenue = new_movies['revenue']



#map popular again
new_movies_predictors['Popular'] = new_movies_predictors['Popular'].map(pop_mapping)

#again there are 7 missing values in runtime
sum(new_movies_predictors['runtime'].isna())
#apply random imputation to NaN
new_movies_predictors["runtime"].fillna(np.random.choice(new_movies_predictors[new_movies_predictors['runtime'].notnull()]['runtime']), inplace =True)

#predict revenue
pred_revenue= lm_revenue.predict(new_movies_predictors)
pred_revenue = pred_revenue[:,0]
df_pred_revenue = pd.DataFrame(pred_revenue)
df_pred_revenue.to_csv('pred_revenue.csv',index=True)

actual_revenue = new_movies_revenue.values
df_actual_revenue = pd.DataFrame(actual_revenue)
df_actual_revenue.to_csv('actual_revenue.csv',index=True)

#actual_revenue vs pred_revenue 
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
sns.scatterplot(x=actual_revenue,y=pred_revenue)
plt.plot(np.arange(0,5e+08),'r--')
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual Revenue vs Predicted Revenue")
plt.savefig('Actual Rev vs Pred Rev.pdf',format='pdf')
#distrbution of residuals
sns.distplot(actual_revenue-pred_revenue)
#residualt pot
sns.scatterplot(x=actual_revenue,y=actual_revenue-pred_revenue)
plt.xlabel("Actual Revenue")
plt.ylabel("Residuals")
plt.title("Revenue Residuals")
plt.ylim(-0.5e+08,0.5e+08)
plt.plot([0,4e+08], [0,0], 'r--')
plt.savefig('Residual Revenue.pdf',format='pdf')

metrics.r2_score(actual_revenue,pred_revenue)

print('MAE:', metrics.mean_absolute_error(actual_revenue,pred_revenue))
print('MSE:', metrics.mean_squared_error(actual_revenue,pred_revenue))
print('RMSE:', np.sqrt(metrics.mean_squared_error(actual_revenue,pred_revenue)))

os.getcwd()
#set path
path  = '/Users/janmichaelaustria/Documents/UNH Analytics Summer 2019/DATA 800/Final Case Analysis'
os.chdir(path)
#import the file
movies = pd.read_excel('movies_final.xlxs.xlsx')

#drop some columns
columns_to_drop = ['Spoken_Language',]










