#import required libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


#Function to find relation between all data parameters 
def scatter_plot (data):
    scatter_matrix_plot = scatter_matrix(dataset, figsize=(20, 20))
    for ax in scatter_matrix_plot.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize = 7, rotation = 45)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 7, rotation = 90)
    return scatter_matrix_plot


#import test data
dataset = pd.read_csv('Dataset.csv')


#Find relation between all data parameters
scatter_matrix_plot = scatter_matrix(dataset, figsize=(20, 20))
for ax in scatter_matrix_plot.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 7, rotation = 45)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 7, rotation = 90)
plt.show()


#Using Pearson Correlation find relation between various parameters
plt.figure(figsize=(10,10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


#Find out co-relation between variables
corr_matrix = dataset.corr()
corr_matrix["Dyno_Torque"].sort_values(ascending=False)








############################################ Younes 

sns.boxplot(dataset.target)



#----------------------------------------------
dataset.corrwith(dataset['target']).plot.bar(figsize = (20,10), grid = True, title = 'Correlation with ETA')






#----------------------------------------------
sns.distplot(dataset.target.transform(np.sqrt))

sns.distplot(final_result.target)



#----------------------------------------------
plt.figure(figsize = (15 , 8))
sns.scatterplot(x = 'target' , y = 'total_time' , data = train)

#---------------------------------------------- 
sns.boxplot(train.target)



#---------------------------------------------- 


dataset.STATUS.value_counts().plot(kind = 'bar')















#define X(input) and y(output)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values




#devide data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)









#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #This uses normal equation to calcuate theta for minimum cost function
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#Checking efficiency of model
print('Variance score for training data: %.2f' % regressor.score(X_train, y_train))
print('Variance score for test test: %.2f' % regressor.score(X_test, y_test))



