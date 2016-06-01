#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.cross_validation import train_test_split,cross_val_score,cross_val_predict
from sklearn import svm
#import matplotlib.pyplot as plt
import sklearn.metrics as sm

all_data = pd.read_csv('clean_data.csv')

#all_data = all_data.loc[all_data['type'] == 'bulk']
#all_data = all_data[(all_data['type'] == 'bulk') &
#                        (all_data['elapsed-time']<1)]
#cat_columns = ['isif','ibrion','ispin','algo','gga','prec','xc','ismear']

cat_columns = ['isif','ibrion','ispin','algo','gga','type','prec','xc','ismear']

for col in cat_columns:
    all_data[col] = all_data[col].astype('category')
    dum = pd.get_dummies(all_data[col],prefix='{0}_'.format(col))
    all_data = pd.concat([all_data,dum],axis=1)

all_data = all_data.drop(cat_columns,axis=1)
#all_data = all_data.drop(['type'],axis=1)

imp_features_all = ['total_atoms',
'nbands',
'total_val_ele',
'ibrion__6.0',
'volume',
'nsw',
'type__surface',
'kp2',
'ibrion__5.0',
'kp1']

imp_features_bulk = ['total_atoms',
'nsw',
'nbands',
'isif__3.0',
'volume',
'total_val_ele',
'prec__Normal',
'prec__High',
'ibrion__2.0',
'isif__2.0']

imp_features_surf = ['total_atoms',
'nbands',
'total_val_ele',
'ibrion__6.0',
'nsw',
'volume',
'ibrion__1.0',
'ibrion__5.0',
'ibrion__2.0',
'kp1']

y_data = all_data['elapsed-time']
x_data = all_data[imp_features_all]


#scores = cross_val_score(clf,x_data,y_data,cv=10,scoring='mean_squared_error')
#rmse_score = (np.sqrt(-scores)).mean()
 
# Decision tree regressors
rfr = RandomForestRegressor(n_estimators=1000,random_state=2)

## Gradient Boosting regression
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = GradientBoostingRegressor(**params)

## Linear regression 
lr = linear_model.LinearRegression()

## Lasso regressor
lasso = linear_model.Lasso(alpha=0.1)

## Elastic net
en = linear_model.ElasticNet(alpha = 0.5, l1_ratio = 0.7)

## SVM
svr = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)

lreg = {'lr':lr,'lasso':lasso,'ENet':en,'SVR':svr, 'RFR': rfr, 'GBR': gbr}

for c in lreg:
    reg = lreg[c]
    #x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
    #                                                    random_state=0)
    #reg.fit(x_train,y_train)
    #y_pred = reg.predict(x_test)
    #print c,
    #mse = sm.mean_squared_error(y_test,y_pred), 
    #print np.sqrt(mse)[0],
    #print sm.r2_score(y_test,y_pred)
    scores = cross_val_score(reg,x_data,y_data,cv=10,scoring='mean_squared_error')
    #rmse_score = (np.sqrt(-scores)).mean()
    rmse_score = (np.sqrt((-scores)/60)).mean()
    r2_scores = cross_val_score(reg,x_data,y_data,cv=10)
    avg_r2_score = r2_scores.mean()
    print '| |{0}|{1}|'.format(c,round(rmse_score,2))
#plt.scatter(np.array(y_test),y_pred)
#plt.show()
#scores = cross_val_score(forest,x_data,y_data,scoring='mean_squared_error')
#print (np.sqrt(-scores)).mean()

# Find optimal parameters for lasso and elasticnet

