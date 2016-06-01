#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.cross_validation import train_test_split,cross_val_score,cross_val_predict
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = 5,5
mpl.rcParams.update({'figure.autolayout':True})

all_data = pd.read_csv('clean_data.csv')

# Check and modify type of columns
# Most ML models don't work on categorical string variables
# Create dummies for categorical variables

all_data = all_data.loc[all_data['type'] == 'surface']
cat_columns = ['isif','ibrion','ispin','algo','gga','prec','xc','ismear']

#cat_columns = ['isif','ibrion','ispin','algo','gga','type','prec','xc','ismear']

for col in cat_columns:
    all_data[col] = all_data[col].astype('category')
    dum = pd.get_dummies(all_data[col],prefix='{0}_'.format(col))
    all_data = pd.concat([all_data,dum],axis=1)

all_data = all_data.drop(cat_columns,axis=1)
all_data = all_data.drop(['type'],axis=1)
#all_data.to_csv('data_wdummy.csv',index=False) 

y_data = all_data['elapsed-time']
x_data = all_data.drop(['elapsed-time','memory-used'],axis=1)

print len(x_data.dtypes)
## Feature selection from Random forest regressor 
forest = RandomForestRegressor(n_estimators=1000,random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, random_state=0)
forest.fit(x_train,y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
count = 0
sel_features = []
sel_importances = []
for f in range(x_train.shape[1]):
    if importances[indices[f]] > 0.00 and count < 10:
        count = count + 1
        print x_data.columns.values[indices[f]]
        sel_features.append(x_data.columns.values[indices[f]])
        sel_importances.append(importances[indices[f]])
    #print("%d. feature %s (%f)" % (f + 1, x_data.columns.values[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
fig = plt.gcf()
plt.title("Feature importances")
plt.bar(range(len(sel_features)), sel_importances,
       color="r", align="center")
plt.xticks(range(len(sel_features)),sel_features,rotation='vertical')
plt.yticks(np.arange(0,0.5,0.1))
plt.xlabel('')
plt.xlim([-1, len(sel_features)])
fig.gca().annotate('a) Surface',xy=(4,0.37),fontsize=12)
fig.savefig('surffeatures.png')

