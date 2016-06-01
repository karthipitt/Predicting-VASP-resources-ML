#!/usr/bin/env python
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

all_data = pd.read_csv('clean_data.csv')

# Check and modify type of columns
# Most ML models don't work on categorical string variables
# Create dummies for categorical variables
#all_data = all_data[(all_data['type'] == 'bulk') &
#                        (all_data['elapsed-time']<1)]

all_data = all_data.loc[all_data['type'] == 'surface']
cat_columns = ['isif','ibrion','ispin','algo','gga','type','prec','xc','ismear']
for col in cat_columns:
    all_data[col] = all_data[col].astype('category')
    dum = pd.get_dummies(all_data[col],prefix='{0}_'.format(col))
    all_data = pd.concat([all_data,dum],axis=1)

#all_data = all_data.drop(cat_columns,axis=1)

y_data = all_data['elapsed-time']
x_data = all_data.drop(['elapsed-time','memory-used'],axis=1)

sns.set()
#g = sns.FacetGrid(all_data,col='type')
#g.map(sns.boxplot,'elapsed-time')
#g.savefig('boxgrid.png',format='png')
#sns_plot = sns.boxplot(x='type',y='elapsed-time',data=all_data)
fig, ax = plt.subplots()
fig.set_size_inches(4,3.33)
sns_plot=sns.boxplot(x='type',y='elapsed-time',data=all_data, palette="Set2", ax = ax)
sns_plot.set(ylabel='Elapsed time(min)')
fig.tight_layout()
fig.savefig('bpsurf.png',bbox_inches='tight')
#sns_plot = sns.boxplot(x='type',y='elapsed-time',data=all_data, palette="Set2")
#sns_plot.get_figure().savefig('full_boxplot.jpg',format='jpg',width=1,height=1)

#matplotlib.style.use('ggplot')
#pd.options.display.mpl_style = 'default'

#plt.figure()
#s = scatter_matrix(all_data,alpha=0.2,figsize=(6,6),diagonal='kde')
#fig = s.get_figure()
#all_data.boxplot()
#plt.savefig('sm.jpg',format='jpg')
