import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from kneed import KneeLocator
import matplotlib.pyplot as plt

import ppscore as pps
from scipy.stats import pearsonr

train = pd.read_csv('train.csv')
data = train.iloc[:, range(87552)]

# Find components:
    
kpca = KernelPCA(kernel = 'poly')
components = []
ks = []
for i in range(len(data)):
    
    row = data.iloc[i, :].values.reshape(512,171).transpose()
    kpca_transform = kpca.fit_transform(row)
    explained_variance = np.var(kpca_transform, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    y = np.cumsum(explained_variance_ratio)
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, curve='concave', direction='increasing')
    ks.append(kn.knee)
    components.append(kpca.alphas_)

optimal_k = round(np.quantile(np.array(ks), q = .95))
plt.hist(ks)

# Create df with the components:

new_data = np.empty(shape=[0, 171*int(optimal_k)])
for comp in components:
    new_data = np.append(new_data, [comp[:, range(int(optimal_k))].reshape(-1)], axis = 0)


new_data= pd.DataFrame(new_data)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
Y = train['tempo']

regr = RandomForestRegressor(n_estimators=10)
scores = cross_val_score(regr, new_data, Y, scoring='neg_root_mean_squared_error', verbose = 3)


# Study Correltion:


from scipy.stats import pearsonr
names = []
for i in range(len(train.iloc[:, 87552:])):
    corr = pearsonr(train.iloc[:, i], train['tempo'])
    if abs(corr[0]) >= 0.1:
        names.append(train.iloc[:, 87552:].columns[i])


special_cor = pps.predictors(train.iloc[:, 87552:], "tempo")
sc=[]
for i in range(len(special_cor)):
    score = special_cor.iloc[i, 1]
    col = special_cor.iloc[i, 0]
    if score > 0:
        sc.append(col)
        
# Create cool DB:

def binding(d,x):
    start = 0
    end = x
    if 171 % x != 0:
        print("Value not valid!")
    else:
        diz = {}
        while end <= d.shape[1]:
            temp = d[d.columns[start:end]]
            mean = temp.mean(axis = 1)
            diz[str(start)+'_'+str(end)] = list(mean)
            start += x
            end += x
        final = pd.DataFrame(diz)
    return final

final_df = binding(train.iloc[:,:87522],57)      
train['genre'] = train['genre'].astype('object')
from sklearn.preprocessing import scale
final_df = pd.DataFrame(scale(final_df))
train_X = pd.concat([final_df,pd.get_dummies(train['genre'])], axis=1)

# Study correlation:

names = []
for i in range(train_X.shape[1]):
    corr = pearsonr(train_X.iloc[:, i], train['tempo'])
    if abs(corr[0]) >= 0.2:
        names.append(train_X.columns[i])

from sklearn.svm import SVR
train_X  = train_X.loc[:, names]
reg = SVR(C=1000)
reg.fit(train_X ,Y)

scores = cross_val_score(reg, train_X, Y, scoring='neg_root_mean_squared_error', verbose = 3)

# Hyperparameter grid

param_grid = {
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'goss'], # I don't add 'dart' mode because it takes a lot (but it usually gives the best result)
    'num_leaves': list(range(30, 150)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 150, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10))
}

# Subsampling (only applicable with 'goss')
subsample_dist = list(np.linspace(0.5, 1, 100))

# Randomly sample parameters for gbm
params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}

params['subsample'] = random.sample(subsample_dist, 1)[0] if params['boosting_type'] != 'goss' else 1.0


# Create a lgb dataset
train_set = lgb.Dataset(train_X, Y)

# Perform cross validation with 10 folds
import lightgbm as lgb
r = lgb.cv(params, train_set, num_boost_round = 1000, nfold = 10, metrics = 'l2', 
           early_stopping_rounds = 100, verbose_eval = True, seed = 50,stratified=False)
from math import sqrt
# Highest score
r_best = sqrt(np.max(r['l2-mean']))

# Standard deviation of best score
r_best_std = r['l2-stdv'][np.argmax(r['l2-mean'])]

# Dataframe to hold cv results
random_results = pd.DataFrame(columns = ['loss', 'params', 'iteration', 'estimators', 'time'],
                       index = list(range(MAX_EVALS)))

print('The maximium MSE on the validation set was {:.5f} with std of {:.5f}.'.format(r_best, r_best_std))
print('The ideal number of iterations was {}.'.format(np.argmax(r['mse-mean']) + 1))

def random_objective(params, iteration, n_folds = N_FOLDS):
    """Random search objective function. Takes in hyperparameters
       and returns a list of results to be saved."""

    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'l2', seed = 50, stratified=False)
    end = timer()
    best_score = np.max(cv_results['l2-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['l2-mean']) + 1)
    
    # Return list of results
    return [loss, params, iteration, n_estimators, end - start]

MAX_EVALS = 500
N_FOLDS = 10

for i in range(MAX_EVALS): # The higher the MAX_WVALS, the better the final score (but more time)
    
    # Randomly sample parameters for gbm
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
    
    print(params)
    
    if params['boosting_type'] == 'goss':
        # Cannot subsample with goss
        params['subsample'] = 1.0
    else:
        # Subsample supported for gdbt and dart
        params['subsample'] = random.sample(subsample_dist, 1)[0]
        
        
    results_list = random_objective(params, i)
    
    # Add results to next row in dataframe
    random_results.loc[i, :] = results_list

# Funzione riduzione:
    
    
