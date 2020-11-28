#%%
import sys, warnings
warnings.filterwarnings("ignore")
# sys.path.append('./iterativestratification')

import pandas as pd, numpy as np
# import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.metrics import log_loss

import cuml
print('RAPIDS',cuml.__version__)
# %%
train = pd.read_csv('./data/train_features.csv')
print('train shape',train.shape)
test = pd.read_csv('./data/test_features.csv')
print('test shape',test.shape)
targets = pd.read_csv('./data/train_targets_scored.csv')

train.cp_dose = train.cp_dose.map({'D1':-1,'D2':1})
test.cp_dose = test.cp_dose.map({'D1':-1,'D2':1})

train.cp_time = train.cp_time.map({24:-1, 48:0, 72:1})
test.cp_time = test.cp_time.map({24:-1, 48:0, 72:1})

train.cp_type = train.cp_type.map({'trt_cp':-1, 'ctl_vehicle':1})
test.cp_type = test.cp_type.map({'trt_cp':-1, 'ctl_vehicle':1})

train.head()
# %% Double Stratified K Fold
def make_folds(folds = 5, random_state = 0, stratify = True, scored = None):
    
    drug = pd.read_csv('./data/train_drug.csv')
    if scored is None:
        scored = pd.read_csv('./data/train_targets_scored.csv')
    targets = scored.columns[1:]
    scored = scored.merge(drug, on='sig_id', how='left')

    # LOCATE DRUGS
    vc = scored.drug_id.value_counts()
    vc1 = vc.loc[vc<=18].index.sort_values()
    vc2 = vc.loc[vc>18].index.sort_values()

    # STRATIFY DRUGS 18 OR LESS
    dct1 = {}; dct2 = {}
    if stratify:
        skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        skf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    tmp = scored.groupby('drug_id')[targets].mean().loc[vc1]
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)
    
    # STRATIFY DRUGS MORE THAN 18
    if stratify:
        skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        skf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)
    
    # ASSIGN FOLDS
    scored['fold'] = np.nan
    scored['fold'] = scored.drug_id.map(dct1)
    scored.loc[scored.fold.isna(),'fold'] = scored.loc[scored.fold.isna(),'sig_id'].map(dct2)
    scored.fold = scored.fold.astype('int8')
    
    return scored[['sig_id','fold']].copy()
# %%
'''
Each generation, we take the best sqrt(POPULATION) sets of weights and then we randomly combine them in pairs to create a new population of size POPULATION. We also apply MUTATE percentage of mutations (i.e. randomly change some weights). We can think of a set of weights as "DNA".
'''
GENERATIONS = 20
POPULATION = 100 # must be perfect square
PARENTS = int( np.sqrt(POPULATION) )
MUTATE = 0.05

# RANDOMLY CREATE CV
FOLDS = 5; SEED = 42
ff = make_folds(folds=FOLDS, random_state=SEED, stratify=True, scored=targets)
train['fold'] = ff.fold.values
targets['fold'] = ff.fold.values

# INITIALIZE
oof = np.zeros((len(train),206))
dna = np.random.uniform(0,1,(POPULATION,875))**2.0
cvs = np.zeros((POPULATION))

#%%
for jj in range(GENERATIONS):

    # ALL PREVIOUS POPULATION "DNA" AND CV SCORES
    df = pd.DataFrame(data=dna)
    df['cv'] = cvs
    df.sort_values('cv',inplace=True)
    print('Evolving...')

    # GENERATE AND EVALUATE CHILDREN
    for k in range(POPULATION):
        print(k,', ',end='')
        
        # GENERATE CHILD
        if jj!=0:
            parent1 = k//PARENTS; parent2 = k%PARENTS
            TMP = np.random.randint(0,2,875)
            dna[k,] = TMP * df.iloc[parent1,:-1] + (1-TMP) * df.iloc[parent2,:-1]
            x = np.random.uniform(0,1,875)
            IDX = np.where(x<MUTATE)[0]
            dna[k,IDX] = np.random.uniform(0,1,len(IDX))**2.0
        else:
            dna[k,] = df.iloc[k,:-1]
        
        # KNN WEIGHTS
        WGT = dna[k,]
        # WEIGHTS FOR CP_TYPE, CP_TIME, CP_DOSE
        WGT[0]= 100 ; WGT[1] = 12/2; WGT[2] = 5/2

        # KNN KFOLD VALIDATE
        for fold in range(FOLDS):    
            model = cuml.neighbors.KNeighborsClassifier(n_neighbors=1000)
            model.fit( train.loc[ train.fold!=fold, train.columns[1:-1] ].values * WGT,
                       targets.loc[targets.fold!=fold, targets.columns[1:-1] ] )
    
            pp = model.predict_proba( train.loc[ train.fold==fold, train.columns[1:-1] ].values * WGT )
            pp = np.stack( [(1 - pp[x][:,0]) for x in range(len(pp))] ).T
            oof[targets.fold==fold,] = pp
        
        cv_score = log_loss( targets.iloc[:,1:-1].values.flatten(), oof.flatten() )
        cvs[k] = cv_score
#%%

df = pd.DataFrame(data=dna)
df['cv'] = cvs
df = df[['cv']+list(np.arange(0,875))]
df.sort_values('cv',inplace=True)
df.to_csv('dna.csv',index=False)
df.head()


# %% output
WGT = df.iloc[0,1:].values
oof = np.zeros((len(train),206))
preds = np.zeros((len(test),206))

for fold in range(FOLDS):
    print('FOLD %i'%(fold+1), ' ', end='')
    
    model = cuml.neighbors.KNeighborsClassifier(n_neighbors=1000)
    model.fit( train.loc[ train.fold!=fold, train.columns[1:-1] ].values * WGT,
               targets.loc[targets.fold!=fold, targets.columns[1:-1] ] )
    
    pp = model.predict_proba( train.loc[ train.fold==fold, train.columns[1:-1] ].values * WGT )
    pp = np.stack( [(1 - pp[x][:,0]) for x in range(len(pp))] ).T
    oof[targets.fold==fold,] = pp
    
    pp = model.predict_proba( test[test.columns[1:]].values * WGT )
    pp = np.stack( [(1 - pp[x][:,0]) for x in range(len(pp))] ).T
    preds += pp/FOLDS    
    
print()
cv_score = log_loss( targets.iloc[:,1:-1].values.flatten(), oof.flatten() )
print('CV SCORE = %.5f'%cv_score)

#%%
df_oof = targets.copy()
df_oof.iloc[:,1:-1] = oof
df_oof.to_csv('oof.csv',index=False)

oof = np.clip(oof,0.0005,0.999)
oof[train.cp_type==1,] = 0

cv_score = log_loss( targets.iloc[:,1:-1].values.flatten(), oof.flatten() )
print('CV SCORE with CLIPPING = %.5f'%cv_score)

# %%
for fold in range(FOLDS):
    print('FOLD %i'%(fold+1), ' ', end='')
    
    model = cuml.neighbors.KNeighborsClassifier(n_neighbors=1000)
    model.fit( train.loc[ train.fold!=fold, train.columns[1:-1] ].values * WGT,
               targets.loc[targets.fold!=fold, targets.columns[1:-1] ] )
    
    pp = model.predict_proba( train.loc[ train.fold==fold, train.columns[1:-1] ].values * WGT )
    pp = np.stack( [(1 - pp[x][:,0]) for x in range(len(pp))] ).T
    oof[targets.fold==fold,] = pp
    
    pp = model.predict_proba( test[test.columns[1:]].values * WGT )
    pp = np.stack( [(1 - pp[x][:,0]) for x in range(len(pp))] ).T
    preds += pp/FOLDS    
    
print()
cv_score = log_loss( targets.iloc[:,1:-1].values.flatten(), oof.flatten() )
print('CV SCORE = %.5f'%cv_score)
# %%
for jj in range(GENERATIONS):

    # ALL PREVIOUS POPULATION "DNA" AND CV SCORES
    df = pd.DataFrame(data=dna)
    df['cv'] = cvs
    df.sort_values('cv',inplace=True)
    print('Evolving...')

    # GENERATE AND EVALUATE CHILDREN
    for k in range(POPULATION):
        print(k,', ',end='')
        
        # GENERATE CHILD
        if jj!=0:
            parent1 = k//PARENTS; parent2 = k%PARENTS
            TMP = np.random.randint(0,2,875)
            dna[k,] = TMP * df.iloc[parent1,:-1] + (1-TMP) * df.iloc[parent2,:-1]
            x = np.random.uniform(0,1,875)
            IDX = np.where(x<MUTATE)[0]
            dna[k,IDX] = np.random.uniform(0,1,len(IDX))**2.0
        else:
            dna[k,] = df.iloc[k,:-1]
        
        # KNN WEIGHTS
        WGT = dna[k,]
        # WEIGHTS FOR CP_TYPE, CP_TIME, CP_DOSE
        WGT[0]= 100 ; WGT[1] = 12/2; WGT[2] = 5/2

        # KNN KFOLD VALIDATE
        for fold in range(FOLDS):    
            model = cuml.neighbors.KNeighborsClassifier(n_neighbors=1000)
            model.fit( train.loc[ train.fold!=fold, train.columns[1:-1] ].values * WGT,
                       targets.loc[targets.fold!=fold, targets.columns[1:-1] ] )
    
            pp = model.predict_proba( train.loc[ train.fold==fold, train.columns[1:-1] ].values * WGT )
            pp = np.stack( [(1 - pp[x][:,0]) for x in range(len(pp))] ).T
            oof[targets.fold==fold,] = pp
        
        cv_score = log_loss( targets.iloc[:,1:-1].values.flatten(), oof.flatten() )
        cvs[k] = cv_score
# %%

sub = pd.read_csv('../input/lish-moa/sample_submission.csv')
sub.iloc[:,1:] = np.clip(preds,0.0005,0.999)
sub.loc[test.cp_type==1,sub.columns[1:]] = 0
sub.to_csv('submission.csv',index=False)
sub.head()