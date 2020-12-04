#%%
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import gc,os
from time import time
import datetime,random
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# %%
seed = 1127
id_name = 'sig_id'
variance_threshould = 0.7
ncompo_genes = 80
ncompo_cells = 10

def get_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
get_seed(seed)
# %%
data_path = '/home/scao/Documents/harvard-moa/data/'
train = pd.read_csv(data_path + "train_features.csv")
train_target = pd.read_csv(data_path + "train_targets_scored.csv")
test = pd.read_csv(data_path + "test_features.csv")
sub = pd.read_csv(data_path + "sample_submission.csv")
# %%
genes = [col for col in train.columns if col.startswith("g-")]
cells = [col for col in train.columns if col.startswith("c-")]

features = genes + cells
targets = [col for col in train_target if col!='sig_id']
# %%
ori_train = train.copy()
ctl_train = train.loc[train['cp_type']=='ctl_vehicle'].append(test.loc[test['cp_type']=='ctl_vehicle']).reset_index(drop=True)
ctl_train2 = train.loc[train['cp_type']=='ctl_vehicle'].reset_index(drop=True)

ori_test = test.copy()
ctl_test = test.loc[test['cp_type']=='ctl_vehicle'].reset_index(drop=True)
# %%

def Feature(df):
    for col in ['cp_time','cp_dose']:
        tmp = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,tmp],axis=1)
        df.drop([col],axis=1,inplace=True)
    for col in genes:
        if df[col].std() < 1.0:
            df.drop([col],axis=1,inplace=True)
            print(col)
        genes.remove(col)
  
    df[genes+cells] = df[genes+cells]/10.0
    df['gene_gt_0'] = (df[genes]>0.).mean(axis=1)
    df['gene_lt_0'] = (df[genes]<0.).mean(axis=1)
    df['cell_gt_0'] = (df[cells]>0.).mean(axis=1)
    df['cell_lt_0'] = (df[cells]<0.).mean(axis=1)
    df['gene_gt_0.7'] = (df[genes]>0.7).mean(axis=1)
    df['gene_lt_0.7'] = (df[genes]<0.7).mean(axis=1)
    df['cell_gt_0.7'] = (df[cells]>0.7).mean(axis=1)
    df['cell_lt_0.7'] = (df[cells]<0.7).mean(axis=1)
    df['gene_gt_0.9'] = (df[genes]>0.9).mean(axis=1)
    df['gene_lt_0.9'] = (df[genes]<0.9).mean(axis=1)
    df['cell_gt_0.9'] = (df[cells]>0.9).mean(axis=1)
    df['cell_lt_0.9'] = (df[cells]<0.9).mean(axis=1)
    #df['gene_abs_gt_0.7'] = (df[genes].abs()>0.7).mean(axis=1) * 5
    #df['cell_abs_gt_0.7'] = (df[cells].abs()>0.7).mean(axis=1) * 5
    #df = rankGauss(df,genes+cells)
    pca_genes = PCA(n_components = ncompo_genes,
                    random_state = seed).fit_transform(df[genes])
    pca_cells = PCA(n_components = ncompo_cells,
                    random_state = seed).fit_transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)
    pca_cols = [col for col in df.columns if 'pca' in col]
    for col in pca_cols:
        df[col] = df[col] / df[col].abs().max( )
    return df

def Feature1(df):
    '''df['gene_gt_0'] = (df[genes]>0.).mean(axis=1)
    df['gene_lt_0'] = (df[genes]<0.).mean(axis=1)
    df['cell_gt_0'] = (df[cells]>0.).mean(axis=1)
    df['cell_lt_0'] = (df[cells]<0.).mean(axis=1)
    df['gene_gt_0.7'] = (df[genes]>7.).mean(axis=1)
    df['gene_lt_0.7'] = (df[genes]<-7.).mean(axis=1)
    df['cell_gt_0.7'] = (df[cells]>7.).mean(axis=1)
    df['cell_lt_0.7'] = (df[cells]<-7.).mean(axis=1)
    df['gene_gt_0.9'] = (df[genes]>9.).mean(axis=1)
    df['gene_lt_0.9'] = (df[genes]<-9.).mean(axis=1)
    df['cell_gt_0.9'] = (df[cells]>9.).mean(axis=1)
    df['cell_lt_0.9'] = (df[cells]<-9.).mean(axis=1)'''
    for col in tqdm(genes+cells):
        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution='normal')
        transformer.fit(df[:train.shape[0]][col].values.reshape(-1,1))
        df[col] = transformer.transform(df[col].values.reshape(-1,1)).reshape(1,-1)[0]
    pca_genes = PCA(n_components = ncompo_genes,
                    random_state = 42).fit_transform(df[genes])
    pca_cells = PCA(n_components = ncompo_cells,
                    random_state = 42).fit_transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)
    
    nor_var_col = [col for col in df.columns if col in ['sig_id','cp_type','cp_time','cp_dose'] or '_gt_' in col or '_lt_' in col]
    
    var_thresh = VarianceThreshold(0.8)
    var_cols = [col for col in df.columns if col not in ['sig_id','cp_type','cp_time','cp_dose'] and '_gt_' not in col and '_lt_' not in col]
    var_data = var_thresh.fit_transform(df[var_cols])
    df = pd.concat([df[nor_var_col],pd.DataFrame(var_data)],axis=1)
    '''for col in df.columns:
        if col not in nor_var_col:
            df[col] /= df[col].abs().max()'''
    for col in ['cp_time','cp_dose']:
        tmp = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,tmp],axis=1)
        df.drop([col],axis=1,inplace=True)
    return df

def Feature2(df):
    '''df['gene_gt_0'] = (df[genes]>0.).mean(axis=1)
    df['gene_lt_0'] = (df[genes]<0.).mean(axis=1)
    df['cell_gt_0'] = (df[cells]>0.).mean(axis=1)
    df['cell_lt_0'] = (df[cells]<0.).mean(axis=1)
    df['gene_gt_0.7'] = (df[genes]>7.).mean(axis=1)
    df['gene_lt_0.7'] = (df[genes]<-7.).mean(axis=1)
    df['cell_gt_0.7'] = (df[cells]>7.).mean(axis=1)
    df['cell_lt_0.7'] = (df[cells]<-7.).mean(axis=1)
    df['gene_gt_0.9'] = (df[genes]>9.).mean(axis=1)
    df['gene_lt_0.9'] = (df[genes]<-9.).mean(axis=1)
    df['cell_gt_0.9'] = (df[cells]>9.).mean(axis=1)
    df['cell_lt_0.9'] = (df[cells]<-9.).mean(axis=1)'''
    #df['gene_2'] = (df[genes]**2).mean(axis=1)
    #df['cell_2'] = (df[cells]**2).mean(axis=1)
    for col in tqdm(genes+cells):
        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution='normal')
        transformer.fit(df[:train.shape[0]][col].values.reshape(-1,1))
        df[col] = transformer.transform(df[col].values.reshape(-1,1)).reshape(1,-1)[0]
    gene_pca = PCA(n_components = ncompo_genes,
                    random_state = 42).fit(df[genes])
    pca_genes = gene_pca.transform(df[genes])
    cell_pca = PCA(n_components = ncompo_cells,
                    random_state = 42).fit(df[cells])
    pca_cells = cell_pca.transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)
    
    '''nor_var_col = [col for col in df.columns if col in ['sig_id','cp_type','cp_time','cp_dose'] or '_gt_' in col or '_lt_' in col]
    
    var_thresh = VarianceThreshold(0.8)
    var_cols = [col for col in df.columns if col not in ['sig_id','cp_type','cp_time','cp_dose'] and '_gt_' not in col and '_lt_' not in col]
    var_data = var_thresh.fit_transform(df[var_cols])
    df = pd.concat([df[nor_var_col],pd.DataFrame(var_data)],axis=1)'''
    for col in ['cp_time','cp_dose']:
        tmp = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,tmp],axis=1)
        df.drop([col],axis=1,inplace=True)
    return df,transformer,gene_pca,cell_pca

# %%
tt = train.append(test).reset_index(drop=True)
tt,transformer,gene_pca,cell_pca = Feature2(tt)
train = tt[:train.shape[0]]
test = tt[train.shape[0]:].reset_index(drop=True)
# %%
if 1:
    train_target = train_target.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    ori_train = ori_train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train = train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
# %%
class resnetModel(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size,ispretrain=False):
        super(resnetModel, self).__init__()
        self.ispretrain=ispretrain
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        
        self.batch_norm2 = nn.BatchNorm1d(num_features+hidden_size)
        self.dropout2 = nn.Dropout(0.26)
        self.dense2 = nn.utils.weight_norm(nn.Linear(num_features+hidden_size, hidden_size))
        self.batch_norm20 = nn.BatchNorm1d(hidden_size)
        self.dropout20 = nn.Dropout(0.26)
        self.dense20 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        

        self.batch_norm3 = nn.BatchNorm1d(2*hidden_size)
        self.dropout3 = nn.Dropout(0.26)
        self.dense3 = nn.utils.weight_norm(nn.Linear(2*hidden_size, hidden_size))
        self.batch_norm30 = nn.BatchNorm1d(hidden_size)
        self.dropout30 = nn.Dropout(0.26)
        self.dense30 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        

        self.batch_norm4 = nn.BatchNorm1d(2*hidden_size)
        self.dropout4 = nn.Dropout(0.26)
        if self.ispretrain:
          self.dense4 = nn.utils.weight_norm(nn.Linear(2*hidden_size, num_targets))
        else:
          self.dense5 = nn.utils.weight_norm(nn.Linear(2*hidden_size, num_targets))
    
    def forward(self, x):
        x1 = self.batch_norm1(x)
        x1 = F.leaky_relu(self.dense1(x1))
        x = torch.cat([x,x1],1)
        
        x2 = self.batch_norm2(x)
        x2 = self.dropout2(x2)
        x2 = F.leaky_relu(self.dense2(x2))
        x2 = self.batch_norm20(x2)
        x2 = self.dropout20(x2)
        x2 = F.leaky_relu(self.dense20(x2))
        x = torch.cat([x1,x2],1)

        x3 = self.batch_norm3(x)
        x3 = self.dropout3(x3)
        x3 = F.leaky_relu(self.dense3(x3))
        x3 = self.batch_norm30(x3)
        x3 = self.dropout30(x3)
        x3 = F.leaky_relu(self.dense30(x3))
        x3 = torch.cat([x2,x3],1)
        
        x3 = self.batch_norm4(x3)
        x3 = self.dropout4(x3)
        if self.ispretrain:
          x3 = self.dense4(x3)
        else:
          x3 = self.dense5(x3)
        return x3

class resnetModel2(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(resnetModel, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        
        self.batch_norm2 = nn.BatchNorm1d(num_features+hidden_size)
        self.dropout2 = nn.Dropout(0.26)
        self.dense2 = nn.utils.weight_norm(nn.Linear(num_features+hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(2*hidden_size)
        self.dropout3 = nn.Dropout(0.26)
        self.dense3 = nn.utils.weight_norm(nn.Linear(2*hidden_size, hidden_size))
        
        self.batch_norm4 = nn.BatchNorm1d(2*hidden_size)
        self.dropout4 = nn.Dropout(0.26)
        self.dense4 = nn.utils.weight_norm(nn.Linear(2*hidden_size, num_targets))
    
    def forward(self, x):
        x1 = self.batch_norm1(x)
        x1 = F.leaky_relu(self.dense1(x1))
        x = torch.cat([x,x1],1)
        
        x2 = self.batch_norm2(x)
        x2 = self.dropout2(x2)
        x2 = F.leaky_relu(self.dense2(x2))
        x = torch.cat([x1,x2],1)
        
        x3 = self.batch_norm3(x)
        x3 = self.dropout3(x3)
        x3 = self.dense3(x3)
        x3 = torch.cat([x2,x3],1)
        
        x3 = self.batch_norm4(x3)
        x3 = self.dropout4(x3)
        x3 = self.dense4(x3)
        return x3
# %%

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
# %%
def Ctl_augment(train,target,include_test=0):
    if include_test==0:
        ctl_aug=ctl_train2.copy()
    if include_test==1:
        ctl_aug=ctl_train.copy()
    aug_trains = []
    aug_targets = []
    for t in [24,48,72]:
        for d in ['D1','D2']:
            for _ in range(3):
                train1 = train.loc[(train['cp_time']==t)&(train['cp_dose']==d)]
                target1 = target.loc[(train['cp_time']==t)&(train['cp_dose']==d)]
                ctl1 = ctl_aug.loc[(ctl_aug['cp_time']==t)&(ctl_aug['cp_dose']==d)].sample(train1.shape[0],replace=True)
                ctl2 = ctl_aug.loc[(ctl_aug['cp_time']==t)&(ctl_aug['cp_dose']==d)].sample(train1.shape[0],replace=True)
                train1[genes+cells] = train1[genes+cells].values + ctl1[genes+cells].values - ctl2[genes+cells].values
                aug_train = train1.merge(target1,how='left',on='sig_id')
                aug_trains.append(aug_train[['cp_time','cp_dose']+genes+cells])
                aug_targets.append(aug_train[targets])
    df = pd.concat(aug_trains).reset_index(drop=True)
    target = pd.concat(aug_targets).reset_index(drop=True)
    for col in tqdm(genes+cells):
        df[col] = transformer.transform(df[col].values.reshape(-1,1)).reshape(1,-1)[0]
    pca_genes = gene_pca.transform(df[genes])
    pca_cells = cell_pca.transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)
    for col in ['cp_time','cp_dose']:
        tmp = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,tmp],axis=1)
        df.drop([col],axis=1,inplace=True)
    xs = df[train_cols].values
    ys = target[targets]
    #ys_ns = target[targets_ns]
    return xs,ys#,ys_ns
# %%
class MoADataset:
    def __init__(self, features, targets,noise=0.1,val=0):
        self.features = features
        self.targets = targets
        self.noise = noise
        self.val = val
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        sample = self.features[idx, :].copy()
        
        if 0 and np.random.rand()<0.3 and not self.val:
            sample = self.swap_sample(sample)
        
        dct = {
            'x' : torch.tensor(sample, dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    
    def swap_sample(self,sample):
            #print(sample.shape)
            num_samples = self.features.shape[0]
            num_features = self.features.shape[1]
            if len(sample.shape) == 2:
                batch_size = sample.shape[0]
                random_row = np.random.randint(0, num_samples, size=batch_size)
                for i in range(batch_size):
                    random_col = np.random.rand(num_features) < self.noise
                    #print(random_col)
                    sample[i, random_col] = self.features[random_row[i], random_col]
            else:
                batch_size = 1
          
                random_row = np.random.randint(0, num_samples, size=batch_size)
               
            
                random_col = np.random.rand(num_features) < self.noise
                #print(random_col)
                #print(random_col)
       
                sample[ random_col] = self.features[random_row, random_col]
                
            return sample

class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
# %%

device = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 29
trn_loss_=[]
def train_and_predict(features, sub, aug=False, folds=7, seed=1127,lr=1/90.0/3.5*3,weight_decay=1e-5/3):
    dnn_oof = train[['sig_id']]
    oof = train[['sig_id']]
    for t in targets:
        dnn_oof[t] = 0.0
        oof[t] = 0.0
    preds = []
    test_X = test[features].values
    test_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(test_X)), 
                                  batch_size=1024,
                                  shuffle=False)
    
    #aug_test_X = Ctl_augment_test(ori_test)
    #aug_test_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(aug_test_X)),batch_size=1024,shuffle=False)

    mskf = MultilabelStratifiedKFold(n_splits = folds, random_state = seed, shuffle = True)

    for fold, (trn_ind, val_ind) in enumerate(mskf.split(train, train_target)):
        train_X = train.loc[trn_ind,features].values
        #aug_X = aug_df[features].values
        
        train_Y = train_target.loc[trn_ind,targets].values
        train_Y = train_target.loc[trn_ind,targets].values
        #train_Y_ns = train_target.loc[trn_ind,targets_ns].values
        #aug_Y = aug_df[targets].values
        
        valid_X = train.loc[val_ind,features].values
        valid_Y = train_target.loc[val_ind,targets].values
        #valid_Y_ns = train_target.loc[val_ind,targets_ns].values
        if aug:
            # aug_X,aug_Y = Mix_augment(ori_train.loc[trn_ind],train_target.loc[trn_ind,targets])
            aug_X,aug_Y = Ctl_augment(ori_train.loc[val_ind],train_target.loc[val_ind],include_test=1)
            valid_X = np.concatenate([valid_X,aug_X],axis=0)
            valid_Y = np.concatenate([valid_Y,aug_Y],axis=0)
            #train_Y_ns = np.concatenate([train_Y_ns,aug_Y_ns],axis=0)
        #############################################
        #             pretrain model                #
        #############################################
        '''
        train_dataset = MoADataset(train_X, train_Y_ns)
        valid_dataset = MoADataset(valid_X, valid_Y_ns,val=1)
        
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1024, shuffle=False)
    
    
        model = Model(len(features),len(targets_ns),1500,ispretrain=True)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),betas=(0.9, 0.99), lr=1e-3, weight_decay=weight_decay,eps=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=lr, epochs=20, steps_per_epoch=len(train_data_loader))
        
        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing =0.001)
        
        best_valid_metric = 1e9
        not_improve_epochs = 0
        for epoch in range(20):
            # train
            train_loss = 0.0
            train_num = 0
            for data in (train_data_loader):
                optimizer.zero_grad()
                x,y = data['x'].to(device),data['y'].to(device)
                outputs = model(x)
                loss = loss_tr(outputs, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_num += x.shape[0]
                train_loss += (loss.item()*x.shape[0])
                
            train_loss /= train_num
            # eval
            model.eval()
            valid_loss = 0.0
            valid_num = 0
            for data in (valid_data_loader):
                x,y = data['x'].to(device),data['y'].to(device)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                valid_num += x.shape[0]
                valid_loss += (loss.item()*x.shape[0])
            valid_loss /= valid_num
            
            if valid_loss < best_valid_metric:
                torch.save(model.state_dict(),'./model_resnet_fold%s.ckpt'%fold)
                not_improve_epochs = 0 
                best_valid_metric = valid_loss
                print('[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_loss))
                trn_loss_.append(train_loss)
            else:
                not_improve_epochs += 1
                print('[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, NIE +1 ---> %s'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_loss,not_improve_epochs))
                if not_improve_epochs >= 50:
                    break
            model.train()'''
        #############################################
        #         pretrain model finish             #
        #############################################
    
        #train_dataset = MoADataset(train_X, train_Y)
        valid_dataset = MoADataset(valid_X, valid_Y,val=1)
        
        #train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1024, shuffle=False)
    
    
        model = resnetModel(len(features),len(targets),1500)
        model.to(device)
        #state_dict = torch.load('./model_resnet_fold%s.ckpt'%fold, torch.device("cuda" if torch.cuda.is_available() else "cpu") )
        #model.load_state_dict(state_dict,strict=False)
        aug_X,aug_Y = Ctl_augment(ori_train.loc[trn_ind],train_target.loc[trn_ind],include_test=1)
        train_X_ = np.concatenate([train_X,aug_X],axis=0)
        train_Y_ = np.concatenate([train_Y,aug_Y],axis=0)
        train_dataset = MoADataset(train_X_, train_Y_)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(),betas=(0.9, 0.99), lr=1e-3, weight_decay=weight_decay,eps=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=lr, epochs=EPOCHS, steps_per_epoch=len(train_data_loader))
        
        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing =0.001)
        
        best_valid_metric = 1e9
        not_improve_epochs = 0
        for epoch in range(EPOCHS):
            # train
            train_loss = 0.0
            train_num = 0
            for data in (train_data_loader):
                optimizer.zero_grad()
                x,y = data['x'].to(device),data['y'].to(device)
                outputs = model(x)
                loss = loss_tr(outputs, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_num += x.shape[0]
                train_loss += (loss.item()*x.shape[0])
                
            train_loss /= train_num
            # eval
            model.eval()
            valid_loss = 0.0
            valid_num = 0
            for data in (valid_data_loader):
                x,y = data['x'].to(device),data['y'].to(device)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                valid_num += x.shape[0]
                valid_loss += (loss.item()*x.shape[0])
            valid_loss /= valid_num
            t_preds = []
            for data in (test_data_loader):
                x = data[0].to(device)
                with torch.no_grad():
                    outputs = model(x)
                t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
            pred_mean = np.mean(t_preds)
            if valid_loss < best_valid_metric:
                torch.save(model.state_dict(),'./model_resnet_fold%s.ckpt'%fold)
                not_improve_epochs = 0 
                best_valid_metric = valid_loss
                print('[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, pred_mean:%.6f'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_loss,pred_mean))
                trn_loss_.append(train_loss)
            else:
                not_improve_epochs += 1
                print('[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, pred_mean:%.6f, NIE +1 ---> %s'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_loss,pred_mean,not_improve_epochs))
                if not_improve_epochs >= 50:
                    break
            model.train()
            if epoch!=EPOCHS-1:
                aug_X,aug_Y = Ctl_augment(ori_train.loc[trn_ind],train_target.loc[trn_ind],include_test=1)
                train_X_ = np.concatenate([train_X,aug_X],axis=0)
                train_Y_ = np.concatenate([train_Y,aug_Y],axis=0)
                train_dataset = MoADataset(train_X_, train_Y_)
                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

        model = resnetModel(len(features),len(targets),1500)
        model.to(device)
        state_dict = torch.load('./model_resnet_fold%s.ckpt'%fold, torch.device("cuda" if torch.cuda.is_available() else "cpu") )
        model.load_state_dict(state_dict)
        model.eval()
        valid_preds = []
        for data in tqdm(valid_data_loader):
            x,y = data['x'].to(device),data['y'].to(device)
            with torch.no_grad():
                outputs = model(x)
            valid_preds.extend(list(outputs.cpu().detach().numpy()))
        dnn_oof.loc[val_ind,targets] = 1 / (1+np.exp(-np.array(valid_preds)))
        t_preds = []
        for data in tqdm(test_data_loader):
            x = data[0].to(device)
            with torch.no_grad():
                outputs = model(x)
            t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
        print(np.mean(t_preds))
        preds.append(t_preds)
    sub[targets] = np.array(preds).mean(axis=0)
    return dnn_oof,oof,sub
# %%
train_cols = [col for col in train.columns if col not in ['sig_id','cp_type','cs','most_cs_ind','most_cs_same_target']]
#train_cols = sorted(train_cols)
len(train_cols)
# %%
dnn_oof,oof,sub = train_and_predict(train_cols,sub.copy(),aug=True,seed=817119,lr=1/90.0/3.5*4,weight_decay=1e-5/4)
# %%
def Metric(labels,preds):
    labels = np.array(labels)
    preds = np.array(preds)
    metric = 0
    for i in range(labels.shape[1]):
        metric += (-np.mean(labels[:,i]*np.log(np.maximum(preds[:,i],1e-15))+(1-labels[:,i])*np.log(np.maximum(1-preds[:,i],1e-15))))
    return metric/labels.shape[1]

valid_metric = Metric(train_target[targets].values,dnn_oof[targets].values)
print(valid_metric)

#%%
sub.loc[test['cp_type']=='ctl_vehicle',targets] = 0.0
dnn_oof.to_csv('./oof.csv',index=False)
sub.to_csv('./submission.csv',index=False)
print(train_target[targets].mean().mean(), sub[targets].mean().mean(), dnn_oof[targets].mean().mean())