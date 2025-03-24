import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader,Dataset
import joblib

class CustomDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
    

class DataPreprocess:
    def __init__(self,path,state=42):
        self.state=state
        self.df = pd.read_csv(path)
        self.features = self.df.columns
    
    def make_tensor(self,X,y):
        X = torch.tensor(X,dtype=torch.float32)
        y = torch.tensor(y,dtype=torch.long)
        return X,y

    def load_data(self):
        target = self.features[-1]
        encoder = LabelEncoder()
        self.df["label"]= encoder.fit_transform(self.df["label"])
        joblib.dump(encoder, './Label_Encoder/label_encoder.pkl')
        X,y = self.df.drop(columns=[target]).values,self.df[target].values
        # Split data in train, test and validation
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=self.state)
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.1,random_state=self.state)
        # Standarise the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        # make pytorch tensors
        X_train,y_train = self.make_tensor(X_train,y_train)
        X_val,y_val = self.make_tensor(X_val,y_val)
        X_test,y_test = self.make_tensor(X_test,y_test)

        return X_train,y_train,X_val,y_val,X_test,y_test
        
        
    
        

