import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch


class DataPreprocess:
    def __init__(self,path,state=42):
        self.path = path
        self.state=state
        self.df = pd.read_csv(path)
        self.features = self.df.columns

    def preprocess(self):
        target = self.features[-1]
        # Encode the target
        encoder = LabelEncoder()
        self.df["encode_label"] = encoder.fit_transform(self.df[target])
        X,y = self.df.drop(columns=[target,"encode_label"]).values,self.df["encode_label"].values
        # Split data in train, test and validation
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=self.state)
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.1,random_state=self.state)
        # Standarise the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        # Convert the data into pytorch tensors
        X_train = torch.tensor(X_train,dtype=torch.float32)
        X_val = torch.tensor(X_val,dtype=torch.float32)
        X_test = torch.tensor(X_test,dtype=torch.float32)
        y_train = torch.tensor(y_train,dtype=torch.long)
        y_val = torch.tensor(y_val,dtype=torch.long)
        y_test = torch.tensor(y_test,dtype=torch.long)
        return X_train,y_train,X_val,y_val,X_test,y_test
        

