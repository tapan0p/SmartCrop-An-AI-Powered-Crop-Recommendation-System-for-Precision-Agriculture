import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd
from preprocessing import DataPreprocess,CustomDataset
from sklearn.metrics import accuracy




class LogisticRegression(nn.Module):
    def __init__(self,input_dim,num_class,lr=0.01):
        super().__init__()
        self.lr = lr
        self.input_dim = input_dim
        self.num_class = num_class 
        # weights
        self.W = torch.randn(input_dim,num_class,dtype=torch.float32,required_grad=True)
        self.b = torch.zeros(num_class,dtype=torch.float32,required_grad=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x):
        return x@self.W+self.b
    
    def load_data(self,path):
        self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test = DataPreprocess(path)

    def train(self,epochs,batch_size):
        if self.X_train is None:
            print("Load the data first")
            return
        train_dataset = CustomDataset(self.X_train,self.y_train)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

        self.loss_list = []
        for epoch in range(epochs):
            for X,y in train_loader:
                output = self.forward(X)
                loss = self.criterion(output,y)
                loss.backward()
                self.loss_list.append(loss.item())

                with torch.no_grad():
                    self.W -= self.lr * self.W.grad()
                    self.b -= self.lr * self.b.grad()
                    self.W.grad.zero_()
                    self.b.grad.zero_()

            
            val_preds = self.forward(self.X_val).argmax(dim=1)

                

                
                

        


        


