import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd
from preprocessing import DataPreprocess,CustomDataset
from sklearn.metrics import accuracy
import matplotlib.pyplot as plt


class LogisticRegression(nn.Module):
    def __init__(self,input_dim,num_class,lr=0.01):
        super().__init__()
        self.lr = lr
        self.input_dim = input_dim
        self.num_class = num_class 
        # weights
        self.W = torch.randn(input_dim,num_class,dtype=torch.float32,requires_grad=True)
        self.b = torch.zeros(num_class,dtype=torch.float32,requires_grad=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x):
        return x@self.W+self.b
    
    def load_data(self,df):
        self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test = DataPreprocess(df)

    def valid(self,X,y):
        y_cap = self.forward(X)
        val_loss = self.criterion(y_cap,y)
        val_preds = y_cap.argmax(dim=1)
        val_acc = accuracy(y.numpy(),val_preds)
        return val_loss, val_acc

    def test(self):
        _, test_acc= self.valid(self.X_test,self.y_test)
        print(f"Test accuracy = {test_acc}")
    

    def train(self,epochs,batch_size):
        if self.X_train is None:
            print("Load the data first")
            return
        train_dataset = CustomDataset(self.X_train,self.y_train)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

        self.train_loss = []
        self.val_loss = []
        self.val_accuracy = []

        for epoch in range(epochs):
            total_loss = 0
            for X,y in train_loader:
                output = self.forward(X)
                loss = self.criterion(output,y)
                loss.backward()
                total_loss += loss.itam()

                with torch.no_grad():
                    self.W -= self.lr * self.W.grad()
                    self.b -= self.lr * self.b.grad()
                    self.W.grad.zero_()
                    self.b.grad.zero_()
            

            with torch.no_grad():
                val_loss,val_acc = self.valid(self.X_val,self.y_val)
                self.val_loss.append(val_loss)
                self.val_accuracy.append(val_acc)
                print(f"Epoch {epoch+1} : train_loss = {total_loss:.4f} val_loss = {val_loss:.4f} val_accuracy = {val_acc:.4f}")

    def plot(self):
        # Plot train_loss and val_loss
        plt.plot(range(1,len(self.train_loss)+1),self.train_loss,label="Training loss",marker = 'o')
        plt.plot(range(1,len(self.val_loss)+1),self.val_loss,label="Validation loss",marker = 'o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training & Validation loss")
        plt.grid()
        plt.show()
                

                
                

        


        


