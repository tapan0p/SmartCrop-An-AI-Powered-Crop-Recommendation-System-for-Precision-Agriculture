import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd
from preprocessing import DataPreprocess,CustomDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class LogisticRegression(nn.Module):
    def __init__(self,input_dim,num_class):
        super().__init__()
        self.input_dim = input_dim
        self.num_class = num_class 
        # weights
        self.W = torch.randn(input_dim,num_class,dtype=torch.float32,requires_grad=True)
        self.b = torch.zeros(num_class,dtype=torch.float32,requires_grad=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x):
        return x@self.W+self.b
    
    def load_data(self, df):
        preprocess = DataPreprocess(df)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = preprocess.load_data()


    def valid(self,X,y):
        y_cap = self.forward(X)
        val_loss = self.criterion(y_cap,y)
        val_preds = y_cap.argmax(dim=1)
        val_acc = accuracy_score(y.numpy(),val_preds)
        return val_loss, val_acc

    def test(self):
        _, test_acc= self.valid(self.X_test,self.y_test)
        print(f"Test accuracy = {test_acc}")
    

    def train(self,epochs,batch_size,lr):
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
            total_sample = 0
            for X,y in train_loader:
                output = self.forward(X)
                loss = self.criterion(output,y)
                loss.backward()
                total_loss += loss.item()*y.size(0)
                total_sample += y.size(0)
                with torch.no_grad():
                    self.W -= lr * self.W.grad
                    self.b -= lr * self.b.grad
                    self.W.grad.zero_()
                    self.b.grad.zero_()
            
            self.train_loss.append(total_loss/total_sample)
            with torch.no_grad():
                val_loss,val_acc = self.valid(self.X_val,self.y_val)
                self.val_loss.append(val_loss)
                self.val_accuracy.append(val_acc)
                print(f"Epoch {epoch+1} : train_loss = {total_loss:.4f} val_loss = {val_loss:.4f} val_accuracy = {val_acc:.4f}")

    def plot(self):
        epochs = range(1, len(self.train_loss) + 1)

        # Plot training & validation loss
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)  # First subplot
        plt.plot(epochs, self.train_loss, label="Training Loss", marker='o',markersize=1)
        plt.plot(epochs, self.val_loss, label="Validation Loss", marker='o',markersize=1)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid()

        # Plot validation accuracy
        plt.subplot(1, 2, 2)  # Second subplot
        plt.plot(epochs, self.val_accuracy, label="Validation Accuracy", marker='o', color='g',markersize=1)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()