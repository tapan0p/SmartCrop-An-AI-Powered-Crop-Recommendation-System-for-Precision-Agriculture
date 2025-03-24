import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd
from preprocessing import DataPreprocess,CustomDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression


class Pipeline:
     def __init__(self,model,path,epochs,batch_size,lr):
          super().__init__()
          self.path = path
          self.epochs = epochs
          self.batch_size = batch_size
          self.lr = lr
          self.model = model

     def load_data(self):
        preprocess = DataPreprocess(self.path)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = preprocess.load_data()
          
     def train(self):
            train_dataset = CustomDataset(self.X_train,self.y_train)
            train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True)
            self.criterion = nn.CrossEntropyLoss()

            self.train_loss = []
            self.val_loss = []
            self.val_accuracy = []

            for epoch in range(self.epochs):
                total_loss = 0
                total_sample = 0
                for X,y in train_loader:
                    output = self.model.forward(X)
                    loss = self.criterion(output,y)
                    loss.backward()
                    total_loss += loss.item()*y.size(0)
                    total_sample += y.size(0)
                    with torch.no_grad():
                        self.model.W -= self.lr * self.model.W.grad
                        self.model.b -= self.lr * self.model.b.grad
                        self.model.W.grad.zero_()
                        self.model.b.grad.zero_()
                
                self.train_loss.append(total_loss/total_sample)
                with torch.no_grad():
                    val_loss,val_acc = self.valid(self.X_val,self.y_val)
                    self.val_loss.append(val_loss)
                    self.val_accuracy.append(val_acc)
                    print(f"Epoch {epoch+1} : train_loss = {total_loss:.4f} val_loss = {val_loss:.4f} val_accuracy = {val_acc:.4f}")

     def valid(self,X,y):
        y_cap = self.model.forward(X)
        val_loss = self.criterion(y_cap,y)
        val_preds = y_cap.argmax(dim=1)
        val_acc = accuracy_score(y.numpy(),val_preds)
        return val_loss, val_acc

     def test(self):
        _, test_acc= self.valid(self.X_test,self.y_test)
        print(f"Test accuracy = {test_acc}")
    

     def plot(self,plot_name):
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
        plt.savefig(f'Figures/{plot_name}.png')
        plt.show()

     def save_model(self,model_name):
         torch.save(self.model.state_dict(), f'./Models/{model_name}.pth')
    