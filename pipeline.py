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
            train_dataset = CustomDataset(self.X_train, self.y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            self.criterion = nn.CrossEntropyLoss()

            self.train_loss = []
            self.val_loss = []
            self.val_accuracy = []

            for epoch in range(self.epochs):
               self.model.train()
               epoch_loss = 0
               total = 0
               
               for X, y in train_loader:
                  optimizer.zero_grad()
                  outputs = self.model(X)
                  loss = self.criterion(outputs, y)
                  loss.backward()
                  optimizer.step()
                  
                  epoch_loss += loss.item() * y.size(0)
                  total += y.size(0)
                
               avg_loss = epoch_loss / total
               self.train_loss.append(avg_loss)
               
               # Validation
               val_loss, val_acc = self.validate()
               self.val_loss.append(val_loss)
               self.val_accuracy.append(val_acc)
               print(f"Epoch {epoch+1}/{self.epochs}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

     def validate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_val)
            loss = self.criterion(outputs, self.y_val)
            preds = outputs.argmax(dim=1)
            acc = accuracy_score(self.y_val.numpy(), preds.numpy())
        return loss.item(), acc

     def test(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            preds = outputs.argmax(dim=1)
            acc = accuracy_score(self.y_test.numpy(), preds.numpy())
        print(f"Test Accuracy: {acc:.4f}")
    

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

     def save_model(self, model_name):
        torch.save(self.model.state_dict(), f'Models/{model_name}.pth')
    