import pandas as pd
from logistic_regression import LogisticRegression
import argparse


def main(args):
    df = pd.read_csv(args.data_path)
    input_dim = len(df.columns)-1
    num_columns = len(df['label'].unique())
    model = LogisticRegression(input_dim=input_dim,num_class=num_columns)
    model.load_data(df)
    model.train(epochs=args.epochs,batch_size=args.b,lr=args.lr)
    model.plot()
    model.test()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression model for crop recomendation.")
    parser.add_argument("--data_path",type=str,default="Dataset\Crop_recommendation.csv",help="Path to dataset.")
    parser.add_argument("--lr",type=float,default=0.01,help="Learning rate")
    parser.add_argument("--b",type=int,default=32,help="Batch size")
    parser.add_argument("--epochs",type=int,default=250,help="Number of epochs to run")

    args = parser.parse_args()
    main(args)


