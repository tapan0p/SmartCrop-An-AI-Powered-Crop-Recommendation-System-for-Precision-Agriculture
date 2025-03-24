import pandas as pd
from logistic_regression import LogisticRegression
import argparse
import torch
from pipeline import Pipeline


def main(args):
    model = LogisticRegression(input_dim=7,num_class=22)
    pipeline = Pipeline(model=model,path=args.data_path,epochs=args.epochs,batch_size=args.b,lr=args.lr)
    pipeline.load_data()
    pipeline.train()
    pipeline.plot(plot_name="Logistic_regression_plot")
    pipeline.test()
    pipeline.save_model(model_name="Logistic_regression_model")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression model for crop recomendation.")
    parser.add_argument("--data_path",type=str,default="Dataset\Crop_recommendation.csv",help="Path to dataset.")
    parser.add_argument("--lr",type=float,default=0.01,help="Learning rate")
    parser.add_argument("--b",type=int,default=32,help="Batch size")
    parser.add_argument("--epochs",type=int,default=250,help="Number of epochs to run")

    args = parser.parse_args()
    main(args)


