import pandas as pd
from autogluon.tabular import TabularPredictor
from src.data_loader import DataLoader
from src.data_transformer import DataTransformer
from src.prediction_utils import *

#data_path = "./data/2024/"
#file_path = ["march-machine-learning-mania-2024/","MDataFiles_Stage2/"]
data_path = "./data/2025/"
file_path = ["march-machine-learning-mania-2025/"]
CurrentYear=2024

print('Loading Files')
data_loader = DataLoader(data_path, file_path)
files = data_loader.load_csvs()

print('Processing files to training format')
transformer = DataTransformer(files, label="label", currentyear=CurrentYear)

print('Training from data...')
train = transformer.get_train()
test = transformer.get_test()

print('Making predictions...')
predictor = TabularPredictor(label="label", eval_metric="log_loss").fit(train.drop(["TeamID_team1", "TeamID_team2"],axis=1))

y_pred = predictor.predict(test.drop(["TeamID_team1", "TeamID_team2"],axis=1))
probs = predictor.predict_proba(test.drop(["TeamID_team1", "TeamID_team2"],axis=1))

test["pred"] = y_pred
test["proba"] = probs[1]
names = files["MTeams.csv"]
test_with_names = test.merge(names[["TeamID", "TeamName"]], left_on="TeamID_team1", right_on="TeamID", suffixes=("", "_team1"))
test_with_names = test_with_names.merge(names[["TeamID", "TeamName"]], left_on="TeamID_team2", right_on="TeamID", suffixes=("", "_team2"))

pred_df = predict_probs_and_moneylines(test_with_names)

pred_df.to_csv("predictions/predictions.csv")

print('pred_df  ',type(pred_df))
print('Printing out predictions')

#breakpoint()

first_four = get_first_four(files)
pretty_print_matchups(pred_df, first_four, include_moneyline=False)
round_1_matchups = get_round1(files) # excludes round1 which need results of pigtail games
pretty_print_matchups(pred_df, round_1_matchups, include_moneyline=False)
