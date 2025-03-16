import pandas as pd
from autogluon.tabular import TabularPredictor
from src.data_loader import DataLoader
from src.data_transformer import DataTransformer
from src.prediction_utils import *
from src.custom_metrics import *


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

#breakpoint()

print("Making men's tournament predictions...")

#predictor = TabularPredictor(label="label", eval_metric="log_loss").fit(train.drop(["TeamID_team1", "TeamID_team2"],axis=1))
predictor = TabularPredictor(label="label", eval_metric=ag_brier_score).fit(train.drop(["TeamID_team1", "TeamID_team2"],axis=1))
#predictor = TabularPredictor.load("/Users/chiu/Documents/AIML/kaggle_mm/AutogluonModels/ag-20250302_212934")
print(predictor.leaderboard())

y_pred = predictor.predict(test.drop(["TeamID_team1", "TeamID_team2"],axis=1))
probs = predictor.predict_proba(test.drop(["TeamID_team1", "TeamID_team2"],axis=1))

test["pred"] = y_pred
test["Pred"] = probs[1]
test["ID"] = "2025_" + test["TeamID_team1"].astype(str) + "_" + test["TeamID_team2"].astype(str)

#test.to_csv("test.csv")

test4submit = test[['ID', 'Pred']].copy().set_index('ID')

submission = files['SampleSubmissionStage2.csv'].set_index('ID')
submission.update(test4submit)
submission.to_csv("submission.csv")

names = files["MTeams.csv"]
test_with_names = test.merge(names[["TeamID", "TeamName"]], left_on="TeamID_team1", right_on="TeamID", suffixes=("", "_team1"))
test_with_names = test_with_names.merge(names[["TeamID", "TeamName"]], left_on="TeamID_team2", right_on="TeamID", suffixes=("", "_team2"))

#breakpoint()

pred_df = predict_probs_and_moneylines(test_with_names)

pred_df.to_csv("predictions/predictions.csv")

#print('Printing out predictions')

#first_four = get_first_four(files)
#pretty_print_matchups(pred_df, first_four, include_moneyline=False)
#round_1_matchups = get_round1(files) # excludes round1 which need results of pigtail games
#pretty_print_matchups(pred_df, round_1_matchups, include_moneyline=False)


print("Making women's tournament predictions...")


wtransformer = DataTransformer(files, label="label", currentyear=CurrentYear, type='women')
wtrain = wtransformer.get_train()
wtest = wtransformer.get_test()
wpredictor = TabularPredictor(label="label", eval_metric=ag_brier_score).fit(wtrain.drop(["T1_TeamID", "T2_TeamID"],axis=1))
#wpredictor = TabularPredictor.load("/Users/chiu/Documents/AIML/kaggle_mm/AutogluonModels/ag-20250316_002757")
print(wpredictor.leaderboard())

w_y_pred = wpredictor.predict(wtest.drop(["T1_TeamID", "T2_TeamID"],axis=1))
wprobs = wpredictor.predict_proba(wtest.drop(["T1_TeamID", "T2_TeamID"],axis=1))

wtest["pred"] = w_y_pred
wtest["Pred"] = wprobs[1]
wtest["ID"] = "2025_" + wtest["T1_TeamID"].astype(str) + "_" + wtest["T2_TeamID"].astype(str)

wtest.to_csv("wtest.csv")

wtest4submit = wtest[['ID', 'Pred']].copy().set_index('ID')
wtest4submit.to_csv('wtest4submit.csv')
submission.update(wtest4submit)
submission.to_csv("submission.csv")
