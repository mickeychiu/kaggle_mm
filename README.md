# Kaggle March Madness
This project is used to practice and learn AI/ML techniques, by learning how to optimally train various models to output March Madness bracket predictions using data provided by Kaggle for their March Machine Learning Mania competitions.

## Models and/or ML Frameworks
The models used and their original source code if it exists are

Adaboost:   
https://github.com/wdg3/march-madness-2021/tree/main

Autogluon (using XGBoost):
https://github.com/wdg3/marchmadness


### Dependencies
- Python3
- numpy
- pandas
- sklearn
- autogluon

## Directions
First, go to Kaggle to download the [dataset](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data).
Either save this in a directory named <code>/data/march-madness</code> or edit <code>data-processing.py</code> and <code>model.py</code> to reflect the data directory.

Then run <code>python run_[model].py</code>, where you should replace [model] with the model name.  Done! Predictions for each game will be saved to <code>[data directory]/MGamePreds.csv</code>, where for each possible game, there is a line with the two teams playing and the probability that the first team wins.

## Instructions to run on RCF

This code was derived from 
https://github.com/wdg3/march-madness-2021/tree/main
