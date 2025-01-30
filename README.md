# Kaggle March Madness
This project trains various models (eg an AdaBoost model) to output March Madness bracket predictions using data provided by Kaggle.

## Dependencies
- Python3
- numpy
- pandas
- sklearn

## Directions
First, go to Kaggle to download the [dataset](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data).
Either save this in a directory named <code>/data/march-madness</code> or edit <code>data-processing.py</code> and <code>model.py</code> to reflect the data directory.

Then run <code>python main.py</code>. Done! Predictions for each game will be saved to <code>[data directory]/MGamePreds.csv</code>, where for each possible game, there is a line with the two teams playing and the probability that the first team wins.

## Instructions to run on RCF

