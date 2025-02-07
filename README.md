# Kaggle March Madness
This project is used to practice and learn AI/ML techniques, by learning how to optimally train various models to output March Madness bracket predictions using data provided by Kaggle for their March Machine Learning Mania competitions.

## Models and/or ML Frameworks
The models used and their original source code if it exists are

Adaboost (pre-2024 data):   
https://github.com/wdg3/march-madness-2021/tree/main

Autogluon (using XGBoost, tested on 2024 dataset):  
https://github.com/wdg3/marchmadness

## Directions
First, go to Kaggle to download the [dataset](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data).
Either save this in a directory named <code>/data/march-madness</code> or edit <code>data-processing.py</code> and <code>model.py</code> to reflect the data directory.

You will likely have to install all the dependencies below.  It's generally recommended to use the built-in python virtual env, [venv](https://docs.python.org/3/tutorial/venv.html),  or use [UV](https://docs.astral.sh/uv/).

Then run <code>python run_[model].py</code>, where you should replace [model] with the model name. Done! 

The final output should be a set of predictions for each possible March Madness game, and saved in a csv file. For each possible game, there is a line in the csv with the two teams playing and the probability that the first team wins.  The name and 
 
## Instructions for using UV to install python modules on SDCC

First install UV using

`curl -LsSf https://astral.sh/uv/install.sh | sh`

Next set up a UV virtual environment:

`uv init            # create uv project`

`uv venv .venv     # create virtual environment`

Activate the virtual environment:

`source .venv/bin/activate`

Install the dependencies, eg,

`uv pip install autogluon`

Install whatever modules are needed.

If you run out of disk quota on your home disk, you can move the ~/.cache and ~/.local directories to a work disk, and make symlinks back to your home disk:

mv ~/.cache /sphenix/user/[my_username]/
ln -s /sphenix/user/[my_username]/.cache ~/.cache

### Dependencies
- Python3
- numpy
- pandas
- sklearn
- autogluon

