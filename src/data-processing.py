from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

# Settings
CurrentYear = 2024

# Get data file names
data_path = "./data/2024/"
#data_path = "./march-madness-2021/data_2022/march_madness/"
#data_path = "./data_2022/MDataFiles_Stage2/"
#data_path = "./warmup_2023/data/"
listdir(data_path)

SAMPLESUBMIT = "sample_submission.csv"
#SAMPLESUBMIT = "MSampleSubmissionStage2.csv"
#SAMPLESUBMIT = "SampleSubmissionWarmup.csv"

files = [f[:-4] for f in listdir(data_path) if isfile(join(data_path, f))]
print(files)

# Read data files into DataFrames
dfs = {}
for f in files:
    # skip hidden files
    if f.startswith('.'):
        continue
    print(data_path + f + ".csv")
    dfs[f] = pd.read_csv(data_path + f + ".csv", encoding='cp1252')

# Generate training datasets
def generate_dataset(start, end):

    # seasons in dataset
    first_season = start
    last_season = end
    seasons = range(first_season, last_season)

    # detailed results from those seasons
    results = dfs["MNCAATourneyDetailedResults"]
    results = results[results["Season"] >= first_season].reset_index(drop=True)
    print("results\n",results)

    # rankings during those seasons
    rankings = dfs["MMasseyOrdinals_thruSeason2024_day128"]
    rankings = rankings[rankings["Season"] >= first_season].reset_index(drop=True)
    print("rankings\n",rankings)

    # ranking systems that are common to all seasons in training set
    systems = set(rankings["SystemName"].unique())
    for season in rankings["Season"].unique():
        systems = systems.intersection(rankings[rankings["Season"] == season]["SystemName"])
    systems = list(systems)
    print('systems\n', systems)

    # Initialize input X and output Y
    X = pd.DataFrame(np.zeros((2 * len(results), 2 + 2 * len(systems))))
    Y = pd.DataFrame(np.zeros(2 * len(results)))

    #print( "shape(X) ", X.shape ) 

    for i, row in results.iterrows():

        # Get all the seeds and rankings for the season in this row from the results
        seed_season = dfs["MNCAATourneySeeds"][dfs["MNCAATourneySeeds"]["Season"] == row["Season"]]
        rank_season = rankings[rankings["Season"] == row["Season"]]

        # Get the W and L team rankings
        w_team_rankings = rank_season[rank_season["TeamID"] == row["WTeamID"]]
        l_team_rankings = rank_season[rank_season["TeamID"] == row["LTeamID"]]

        # Get the W and L team seeds
        w_seed = int(seed_season[seed_season["TeamID"] == row["WTeamID"]]["Seed"].to_numpy()[0][1:3])
        l_seed = int(seed_season[seed_season["TeamID"] == row["LTeamID"]]["Seed"].to_numpy()[0][1:3])

        # make the flipped seeds
        x1 = [w_seed, l_seed]
        x2 = [l_seed, w_seed]

        # get the rankings for each system system
        for sys in systems:

            w_team_sys = w_team_rankings[w_team_rankings["SystemName"] == sys]
            l_team_sys = l_team_rankings[l_team_rankings["SystemName"] == sys]

            w = w_team_sys[w_team_sys["RankingDayNum"] == w_team_sys["RankingDayNum"].max()]["OrdinalRank"].to_numpy()
            if w.size == 0:
                w = 50  # if there's no ranking, default to 50
            else:
                w = w[0]
            l = l_team_sys[l_team_sys["RankingDayNum"] == l_team_sys["RankingDayNum"].max()]["OrdinalRank"].to_numpy()
            if l.size == 0:
                l = 50
            else:
                l = l[0]
            x1 += w, l
            x2 += l, w

        #print( len(x1) )
        #print( x1 )
        #print( 2*i )
        #print( "X is type ", type(X) ) 
        #print( i )
        #X.head()

        #print( 'x1: ', np.array(x1) )
        #print( 'x2: ', np.array(x2) )

        X.loc[2 * i] = np.array(x1)
        X.loc[2 * i + 1] = np.array(x2)
        Y.loc[2 * i] = 1
        Y.loc[2 * i + 1] = 0

    return X, Y, systems

def get_test(systems):
    preds_frame = pd.read_csv(data_path + SAMPLESUBMIT)
    X_test = pd.DataFrame(np.zeros((len(preds_frame), 2 + 2 * len(systems))))
    rankings = dfs["MMasseyOrdinals_thruSeason2024_day128"]
    rankings = rankings[rankings["Season"] == CurrentYear].reset_index(drop=True)
    seeds = dfs["MNCAATourneySeeds"][dfs["MNCAATourneySeeds"]["Season"] == CurrentYear]
    print(seeds.head())
    for i, row in preds_frame.iterrows():
        s = row["RowId"]
        arr = s.split('_')
        team1 = arr[1]
        team2 = arr[2]
        team1_rankings = rankings[rankings["TeamID"] == int(team1)]
        team2_rankings = rankings[rankings["TeamID"] == int(team2)]
        seed1 = int(seeds[seeds["TeamID"] == int(team1)]["Seed"].to_numpy()[0][1:3])
        seed2 = int(seeds[seeds["TeamID"] == int(team2)]["Seed"].to_numpy()[0][1:3])
        x = [seed1, seed2]
        for sys in systems:
            sys1 = team1_rankings[team1_rankings["SystemName"] == sys]
            sys2 = team2_rankings[team2_rankings["SystemName"] == sys]
            r1 = sys1[sys1["RankingDayNum"] == sys1["RankingDayNum"].max()]["OrdinalRank"].to_numpy()
            if r1.size == 0:
                r1 = 50
            else:
                r1 = r1[0]
            r2 = sys2[sys2["RankingDayNum"] == sys2["RankingDayNum"].max()]["OrdinalRank"].to_numpy()
            if r2.size == 0:
                r2 = 50
            else:
                r2 = r2[0]

            x += r1, r2

        X_test.loc[i] = np.array(x)

    return X_test

X, Y, systems = generate_dataset(2003, CurrentYear)
#X.to_csv(data_path + "X.csv")
#Y.to_csv(data_path + "Y.csv")

X_test = get_test(systems)
X_test.to_csv(data_path + "X-test.csv")
