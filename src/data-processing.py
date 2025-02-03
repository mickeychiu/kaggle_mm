from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

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
    print(data_path + f + ".csv")
    dfs[f] = pd.read_csv(data_path + f + ".csv", encoding='cp1252')

def generate_dataset(start, end):
    first_season = start
    last_season = end
    print("dfs is type ")
    print(type(dfs))
    seasons = range(first_season, last_season)
    results = dfs["MNCAATourneyDetailedResults"]
    results = results[results["Season"] >= first_season].reset_index(drop=True)
    rankings = dfs["MMasseyOrdinals_thruSeason2024_day128"]
    rankings = rankings[rankings["Season"] >= first_season].reset_index(drop=True)
    systems = set(rankings["SystemName"].unique())
    for season in rankings["Season"].unique():
        systems = systems.intersection(rankings[rankings["Season"] == season]["SystemName"])
    systems = list(systems)
    print(systems)

    # Initialize input X and output Y
    X = pd.DataFrame(np.zeros((2 * len(results), 2 + 2 * len(systems))))
    Y = pd.DataFrame(np.zeros(2 * len(results)))

    print( "shape(X) " )
    print( X.shape )
    print(type(X))
    #print( X.at[0] )

    print("results is type ")
    print(type(results))
    print(results.head())

    for i, row in results.iterrows():
        seed_season = dfs["MNCAATourneySeeds"][dfs["MNCAATourneySeeds"]["Season"] == row["Season"]]
        rank_season = rankings[rankings["Season"] == row["Season"]]
        w_team_rankings = rank_season[rank_season["TeamID"] == row["WTeamID"]]
        l_team_rankings = rank_season[rank_season["TeamID"] == row["LTeamID"]]
        w_seed = int(seed_season[seed_season["TeamID"] == row["WTeamID"]]["Seed"].to_numpy()[0][1:3])
        l_seed = int(seed_season[seed_season["TeamID"] == row["LTeamID"]]["Seed"].to_numpy()[0][1:3])
        x1 = [w_seed, l_seed]
        x2 = [l_seed, w_seed]
        for sys in systems:
            w_team_sys = w_team_rankings[w_team_rankings["SystemName"] == sys]
            l_team_sys = l_team_rankings[l_team_rankings["SystemName"] == sys]
            w = w_team_sys[w_team_sys["RankingDayNum"] == w_team_sys["RankingDayNum"].max()]["OrdinalRank"].to_numpy()
            if w.size == 0:
                w = 50
            else:
                w = w[0]
            l = l_team_sys[l_team_sys["RankingDayNum"] == l_team_sys["RankingDayNum"].max()]["OrdinalRank"].to_numpy()
            if l.size == 0:
                l = 50
            else:
                l = l[0]
            x1 += w, l
            x2 += l, w

        print( len(x1) )
        print( x1 )
        print( 2*i )

        print( type(X) ) 
        X.at[2 * i] = np.array(x1)
        X.at[2 * i + 1] = np.array(x2)
        Y.at[2 * i] = 1
        Y.at[2 * i + 1] = 0

    return X, Y, systems

def get_test(systems):
    preds_frame = pd.read_csv(data_path + SAMPLESUBMIT)
    X_test = pd.DataFrame(np.zeros((len(preds_frame), 2 + 2 * len(systems))))
    rankings = dfs["MMasseyOrdinals_thruSeason2024_day128"]
    rankings = rankings[rankings["Season"] == 2024].reset_index(drop=True)
    seeds = dfs["MNCAATourneySeeds"][dfs["MNCAATourneySeeds"]["Season"] == 2022]
    for i, row in preds_frame.iterrows():
        s = row["ID"]
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

        X_test.at[i] = np.array(x)

    return X_test

X, Y, systems = generate_dataset(2003, 2024)
X.to_csv(data_path + "X.csv")
Y.to_csv(data_path + "Y.csv")

X_test = get_test(systems)
X_test.to_csv(data_path + "X-test.csv")
