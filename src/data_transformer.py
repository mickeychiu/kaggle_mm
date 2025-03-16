import numpy as np
import pandas as pd
import glob
import re
from itertools import combinations

verbosity = 0

class DataTransformer:
    def __init__(self, dataframes, label, currentyear=2025, type='men'):
        print('DataTransformer')
        self.currentyear = currentyear
        self.type = type
        print(self.currentyear)
        self.dfs = dataframes
        #print(self.dfs)

        if type == 'men':
            self.rankings = self.transform_rankings(self.dfs)

#           self.addfeatures = self.add_more_features(self.dfs)

            self.train = self.rankings[self.rankings["Season"] != currentyear]
            self.test = self.rankings[self.rankings["Season"] == currentyear]

            self.train = self.add_labels(self.train, label)
            self.test = self.process_test(self.test)
        else:
            #should add self.dfs['WNCAATourneyDetailedResults.csv'] here
            self.season_stats = self.get_statistics(self.dfs['WRegularSeasonDetailedResults.csv'])
            self.season_stats.to_csv('wseasonstats.csv')

            self.train = self.get_train_seasonstats(self.dfs['WRegularSeasonDetailedResults.csv'])

            tourneyseedsfile = str(self.currentyear) + '_tourney_seeds.csv'
            print(f'Making all combos from {tourneyseedsfile}')
            combos2predict = self.get_tourneycombos(self.dfs[tourneyseedsfile])
            combos2predict.to_csv("combos2predict.csv")

            self.test = self.get_predict_seasonstats(combos2predict)



    def transform_rankings(self, data):
        print("Transforming historical rankings data...", end="")
        #ordinals = data[masseyfile[0]]
        #ordinals = data['MMasseyOrdinals_thruSeason2024_day128.csv']
        ordinals = data['MMasseyOrdinals.csv']
        seasons = np.unique(ordinals["Season"])
        systems = np.unique(ordinals["SystemName"])
        final_ordinals = pd.DataFrame()
        all_finals = []
        for season in seasons:
            season_frame = ordinals.loc[ordinals["Season"] == season]
            for system in systems:
                season_system_frame = season_frame.loc[season_frame["SystemName"] == system]
                if not (season_system_frame.empty):
                    maximum_day = max(season_system_frame["RankingDayNum"])
                    season_system_finals = season_system_frame.loc[season_system_frame["RankingDayNum"] == maximum_day]
                    all_finals.append(season_system_finals)
        final_ordinals = pd.concat(all_finals, axis = 0)
        system_dfs = []
        for system in systems:
            system_dfs.append(final_ordinals.loc[final_ordinals["SystemName"] == system].drop(["SystemName", "RankingDayNum"], axis=1).rename(columns={"OrdinalRank": system}))
        joint_ordinals = system_dfs[0]
        for df in system_dfs[1:]:
            joint_ordinals = joint_ordinals.merge(df, how="outer", on=["Season", "TeamID"])
        seeds = data["MNCAATourneySeeds.csv"]
        confs = data["MTeamConferences.csv"]
        with_seeds = joint_ordinals.merge(seeds, how="inner", on=["Season", "TeamID"])
        with_conf = with_seeds.merge(confs, on=["Season", "TeamID"])
        data = with_conf
        data["Seed"] = data["Seed"].map(lambda x: int(x) if len(x) == 2 else (int(x[1:3]) if len(x) == 4 else int(x[1:])))

        print("done.")
        return data

    def get_statistics(self, df):
        '''Get statistics for each team foe each season
        '''

        df['WDiff'] = df['WScore'] - df['LScore']
        df['LDiff'] = df['LScore'] - df['WScore']


        # Create df with detailed stats for each team (must combine stats from wins and losses)
        wcolumns = ["Season"] + [col for col in df.columns if col.startswith("W")]
        wcolumns.remove('WLoc')   # remove the location
        print(wcolumns)
        df_w = df[wcolumns]

        lcolumns = ["Season"] + [col for col in df.columns if col.startswith("L")]
        print(lcolumns)
        df_l = df[lcolumns]

        df_w = df_w.rename(columns=lambda x: x.lstrip("W") if x != "Season" else x)
        df_l = df_l.rename(columns=lambda x: x.lstrip("L") if x != "Season" else x)

        # Display the first few rows
        #print(df_w.head())
        #print(df_l.head())

        season_data = pd.concat([df_w,df_l]).reset_index()
        boxscore_cols = season_data.columns.drop(['Season','TeamID'])
        season_statistics = season_data.groupby(["Season", 'TeamID'])[boxscore_cols].agg(['mean'])
        season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]

        #print(season_statistics)
    
        return season_statistics


    def get_train_seasonstats(self, df):

        cols2keep = ['Season','WTeamID','LTeamID']
        train = df[cols2keep].copy()

        # Add the stats for the winning team
        train.rename(columns={'WTeamID':'TeamID'}, inplace=True)
        print(train.columns)
        season_stats = self.season_stats.copy()
        season_stats.columns = ['T1_' + x for x in list(season_stats.columns)]

        train = pd.merge(train, season_stats, on = ['Season', 'TeamID'], how = 'left')
        train.rename(columns={'TeamID':'T1_TeamID'}, inplace=True)

        # Add the stats for the losing team
        train.rename(columns={'LTeamID':'TeamID'}, inplace=True)
        season_stats.columns = [x.replace('T1_','T2_') for x in list(season_stats.columns)]

        train = pd.merge(train, season_stats, on = ['Season', 'TeamID'], how = 'left')
        train.rename(columns={'TeamID':'T2_TeamID'}, inplace=True)

        train_swap = train.copy()
        train_swap.columns = [x.replace('T1_','W_') for x in list(train_swap.columns)]
        train_swap.columns = [x.replace('T2_','T1_') for x in list(train_swap.columns)]
        train_swap.columns = [x.replace('W_','T2_') for x in list(train_swap.columns)]

        train['label'] = 1
        train_swap['label'] = 0

        train2 = pd.concat([train,train_swap]).sort_index().reset_index(drop=True)

        return train2




    def add_labels(self, data, label):
        print("Labeling historical data...", end="")
        results = self.dfs["MNCAATourneyCompactResults.csv"].drop(["DayNum", "WScore", "LScore", "WLoc", "NumOT"], axis = 1)
        X = []
        y = []
        for i in range(len(results)):
            result = results.iloc[i]
            team1 = min(result["WTeamID"], result["LTeamID"])
            team2 = max(result["WTeamID"], result["LTeamID"])
            season = result["Season"]
            season_data = data.loc[data["Season"] == season]
            x1 = season_data.loc[(season_data["TeamID"] == team1)]
            x2 = season_data.loc[(season_data["TeamID"] == team2)]

            # We can make our data robust to which team is considered team1 in the input
            # by making a second input with the teams swapped
            if (x1.shape[0] == 1) and (x2.shape[0] == 1):
                x = x1.merge(x2, on=["Season"], suffixes=("_team1", "_team2"))
                xr = x2.merge(x1, on=["Season"], suffixes=("_team1", "_team2"))
                X.append(x)
                X.append(xr)
                if (team1 == result["WTeamID"]):
                    y.append(1)
                    y.append(0)
                else:
                    y.append(0)
                    y.append(1)

        y = pd.Series(y, name=label, dtype=int)
        X = pd.concat(X, axis=0)
        X = X.reset_index()
        X[label] = y

        # Some validation data tinkering suggested the best way to handle missing rankings
        # is to give teams the ~lowest possible rankings. For years where an entire column is
        # missing we set it constant, for systems that don't rank all teams, we treat unranked teams
        # as bad which seems reasonable.
        X = X.fillna(350)

        print("done.")
        return X
        
    
    def get_predict_seasonstats(self, df):
        ''' Get the data needed for the prediction
        '''

        cols2keep = ['T1_TeamID','T2_TeamID']
        dfpred = df[cols2keep].copy()
        dfpred['Season'] = self.currentyear

        # Add the stats for Team1
        dfpred.rename(columns={'T1_TeamID':'TeamID'}, inplace=True)
        #print(dfpred.columns)

        print('xxx')
        print(type(self.season_stats))
        print(self.season_stats)
        #season_stats = self.season_stats.query('Season' == self.currentyear).copy()
        season_stats = self.season_stats.loc[self.season_stats.index.get_level_values("Season") == 2024]
        
        season_stats.columns = ['T1_' + x for x in list(season_stats.columns)]

        dfpred = pd.merge(dfpred, season_stats, on = ['TeamID'], how = 'left')
        dfpred.rename(columns={'TeamID':'T1_TeamID'}, inplace=True)

        # Add the stats for Team2
        dfpred.rename(columns={'T2_TeamID':'TeamID'}, inplace=True)
        season_stats.columns = [x.replace('T1_','T2_') for x in list(season_stats.columns)]

        dfpred = pd.merge(dfpred, season_stats, on = ['TeamID'], how = 'left')
        dfpred.rename(columns={'TeamID':'T2_TeamID'}, inplace=True)

        return dfpred

    def process_test(self, data):
        ''' create unique pairs from all teams in data
        '''
        print("Generating possible tournament matchups for ",self.currentyear,"...", end="")
        X = []
        for i in range(len(data)):
            x1 = data.iloc[i].to_frame().T
            for j in range(len(data)):
                if (i != j):
                    x2 = data.iloc[j].to_frame().T
                    x = pd.merge(x1, x2, on=["Season"], suffixes=("_team1", "_team2"))
                    X.append(x)
        X = pd.concat(X, axis=0)
        X = X.reset_index()
        X = X.fillna(350)

        print("done.")
        
        return X
    
    def get_tourneycombos(self,df):
        '''get all tournament combos, consisting of all uniqure pairs from all teams in data
        '''

        team_combinations = []
        for tournament, group in df.groupby("Tournament"):
            team_ids = group["TeamID"].unique()  # Get unique TeamIDs
            pairs = list(combinations(team_ids, 2))  # Generate all unique pairs
        
            # Store results with tournament info
            for team1, team2 in pairs:
                team_combinations.append({"Tournament": tournament, "T1_TeamID": team1, "T2_TeamID": team2})
        
        # Convert to DataFrame
        df_combinations = pd.DataFrame(team_combinations)
        if type == 'men':
            df_combinations = df_combinations.loc[df_combinations['Tournament'] == 'M'].drop(['Tournament'],axis=1)
        else:
            df_combinations = df_combinations.loc[df_combinations['Tournament'] == 'W'].drop(['Tournament'],axis=1)
        df_sorted = df_combinations.sort_values(by=['T1_TeamID', 'T2_TeamID']).reset_index(drop=True)
        return df_sorted

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test
