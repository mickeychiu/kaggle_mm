import pandas as pd

def break_even_moneyline(probability):
    if probability > 0.5:
        x = -(100 / (1-probability))+100
    else:
        x = 100/probability - 100
    return x

def predict_probs_and_moneylines(data):
    predictions = pd.DataFrame(columns=["Team1", "Seed1", "Team2", "Seed2", "Win%1", "Win%2", "ML1", "ML2"])
    pred_array = []
    for i in range(len(data)):
        game = data[["TeamName", "TeamName_team2", "Seed_team1", "Seed_team2", "pred", "proba"]].iloc[i]
        mirror = data[(data["TeamName"] == game["TeamName_team2"]) & (data["TeamName_team2"] == game["TeamName"])]
        #if i == 0:
        #    print("XXX")
        #    print(type(game))
        #    print(type(mirror))
        prob1 = float((game["proba"] + (1 - mirror["proba"].iloc[0])) / 2)
        prob2 = float(((1-game["proba"]) + mirror["proba"].iloc[0]) / 2)
        pred_array.append([game["TeamName"], game["Seed_team1"], game["TeamName_team2"], game["Seed_team2"], prob1, prob2, int(break_even_moneyline(prob1)), int(break_even_moneyline(prob2))])
    
    return pd.DataFrame(pred_array, columns = predictions.columns)

def get_first_four(files):
    # Get the first four teamids and seeds
    mseeds = files['MNCAATourneySeeds.csv'].query('Season == 2024')
    ff = mseeds[mseeds['Seed'].str.len() == 4]
    #print(ff)

    # merge with MTeams to get TeamName
    mteams = files['MTeams.csv']
    ff_mteams = pd.merge(ff, mteams, on='TeamID', how='inner')

    ff_mteama = ff_mteams.iloc[::2]
    ff_mteamb = ff_mteams.iloc[1::2]
    #print(ff_mteama)
    #print(ff_mteamb)

    first_four = []
    for i in range(len(ff_mteama)):
        first_four.append( (ff_mteama.iloc[i]['TeamName'],ff_mteamb.iloc[i]['TeamName']) )

    return first_four

def get_round1(files):
    # Get the round1 teamids and seeds
    mseeds = files['MNCAATourneySeeds.csv'].query('Season == 2024')
    #ff = mseeds[mseeds['Seed'].str.len() != 4]
    #print(mseeds)

    # merge with MTeams to get TeamName
    mteams = files['MTeams.csv']
    mseedteams = pd.merge(mseeds, mteams, on='TeamID', how='inner')

    #
    mslots = files['MNCAATourneySlots.csv'].query("(Season == 2024) & ((Slot.str.match('^R1') | Slot.str.match('^[WXYZ]')))")

    mround1 = pd.merge(mslots, mseedteams, left_on='StrongSeed', right_on='Seed', how='inner')
    mround1 = pd.merge(mround1, mseedteams, left_on='WeakSeed', right_on='Seed', how='inner')

    mround1.drop(columns=['Season_x','Season_y','StrongSeed','WeakSeed','FirstD1Season_x','LastD1Season_x','FirstD1Season_y','LastD1Season_y'], inplace=True)  # Optional

    #print(mseedteams.to_string())
    #print(mslots)
    #print(mround1)

    round1 = []
    for i in range(len(mround1)):
        round1.append( (mround1.iloc[i]['TeamName_x'],mround1.iloc[i]['TeamName_y']) )

    return round1


def pretty_print_matchups(pred_df, matchups, include_moneyline=True):
    for (x, y) in matchups:
        #game = pred_df[(pred_df["Team1"] == x) & (pred_df["Team2"]==y)].iloc[0]
        game = pred_df.query("Team1 == @x & Team2 == @y")
        print(game["Team1"] + " (" + str(game["Seed1"]) + ") vs. " + game["Team2"] + " (" + str(game["Seed2"]) + "):")
        print("{:.2%}".format(game["Win%1"].iloc[0]) + " : {:.2%}".format(game["Win%2"].iloc[0]))
        if (include_moneyline):
            print("Moneyline: " + str(abs(game["ML1"].iloc[0])))
        print()


