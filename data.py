import csv
import pandas as pd
import numpy as np

# CONSTS
# Todo: Game 201904100POR only had seven players on one of the teams. So we set ppt to 7, find workaround later
# Oh... well that's because portland only played 6 players that game. Still pulled out a W though.
PLAYERS_PER_TEAM = 6  # Number of players to include per team per feature


def labels():
    file = 'data/game_stats.csv'
    use_cols = ['GAME_ID', 'A_PTS', 'H_PTS']

    data = pd.read_csv(file, header=0, usecols=use_cols)

    data = data.values.tolist()
    data.sort(key=lambda x: x[0])
    data = list(map(lambda x: x[1:], data))

    return data


def load_player_avgs():
    file = 'data/player_avgs.csv'
    use_cols = ['PLAYER', 'GAME_ID', 'TEAM', 'AVG_FG', 'AVG_FGA', 'AVG_3P', 'AVG_3PA', 'AVG_FT', 'AVG_FTA', 'AVG_ORB', 'AVG_DRB',
                'AVG_TRB', 'AVG_AST', 'AVG_STL', 'AVG_BLK', 'AVG_TOB', 'AVG_PF', 'AVG_PTS']

    data = pd.read_csv(file, header=0, usecols=use_cols)

    return data


def build_player_map():
    player_map = {}
    data = np.array(load_player_avgs())

    for row in data:
        # If a player did not play for whatever reason, don't include that game in their data
        if True not in pd.isnull(row):
            name = row[0]
            game_id = row[1]
            # If the player already exists
            if name in player_map.keys():
                player_map[name][game_id] = row[2:].tolist()  # Use game_id as key to store data
            else:
                player_map[name] = {}
                player_map[name][game_id] = row[2:].tolist()

    return player_map


def build_game_map():
    game_map = {}
    game_data = load_games()
    player_data = build_player_map()
    game_ids = []
    for row in game_data:
        game_ids.append(row[0])

    for player in player_data.keys():
        for date in player_data[player].keys():
            if date in game_map.keys():
                # Get team name
                team = player_data[player][date][0]
                if team in game_map[date].keys():
                    if len(game_map[date][team]) < PLAYERS_PER_TEAM:
                        game_map[date][team].append(player_data[player][date][1:])
                else:
                    game_map[date][team] = []
                    game_map[date][team].append(player_data[player][date][1:])
            else:
                game_map[date] = {}

                # Get team name, make it a key in the map dict, and then add the player into
                team = player_data[player][date][0]
                game_map[date][team] = []
                game_map[date][team].append(player_data[player][date][1:])

    # print(len(game_map['201904100POR']['SAC']))
    # print(len(game_map['201904100POR']['POR']))
    return game_map


def raw_features():
    game_map = build_game_map()
    features = []

    for key in game_map.keys():
        # print(key)
        game = [key]
        temp = []
        for team in game_map[key].keys():
            # Check if this is the home team
            if team in key:
                temp = game_map[key][team]
            else:
                game.append(game_map[key][team])

        # Add the home team stats to the end
        if len(temp) > 0:
            game.append(temp)

        features.append(game)

    return features


def load_games():
    file = 'data/game_stats.csv'
    use_cols = ['GAME_ID', 'AWAY_TEAM', 'A_FG', 'A_FGA', 'A_FG%', 'A_3P', 'A_3PA', 'A_3P%', 'A_FT', 'A_FTA', 'A_FT%', 'A_ORB',
                'A_DRB', 'A_TRB', 'A_AST', 'A_STL', 'A_BLK', 'A_TOB', 'A_PF', 'A_PTS',
                'HOME_TEAM', 'H_FG', 'H_FGA', 'H_FG%', 'H_3P', 'H_3PA', 'H_3P%', 'H_FT', 'H_FTA', 'H_FT%', 'H_ORB',
                'H_DRB', 'H_TRB', 'H_AST', 'H_STL', 'H_BLK', 'H_TOB', 'H_PF', 'H_PTS']

    data = pd.read_csv(file, header=0, usecols=use_cols)

    return data


# Todo: Create each game as two channels. One for home team another for the away team.
def features():
    all_games = raw_features()
    full_list = []
    for game in all_games:
        game_id = game[0]
        away = game[1]
        home = game[2]

        data = [game_id]
        for player in away:
            for stat in player:
                data.append(stat)
        for player in home:
            for stat in player:
                data.append(stat)

        full_list.append(data)

    full_list.sort(key=lambda x: x[0])  # Sort by GAME_ID

    print(full_list[1228])

    full_list = list(map(lambda x: x[1:], full_list))  # Remove game_id column and pass this along as feature

    return full_list


# Not useful as of right now...
def features_dataframe():
    feats = features()
    labs = labels()

    labs = list(map(lambda x: x[1:], labs))

    all_feautures = []
    for feat, lab in zip(feats, labs):
        all_feautures.append(feat + lab)

    # This doesn't actually work because too many columns
    headers = ['GAME_ID', 'AVG_FG', 'AVG_FGA', 'AVG_3P', 'AVG_3PA', 'AVG_FT', 'AVG_FTA', 'AVG_ORB', 'AVG_DRB',
                'AVG_TRB', 'AVG_AST', 'AVG_STL', 'AVG_BLK', 'AVG_TOB', 'AVG_PF', 'AVG_PTS', 'AWAY_POINTS', 'HOME_POINTS']
    df = pd.DataFrame(data=all_feautures, columns=headers)

    return df
