import os
import pandas as pd
import numpy as np

# CONSTS
# Todo: Game 201904100POR only had seven players on one of the teams. So we set ppt to 7, find workaround later
# Oh... well that's because portland only played 6 players that game. Still pulled out a W though.
PLAYERS_PER_TEAM = 6  # Number of players to include per team per feature


def odds_year(file='data/odds/odds18-19.csv'):
    use_cols = ['GAME_ID', 'AWAY_ML', 'HOME_ML', 'SPREAD', 'TOTAL_POINTS']

    data = pd.read_csv(file, header=0, usecols=use_cols)

    data = data.values.tolist()
    # data.sort(key=lambda x: x[0])
    data = list(map(lambda x: x[1:], data))

    return data


def odds(years=None):
    odds = []
    if years is None:
        years = ['16-17', '17-18', '18-19']

    for year in years:
        odds += odds_year('data/odds/odds' + year + '.csv')

    return odds



def labels(file='data/game_stats_2017-18.csv'):
    use_cols = ['GAME_ID', 'A_PTS', 'H_PTS']

    data = pd.read_csv(file, header=0, usecols=use_cols)

    data = data.values.tolist()
    # data.sort(key=lambda x: x[0])
    data = list(map(lambda x: x[1:], data))

    return data


def load_player_avgs(file='data/player_avgs_2017-18.csv'):
    # All Stats
    # use_cols = ['PLAYER', 'GAME_ID', 'TEAM', 'AVG_FG', 'AVG_FGA', 'AVG_3P', 'AVG_3PA', 'AVG_FT', 'AVG_FTA', 'AVG_ORB', 'AVG_DRB',
    #             'AVG_TRB', 'AVG_AST', 'AVG_STL', 'AVG_BLK', 'AVG_TOB', 'AVG_PF', 'AVG_PTS']

    # No Rebounds
    # use_cols = ['PLAYER', 'GAME_ID', 'TEAM', 'AVG_FG', 'AVG_FGA', 'AVG_3P', 'AVG_3PA', 'AVG_FT', 'AVG_FTA',
    #             'AVG_AST', 'AVG_STL', 'AVG_BLK', 'AVG_TOB', 'AVG_PTS']

    # Field goals only
    # use_cols = ['PLAYER', 'GAME_ID', 'TEAM', 'AVG_FG', 'AVG_3P']

    # Points and turnovers
    # use_cols = ['PLAYER', 'GAME_ID', 'TEAM', 'AVG_PTS', 'AVG_TOB']

    # Shooting stats
    use_cols = ['PLAYER', 'GAME_ID', 'TEAM', 'AVG_FG', 'AVG_3P', 'AVG_FT']

    # 4 Major Stats
    # use_cols = ['PLAYER', 'GAME_ID', 'TEAM', 'AVG_AST', 'AVG_STL', 'AVG_BLK', 'AVG_PTS']

    data = pd.read_csv(file, header=0, usecols=use_cols)

    return data


def build_player_map(file='data/player_avgs_2017-18.csv'):
    player_map = {}
    data = np.array(load_player_avgs(file))

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


def build_game_map(game_file='data/game_stats_2017-18.csv', player_file='data/player_avgs_2017-18.csv'):
    game_map = {}
    game_data = load_games(game_file)
    player_data = build_player_map(player_file)
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


def raw_features(game_file='data/game_stats_2017-18.csv', player_file='data/player_avgs_2017-18.csv'):
    game_map = build_game_map(game_file, player_file)
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


def load_games(file='data/game_stats_2017-18.csv'):
    use_cols = ['GAME_ID', 'AWAY_TEAM', 'A_AST', 'A_STL', 'A_BLK', 'A_TOB', 'A_PTS',
                'HOME_TEAM', 'H_AST', 'H_STL', 'H_BLK', 'H_TOB', 'H_PTS']

    data = pd.read_csv(file, header=0, usecols=use_cols)

    return data


def features(game_file='data/game_stats_2017-18.csv', player_file='data/player_avgs_2017-18.csv'):
    all_games = raw_features(game_file, player_file)
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

    full_list = list(map(lambda x: x[1:], full_list))  # Remove game_id column and pass this along as feature

    return full_list


def game_avg_features_part(data):
    # All features
    # use_cols = ['GAME_ID', 'A_FG', 'A_FGA', 'A_3P', 'A_3PA', 'A_FT', 'A_FTA', 'A_ORB', 'A_DRB', 'A_TRB', 'A_AST', 'A_STL', 'A_BLK', 'A_TOB', 'A_PF', 'A_AVG_PTS',
    #             'H_FG', 'H_FGA', 'H_3P', 'H_3PA', 'H_FT', 'H_FTA', 'H_ORB', 'H_DRB', 'H_TRB', 'H_AST', 'H_STL', 'H_BLK', 'H_TOB', 'H_PF', 'H_AVG_PTS']

    # Offense features
    use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'H_FG', 'H_3P', 'H_FT']

    # Important features
    # use_cols = ['GAME_ID', 'H_FGA', 'A_AVG_PTS', 'H_DRB', 'H_STL', 'H_PF']

    # Test 1
    # use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'A_ORB', 'A_DRB', 'H_FG', 'H_3P', 'H_FT', 'H_ORB', 'H_DRB']

    data = np.array(pd.read_csv(data, header=0, usecols=use_cols)).tolist()

    data.sort(key=lambda x: x[0])  # Sort by GAME_ID

    data = list(map(lambda x: x[1:], data))  #

    # print(np.array(data))
    return data


def game_avg_features():
    features = []
    years = ['2016-17', '2017-18', '2018-19']  # '2017-18', '2018-19'
    for year in years:
        features += game_avg_features_part('data/game_features_' + year + '.csv')

    return features


def game_avg_labels_part(data):
    use_cols = ['GAME_ID', 'AWAY_POINTS', 'HOME_POINTS']

    data = np.array(pd.read_csv(data, header=0, usecols=use_cols)).tolist()

    data.sort(key=lambda x: x[0])  # Sort by GAME_ID

    data = list(map(lambda x: x[1:], data))  #

    print(np.array(data))
    return data


def game_avg_labels():
    labels = []
    years = ['2013-14', '2014-15', '2015-16', '2016-17',
             '2017-18', '2018-19']  # '2009-10', '2010-11', '2011-12', '2012-13',
    for year in years:
        labels += game_avg_labels_part('data/game_avgs_' + year + '.csv')

    return labels


def adv_features_part(data):
    # All features
    # use_cols = ['GAME_ID', 'A_FG', 'A_FGA', 'A_3P', 'A_3PA', 'A_FT', 'A_FTA', 'A_ORB', 'A_DRB', 'A_TRB', 'A_AST', 'A_STL', 'A_BLK', 'A_TOB', 'A_PF', 'A_AVG_PTS',
    #             'H_FG', 'H_FGA', 'H_3P', 'H_3PA', 'H_FT', 'H_FTA', 'H_ORB', 'H_DRB', 'H_TRB', 'H_AST', 'H_STL', 'H_BLK', 'H_TOB', 'H_PF', 'H_AVG_PTS']

    # Offense features
    use_cols = ['GAME_ID', 'A_TS%', 'A_eFG%', 'A_3PAr', 'A_FTr', 'A_ORB%', 'A_DRB%',
                             'A_TRB%', 'A_AST%', 'A_STL%', 'A_BLK%', 'A_TOV%', 'A_ORtg', 'A_DRtg',
                             'H_TS%', 'H_eFG%', 'H_3PAr', 'H_FTr', 'H_ORB%', 'H_DRB%',
                             'H_TRB%', 'H_AST%', 'H_STL%', 'H_BLK%', 'H_TOV%', 'H_ORtg', 'H_DRtg'
                             ]

    # use_cols = ['GAME_ID', 'A_TS%', 'A_DRtg', 'A_ORB%', 'A_DRB%', 'A_TOV%', 'A_AST%', 'A_STL%', 'A_BLK%',
    #             'H_TS%', 'H_DRtg', 'H_ORB%', 'H_DRB%', 'H_TOV%', 'H_AST%', 'H_STL%', 'H_BLK%'
    #             ]

    # Important features
    # use_cols = ['GAME_ID', 'H_FGA', 'A_AVG_PTS', 'H_DRB', 'H_STL', 'H_PF']

    # Test 1
    # use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'A_ORB', 'A_DRB', 'H_FG', 'H_3P', 'H_FT', 'H_ORB', 'H_DRB']

    data = np.array(pd.read_csv(data, header=0, usecols=use_cols)).tolist()

    data.sort(key=lambda x: x[0])  # Sort by GAME_ID

    data = list(map(lambda x: x[1:], data))  #

    # print(np.array(data))
    return data


def adv_features(years=None):
    features = []
    if years is None:
        years = ['2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17',
                 '2017-18']  # '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
    for year in years:
        features += adv_features_part('data/advanced_stats/adv_avg_features_' + year + '.csv')

    return np.array(features)


def adv_labels_part(data):
    use_cols = ['GAME_ID', 'A_PTS', 'H_PTS']

    data = np.array(pd.read_csv(data, header=0, usecols=use_cols)).tolist()

    data.sort(key=lambda x: x[0])  # Sort by GAME_ID

    data = list(map(lambda x: x[1:], data))  #

    return data


def adv_labels(years=None):
    labels = []
    if years is None:
        years = ['2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17',
                 '2017-18']  # '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16',
    for year in years:
        labels += adv_labels_part('data/advanced_stats/adv_avg_features_' + year + '.csv')

    return np.array(labels)


def adv_diff_features_part(data):
    use_cols = ['GAME_ID', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                             'AST%', 'STL%', 'BLK%', 'TOV%', 'ORtg', 'DRtg']
    # use_cols = ['GAME_ID', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%',
    #             'AST%', 'STL%', 'BLK%', 'TOV%', 'DRtg']

    data = np.array(pd.read_csv(data, header=0, usecols=use_cols)).tolist()

    data.sort(key=lambda x: x[0])  # Sort by GAME_ID

    data = list(map(lambda x: x[1:], data))  #

    # print(np.array(data))
    return data


def adv_diff_features(years):
    print('Called features()')
    features = []
    if years is None:
        years = ['2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17',
                 '2017-18']  # '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
    for year in years:
        features += adv_diff_features_part('data/advanced_stats/adv_diff_features_' + year + '.csv')

    return np.array(features)


def adv_diff_labels_part(data):
    use_cols = ['GAME_ID', 'POINT_DIFF']

    data = np.array(pd.read_csv(data, header=0, usecols=use_cols)).tolist()

    data.sort(key=lambda x: x[0])  # Sort by GAME_ID

    data = list(map(lambda x: x[1:], data))  #

    return data


def adv_diff_labels(years=None):
    print('Called labels()')
    labels = []
    if years is None:
        years = ['2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17',
                 '2017-18']  # '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16',
    for year in years:
        labels += adv_diff_labels_part('data/advanced_stats/adv_diff_features_' + year + '.csv')

    return np.array(labels)


def classifier_data(years=['2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17',
                 '2017-18']):
    columns = ['TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                             'AST%', 'STL%', 'BLK%', 'TOV%', 'ORtg', 'DRtg']
    all_features = pd.DataFrame(adv_diff_features(years))
    all_features.columns = columns

    all_labels = pd.DataFrame(adv_diff_labels(years))
    all_labels.columns = ['POINT_DIFF']

    # Convert labels to binary classes
    all_labels.loc[all_labels.POINT_DIFF > 0] = 0
    all_labels.loc[all_labels.POINT_DIFF < 0] = 1
    # print(all_labels)

    all_features['WINNER'] = all_labels['POINT_DIFF']

    away_winners = pd.DataFrame(all_features.loc[all_features.WINNER == 0])
    home_winners = pd.DataFrame(all_features.loc[all_features.WINNER == 1])

    away_winners['HOME'] = 0
    away_winners['WINNER'] = 1
    away_winners.rename(columns={'WINNER': 'AWAY'}, inplace=True)

    home_winners['AWAY'] = 0
    home_winners['WINNER'] = 1
    home_winners.rename(columns={'WINNER': 'HOME'}, inplace=True)
    # print(home_winners)

    remove_count = len(home_winners) - len(away_winners)
    # print(remove_count)

    # Randomly remove rows in home_winners in order to undersample and even out the classes
    np.random.seed(10)
    drop_games = np.random.choice(home_winners.index, remove_count, replace=False)
    home_winners = home_winners.drop(drop_games)
    # print(home_winners)

    pd.set_option('display.max_columns', 15)
    features = pd.concat([away_winners, home_winners], axis=0)
    # print(features)

    # print('Post popped features')
    away = np.array(features.pop('AWAY'))
    # print(away)
    home = np.array(features.pop('HOME'))
    # print(home)
    labels = pd.DataFrame()
    labels['AWAY'] = away
    labels['HOME'] = home
    # print(features)
    # print(labels)

    return features, labels
