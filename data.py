import csv
import pandas as pd


def load_player_avgs():
    file = 'data/player_avgs.csv'
    use_cols = ['PLAYER', 'GAME_ID', 'AVG_FG', 'AVG_FGA', 'AVG_3P', 'AVG_3PA', 'AVG_FT', 'AVG_FTA', 'AVG_ORB', 'AVG_DRB',
                'AVG_TRB', 'AVG_AST', 'AVG_STL', 'AVG_BLK', 'AVG_TOB', 'AVG_PF', 'AVG_PTS']

    data = pd.read_csv(file, header=0, usecols=use_cols)
    print(data)

    return data


def build_player_maps():
    data = load_player_avgs()



def load_games():
    file = 'data/game_stats.csv'
    use_cols = ['AWAY_TEAM', 'A_FG', 'A_FGA', 'A_FG%', 'A_3P', 'A_3PA', 'A_3P%', 'A_FT', 'A_FTA', 'A_FT%', 'A_ORB',
                'A_DRB', 'A_TRB', 'A_AST', 'A_STL', 'A_BLK', 'A_TOB', 'A_PF', 'A_PTS',
                'HOME_TEAM', 'H_FG', 'H_FGA', 'H_FG%', 'H_3P', 'H_3PA', 'H_3P%', 'H_FT', 'H_FTA', 'H_FT%', 'H_ORB',
                'H_DRB', 'H_TRB', 'H_AST', 'H_STL', 'H_BLK', 'H_TOB', 'H_PF', 'H_PTS']

    data = pd.read_csv(file, header=0, usecols=use_cols)
    print(data)

    return data


# load_player_avgs()
# load_games()