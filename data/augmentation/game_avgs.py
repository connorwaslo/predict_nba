import csv


def avg(li):
    return sum(li) / len(li)


def end_season_avgs(file='', team=''):
    """

    :param file: The stats file which we're looking for
    :param team: 3 letter abbreviation of the team name
    :return: List containing the statline for the team's avg at the end of the season
    """

    with open(file, 'r') as f:
        reader = csv.reader(f)

        last_occurrence = []

        # Away: 2
        # Home: 22
        for row in reader:
            if row[1] == team:
                last_occurrence = [float(item) for item in row[2:18]]  # Leave out team name and GAME_ID
            elif row[18] == team:
                last_occurrence = [float(item) for item in row[19:]]  # Leave out team name

        return last_occurrence


in_files = ['../game_stats_2015-16.csv', '../game_stats_2016-17.csv', '../game_stats_2017-18.csv', '../game_stats_2018-19.csv']
out_files = ['../game_avgs_2015-16.csv', '../game_avgs_2016-17.csv', '../game_avgs_2017-18.csv', '../game_avgs_2018-19.csv']


def write_avgs():
    for file_num, file in enumerate(in_files):
        teams = {}

        with open(file, 'r') as f:
            reader = csv.reader(f)

            row_count = 0
            for row in reader:
                if row_count > 0:
                    game_id = row[1]
                    away_team = row[2]

                    statline = [game_id, away_team]

                    # Check if need to add blank team
                    if away_team not in teams.keys():
                        teams[away_team] = {'FG': [], 'FGA': [], '3P': [], '3PA': [], 'FT': [], 'FTA': [],
                        'ORB': [], 'DRB': [], 'TRB': [], 'AST': [], 'STL': [], 'BLK': [],
                        'TOB': [], 'PF': [], 'PTS': []}
                    else:
                        print('Away\t', away_team, teams[away_team])

                    away_fg = float(row[4])
                    away_fga = float(row[5])
                    away_3p = float(row[7])
                    away_3pa = float(row[8])
                    away_ft = float(row[10])
                    away_fta = float(row[11])
                    away_orb = float(row[13])
                    away_drb = float(row[14])
                    away_trb = float(row[15])
                    away_ast = float(row[16])
                    away_stl = float(row[17])
                    away_blk = float(row[18])
                    away_tob = float(row[19])
                    away_pf = float(row[20])
                    away_pts = float(row[21])

                    # Add stats to running list for team
                    teams[away_team]['FG'].append(away_fg)
                    teams[away_team]['FGA'].append(away_fga)
                    teams[away_team]['3P'].append(away_3p)
                    teams[away_team]['3PA'].append(away_3pa)
                    teams[away_team]['FT'].append(away_ft)
                    teams[away_team]['FTA'].append(away_fta)
                    teams[away_team]['ORB'].append(away_orb)
                    teams[away_team]['DRB'].append(away_drb)
                    teams[away_team]['TRB'].append(away_trb)
                    teams[away_team]['AST'].append(away_ast)
                    teams[away_team]['STL'].append(away_stl)
                    teams[away_team]['BLK'].append(away_blk)
                    teams[away_team]['TOB'].append(away_tob)
                    teams[away_team]['PF'].append(away_pf)
                    teams[away_team]['PTS'].append(away_pts)

                    # Append averages to list
                    statline.append(avg(teams[away_team]['FG']))
                    statline.append(avg(teams[away_team]['FGA']))
                    statline.append(avg(teams[away_team]['3P']))
                    statline.append(avg(teams[away_team]['3PA']))
                    statline.append(avg(teams[away_team]['FT']))
                    statline.append(avg(teams[away_team]['FTA']))
                    statline.append(avg(teams[away_team]['ORB']))
                    statline.append(avg(teams[away_team]['DRB']))
                    statline.append(avg(teams[away_team]['TRB']))
                    statline.append(avg(teams[away_team]['AST']))
                    statline.append(avg(teams[away_team]['STL']))
                    statline.append(avg(teams[away_team]['BLK']))
                    statline.append(avg(teams[away_team]['TOB']))
                    statline.append(avg(teams[away_team]['PF']))
                    statline.append(avg(teams[away_team]['PTS']))
                    statline.append(away_pts)

                    home_team = row[22]
                    statline.append(home_team)
                    # Check if need to add blank team
                    if home_team not in teams.keys():
                        teams[home_team] = {'FG': [], 'FGA': [], '3P': [], '3PA': [], 'FT': [], 'FTA': [],
                        'ORB': [], 'DRB': [], 'TRB': [], 'AST': [], 'STL': [], 'BLK': [],
                        'TOB': [], 'PF': [], 'PTS': []}
                    else:
                        print('Home:\t', teams[home_team])

                    home_fg = float(row[24])
                    home_fga = float(row[25])
                    home_3p = float(row[27])
                    home_3pa = float(row[28])
                    home_ft = float(row[30])
                    home_fta = float(row[31])
                    home_orb = float(row[33])
                    home_drb = float(row[34])
                    home_trb = float(row[35])
                    home_ast = float(row[36])
                    home_stl = float(row[37])
                    home_blk = float(row[38])
                    home_tob = float(row[39])
                    home_pf = float(row[40])
                    home_pts = float(row[41])

                    # Add stats to running list for team
                    teams[home_team]['FG'].append(home_fg)
                    teams[home_team]['FGA'].append(home_fga)
                    teams[home_team]['3P'].append(home_3p)
                    teams[home_team]['3PA'].append(home_3pa)
                    teams[home_team]['FT'].append(home_ft)
                    teams[home_team]['FTA'].append(home_fta)
                    teams[home_team]['ORB'].append(home_orb)
                    teams[home_team]['DRB'].append(home_drb)
                    teams[home_team]['TRB'].append(home_trb)
                    teams[home_team]['AST'].append(home_ast)
                    teams[home_team]['STL'].append(home_stl)
                    teams[home_team]['BLK'].append(home_blk)
                    teams[home_team]['TOB'].append(home_tob)
                    teams[home_team]['PF'].append(home_pf)
                    teams[home_team]['PTS'].append(home_pts)

                    # Append averages to list
                    statline.append(avg(teams[home_team]['FG']))
                    statline.append(avg(teams[home_team]['FGA']))
                    statline.append(avg(teams[home_team]['3P']))
                    statline.append(avg(teams[home_team]['3PA']))
                    statline.append(avg(teams[home_team]['FT']))
                    statline.append(avg(teams[home_team]['FTA']))
                    statline.append(avg(teams[home_team]['ORB']))
                    statline.append(avg(teams[home_team]['DRB']))
                    statline.append(avg(teams[home_team]['TRB']))
                    statline.append(avg(teams[home_team]['AST']))
                    statline.append(avg(teams[home_team]['STL']))
                    statline.append(avg(teams[home_team]['BLK']))
                    statline.append(avg(teams[home_team]['TOB']))
                    statline.append(avg(teams[home_team]['PF']))
                    statline.append(avg(teams[home_team]['PTS']))
                    statline.append(home_pts)

                    with open(out_files[file_num], 'a', newline='') as of:
                        writer = csv.writer(of)

                        writer.writerow(statline)

                row_count += 1


def write_features():
    feature_files = [name.replace('avgs', 'features') for name in out_files]

    for file_num, file in enumerate(out_files[1:]):
        teams = {}

        with open(file, 'r') as f:
            reader = csv.reader(f)

            row_count = 0
            for row in reader:
                if row_count > 0:
                    game_id = row[0]
                    away_team = row[1]

                    statline = [game_id, away_team]

                    if away_team not in teams.keys():
                        teams[away_team] = []
                        teams[away_team].append(end_season_avgs(out_files[file_num], away_team))
                        statline.extend(teams[away_team][0])
                        teams[away_team].append(row[2:18])  # Also append this game's stats
                    else:
                        statline.extend(teams[away_team][-1])
                        teams[away_team].append(row[2:18])

                    home_team = row[18]
                    statline.append(home_team)
                    # Check if need to add blank team
                    if home_team not in teams.keys():
                        teams[home_team] = []
                        teams[home_team].append(end_season_avgs(out_files[file_num], home_team))
                        statline.extend(teams[home_team][0])
                        teams[home_team].append(row[19:])  # Also append this game's stats
                    else:
                        statline.extend(teams[home_team][-1])
                        teams[home_team].append(row[19:])

                    with open(feature_files[file_num], 'a', newline='') as of:
                        writer = csv.writer(of)

                        writer.writerow(statline)

                row_count += 1


# write_avgs()
write_features()
