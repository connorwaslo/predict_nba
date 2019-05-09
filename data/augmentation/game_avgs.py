import csv


def avg(li):
    return sum(li) / len(li)


in_files = ['../game_stats_2016-17.csv', '../game_stats_2017-18.csv']
out_files = ['../game_avgs_2016-17.csv', '../game_avgs_2017-18.csv']

away_teams = {}
home_teams = {}

for file_num, file in enumerate(in_files):
    with open(file, 'r') as f:
        reader = csv.reader(f)

        team_stats_template = {'FG': [], 'FGA': [], '3P': [], '3PA': [], 'FT': [], 'FTA': [],
                    'ORB': [], 'DRB': [], 'TRB': [], 'AST': [], 'STL': [], 'BLK': [],
                    'TOB': [], 'PF': [], 'PTS': []}

        row_count = 0
        for row in reader:
            if row_count > 0:
                game_id = row[1]
                away_team = row[2]

                statline = [game_id, away_team]

                # Check if need to add blank team
                if away_team not in away_teams.keys():
                    away_teams[away_team] = {'FG': [], 'FGA': [], '3P': [], '3PA': [], 'FT': [], 'FTA': [],
                    'ORB': [], 'DRB': [], 'TRB': [], 'AST': [], 'STL': [], 'BLK': [],
                    'TOB': [], 'PF': [], 'PTS': []}
                else:
                    print('Away\t', away_team, away_teams[away_team])

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
                away_teams[away_team]['FG'].append(away_fg)
                away_teams[away_team]['FGA'].append(away_fga)
                away_teams[away_team]['3P'].append(away_3p)
                away_teams[away_team]['3PA'].append(away_3pa)
                away_teams[away_team]['FT'].append(away_ft)
                away_teams[away_team]['FTA'].append(away_fta)
                away_teams[away_team]['ORB'].append(away_orb)
                away_teams[away_team]['DRB'].append(away_drb)
                away_teams[away_team]['TRB'].append(away_trb)
                away_teams[away_team]['AST'].append(away_ast)
                away_teams[away_team]['STL'].append(away_stl)
                away_teams[away_team]['BLK'].append(away_blk)
                away_teams[away_team]['TOB'].append(away_tob)
                away_teams[away_team]['PF'].append(away_pf)
                away_teams[away_team]['PTS'].append(away_pts)

                # Append averages to list
                statline.append(avg(away_teams[away_team]['FG']))
                statline.append(avg(away_teams[away_team]['FGA']))
                statline.append(avg(away_teams[away_team]['3P']))
                statline.append(avg(away_teams[away_team]['3PA']))
                statline.append(avg(away_teams[away_team]['FT']))
                statline.append(avg(away_teams[away_team]['FTA']))
                statline.append(avg(away_teams[away_team]['ORB']))
                statline.append(avg(away_teams[away_team]['DRB']))
                statline.append(avg(away_teams[away_team]['TRB']))
                statline.append(avg(away_teams[away_team]['AST']))
                statline.append(avg(away_teams[away_team]['STL']))
                statline.append(avg(away_teams[away_team]['BLK']))
                statline.append(avg(away_teams[away_team]['TOB']))
                statline.append(avg(away_teams[away_team]['PF']))
                statline.append(avg(away_teams[away_team]['PTS']))
                statline.append(away_pts)

                home_team = row[22]
                statline.append(home_team)
                # Check if need to add blank team
                if home_team not in home_teams.keys():
                    home_teams[home_team] = {'FG': [], 'FGA': [], '3P': [], '3PA': [], 'FT': [], 'FTA': [],
                    'ORB': [], 'DRB': [], 'TRB': [], 'AST': [], 'STL': [], 'BLK': [],
                    'TOB': [], 'PF': [], 'PTS': []}
                else:
                    print('Home:\t', home_teams[home_team])

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
                home_teams[home_team]['FG'].append(home_fg)
                home_teams[home_team]['FGA'].append(home_fga)
                home_teams[home_team]['3P'].append(home_3p)
                home_teams[home_team]['3PA'].append(home_3pa)
                home_teams[home_team]['FT'].append(home_ft)
                home_teams[home_team]['FTA'].append(home_fta)
                home_teams[home_team]['ORB'].append(home_orb)
                home_teams[home_team]['DRB'].append(home_drb)
                home_teams[home_team]['TRB'].append(home_trb)
                home_teams[home_team]['AST'].append(home_ast)
                home_teams[home_team]['STL'].append(home_stl)
                home_teams[home_team]['BLK'].append(home_blk)
                home_teams[home_team]['TOB'].append(home_tob)
                home_teams[home_team]['PF'].append(home_pf)
                home_teams[home_team]['PTS'].append(home_pts)

                # Append averages to list
                statline.append(avg(home_teams[home_team]['FG']))
                statline.append(avg(home_teams[home_team]['FGA']))
                statline.append(avg(home_teams[home_team]['3P']))
                statline.append(avg(home_teams[home_team]['3PA']))
                statline.append(avg(home_teams[home_team]['FT']))
                statline.append(avg(home_teams[home_team]['FTA']))
                statline.append(avg(home_teams[home_team]['ORB']))
                statline.append(avg(home_teams[home_team]['DRB']))
                statline.append(avg(home_teams[home_team]['TRB']))
                statline.append(avg(home_teams[home_team]['AST']))
                statline.append(avg(home_teams[home_team]['STL']))
                statline.append(avg(home_teams[home_team]['BLK']))
                statline.append(avg(home_teams[home_team]['TOB']))
                statline.append(avg(home_teams[home_team]['PF']))
                statline.append(avg(home_teams[home_team]['PTS']))
                statline.append(home_pts)

                with open(out_files[file_num], 'a', newline='') as of:
                    writer = csv.writer(of)

                    writer.writerow(statline)

            row_count += 1
