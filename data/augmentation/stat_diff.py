import csv

in_files = ['../advanced_stats/adv_avg_features_2009-10.csv',
            '../advanced_stats/adv_avg_features_2010-11.csv',
            '../advanced_stats/adv_avg_features_2011-12.csv',
            '../advanced_stats/adv_avg_features_2012-13.csv',
            '../advanced_stats/adv_avg_features_2013-14.csv',
            '../advanced_stats/adv_avg_features_2014-15.csv',
            '../advanced_stats/adv_avg_features_2015-16.csv',
            '../advanced_stats/adv_avg_features_2016-17.csv',
            '../advanced_stats/adv_avg_features_2017-18.csv',
            '../advanced_stats/adv_avg_features_2018-19.csv']
out_files = [file.replace('avg', 'diff') for file in in_files]

for infile, outfile in zip(in_files, out_files):
    data = []

    with open(infile, 'r') as f:
        reader = csv.reader(f)

        row_count = 0
        for row in reader:
            game = []
            if row_count == 0:
                game.append(['GAME_ID', 'AWAY', 'HOME', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                             'AST%', 'STL%', 'BLK%', 'TOV%', 'ORtg', 'DRtg', 'POINT_DIFF'])
            else:
                offset = 15
                game.append(row[0])
                game.append(row[1])
                game.append(row[16])
                for i in range(2, 16):
                    print('Index:', i + offset, row_count, infile)
                    game.append(float(row[i]) - float(row[i + offset]))

            data.append(game)
            print(row_count)

            row_count += 1

    with open(outfile, 'w') as f:
        writer = csv.writer(f)

        for game in data:
            writer.writerow(game)
