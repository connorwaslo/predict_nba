# NBA Prediction Algorithm

I used a feedforward neural network and statistics from the past decade to predict the outcome of regular season nba games.

## Data

I scraped individual player statistics, basic team stats, and advanced team stats for use in prediction.

Ultimately, advanced team stats proved to be the most effective with some preprocessing...
- For each team's first game of the season, use their last season's final season avg for the given statistic
- For each subsequent game, average their stats from the current season
- Lag the data by one game (it's not a prediction if you can see the future)
- For prediction, use the difference in each team's stats rather than both teams statistics

Data can be found in data/advanced_stats/adv_avg_features_[season].csv

The data was split up into a training set (9 seasons of data) and a testing set (1 season of data)

-----------

Besides NBA statistics, I also collected historic vegas odds for NBA games so that I could determine the profitability of each algorithm.

## Feedforward Neural Network

The method that worked best was the feedforward neural network that predicted the spread (difference in each team's score) for each game.

A positive spread indicated an away win and a negative spread indicated a home win.

I tried an immense amount of different hidden layer sizes and node counts. 
I just built a loop that would go through a few different options and test each option several times in order to calculate an average and get a reasonable idea of the accuracy.
The most successful network was able to predict NBA games with an accuracy of 65% accross an entire NBA season.

## Profitability

Despite a 65% accuracy on win/loss predictions, the algorithm never broke an ROI of around -4%.

While the accuracy looks very compelling, the Vegas moneylines are developed such that one loss will offset your past few wins.

Using this algorithm as it is you may beat your friends in a pick'em league, but you won't beat Vegas.
