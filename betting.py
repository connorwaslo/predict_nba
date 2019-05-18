

# @param bet_size: How much money you're putting on this bet
# @param pred_spread: The point differential predicted
# @param game_spread: The actual point differential
# @param vegas_spread: The spread that vegas predicted and set the line at
# @param away_ml: Just there to figure out who the favorite is Todo: One of these may not be necessary, because if one is negative the other positive and vice versa
# @param home_ml: Just there to figure out who the favorite is
def spread_profit(bet_size=10):
    # pred_spread > 0, home wins
    # pred_spread < 0, away wins
    return bet_size * (10 / 11)


def moneyline_profit(bet_size=10, pred_winner=1, away_ml=100, home_ml=-100):
    if pred_winner == 0:  # If the away team wins
        # Calculate away ml profit
        if away_ml < 0:  # If the favorite
            print('Multiplier:', abs(100.0 / away_ml))
            return bet_size * abs(100.0 / away_ml)
        else:  # If the underdog
            print('Multiplier:', abs(away_ml / 100.0))
            return bet_size * abs(away_ml / 100.0)
    else:
        # Calculate home ml profit
        if home_ml < 0:  # If the favorite
            print('Multiplier:', abs(100.0 / home_ml))
            return bet_size * abs(100.0 / home_ml)
        else:  # If the underdog
            print('Multiplier:', abs(home_ml / 100.0))
            return bet_size * abs(home_ml / 100.0)


def totals_profit(bet_size=10):
    return bet_size * (100 / 110)
