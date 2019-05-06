# This doesn't work correctly
def spread_profit(bet_size=10, final_spread=0, vegas_spread=0):
    # Spread odds are -110, or bet 11 to win 10 (10/11 profits)
    if final_spread > vegas_spread:
        return bet_size * (10.0 / 11.0)
    elif final_spread == vegas_spread:
        return 0
    else:
        return -bet_size


def moneyline_profit(bet_size=10, pred_winner=1, away_ml=100, home_ml=-100):
    if pred_winner == 1:  # If the away team wins
        # Calculate away ml profit
        if away_ml < 0:  # If the favorite
            return bet_size * (100 / away_ml)
        else:  # If the underdog
            return bet_size * (away_ml / 100)
    else:
        # Calculate home ml profit
        if home_ml < 0:  # If the favorite
            return bet_size * (100 / home_ml)
        else:  # If the underdog
            return bet_size * (home_ml / 100)


def totals_profit(bet_size=10):
    return bet_size * (100 / 110)
