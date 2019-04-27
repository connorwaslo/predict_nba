# Let's try some things

1. Organize data by games.
    - Each game is a data-point
    - Take in 10 players from each team. Stats only.
    - Use the player stats from each team to predict the final stats for each team
        - Note: In the future, may have to include player IDs because different players affect each other in different ways

2. Organize data by player and then nest player in game
    - Map player to playerid. Each game would then have two teams of player ids
    - Features within games will ultimately look the same, just including a playerid which hopefully the algo would learn affects outcomes.
        - This may require more data though so that we get many matchups
       
# Brainstorming

- Factor in how long a player has been with their team as a sort of +/- statistic
