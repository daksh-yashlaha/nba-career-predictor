# NBA Career Predictor

##Project Description

NBA Career Predictor is a model that is used for predicting if a rookie NBA player is gonna have a successful game career for more than 5 years or not. This model mmakes use of Artificial Neural Networks.

##Modules Required

This project makes use of the following modules with the specified versions:
numpy               version 1.13.3
matplotlib          version 2.1.0
pandas              version 0.20.3
keras               version 2.1.3
scikit-learn        version 0.19.1 

##Software Requirements

This project makes use of Anaconda distrubution of following versions:
anaconda-client     version 1.6.5
anaconda-navigator  version 1.6.12
anaconda-project    version 0.8.0

## Setting up 

Please follow the following steps for setting up and running the ANN model :
1. Download the nba_logreg.csv and ann.py files from the main branch.
2. Open Anaconda Navigator
3. Select Spyder
4. Open the project file 'ann.py'
5. Make sure that the files ann.py and nba_logreg.csv are in the same folder.
6. Run the program

##Dataset Description

The CSV file contains data about different rookie NBA players. The column names are as follows :

###Name        Description
Name	       Player Name
GP             Games Played
MIN	       Minutes Played
PTS            Points Per Game
FGM            Field Goals Made
FGA            Field Goals Attempts
FG%            Field Goal Percent
3P Made        3 Points Made
3PA            3 Points Attempts
3P%            3 Points Percent
FTM            Free Throw Made
FTA            Free Throw Attempts
FT%            Free Throw Percent
OREB           Offensive Rebounds
DREB           Defensive Rebounds
REB            Rebounds
AST            Assists
STL            Steals
BLK            Blocks
TOV            Turnovers
TARGET_5Yrs    Outcome : 1 if career length >= 5yrs, otherwise 0
