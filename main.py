import pandas as pd

import matplotlib.pyplot as plt
plt.style.use("bmh")
import seaborn as sns

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from joblib import load

# Load in Pipeline and Model for formatting data and testing
random_forest_regression_model = load('random_forest_regression_model.joblib')
pipeline = load('preprocessing_pipeline.joblib')


# Data Collection From User
name = input("Player's Name : ").lower()
positions = input("Enter Player's Positions : ").lower()
overall = input("Enter Player's Overall : ")
potential = input("Enter Player's Potential : ")
age = input("Enter Player's Age : ")
height = input("Enter Player's Height (cm) : ")
weight = input("Enter Player's Weight (kg) : ")
league_name = input("Enter Player's League Name : ").lower()
league_level = input("Enter Player's League Level : ").lower()
club_name = input("Enter Player's Club Name : ").lower()
nationality = input("Enter Player's Nationality : ").lower()
pref_foot = input("Enter Player's Preferred Foot : ").lower()
weak_foot = input("Enter Player's Weak Foot Rating : ")
skill_moves = input("Enter Player's Skill Moves Rating : ")
international_reputation = input("Enter Player's International Reputation : ")
work_rate = input("Enter Player's Work Rate Rating : ")
body_type = input("Enter Player's Body Type : ").lower()
player_traits = input("Enter Player's Player Traits : ").lower()
pace = input("Enter Player's Pace Rating : ")
shooting = input("Enter Player's Shooting Rating : ")
passing = input("Enter Player's Passing Rating : ")
dribbling = input("Enter Player's Dribbling Rating : ")
defending = input("Enter Player's Defending Rating : ")
physicality = input("Enter Player's Physicality Rating : ")
crossing = input("Enter Player's Crossing Rating : ")
finishing = input("Enter Player's Finishing Rating : ")
heading_accuracy = input("Enter Player's Heading Accuracy Rating : ")
short_passing = input("Enter Player's Short Passing Rating : ")
volleys = input("Enter Player's Volleys Rating : ")
skill_dribbling = input("Enter Player's Skill Dribbling Rating : ")
curve = input("Enter Player's Curve Rating : ")
fk_accuracy = input("Enter Player's FK Accuracy Rating : ")
long_passing = input("Enter Player's Long Passing Rating : ")
ball_control = input("Enter Player's Ball Control Rating : ")
acceleration = input("Enter Player's Acceleration Rating : ")
sprint_speed = input("Enter Player's Sprint Speed Rating : ")
agility = input("Enter Player's Agility Rating : ")
reactions = input("Enter Player's Reactions Rating : ")
balance = input("Enter Player's Balance Rating : ")
shot_power = input("Enter Player's Shot Power Rating : ")
jumping = input("Enter Player's Jumping Rating : ")
stamina = input("Enter Player's Stamina Rating : ")
strength = input("Enter Player's Strength Rating : ")
long_shots = input("Enter Player's Long Shots Rating : ")
aggression = input("Enter Player's Aggression Rating : ")
interception = input("Enter Player's Interception Rating : ")
positioning = input("Enter Player's Positioning Rating : ")
vision = input("Enter Player's Vision Rating : ")
penalties = input("Enter Player's Penalties Rating : ")
composure = input("Enter Player's Composure Rating : ")
marking_awareness = input("Enter Player's Marking Awareness Rating : ")
standing_tackle = input("Enter Player's Standing Tackle Rating : ")
sliding_tackle = input("Enter Player's Sliding Tackle Rating : ")
diving = input("Enter Player's Diving Rating : ")
handling = input("Enter Player's Handling Rating : ")
kicking = input("Enter Player's Kicking Rating : ")
goalkeeper_positioning = input("Enter Player's Goalkeeper Positioning Rating : ")
reflexes = input("Enter Player's Reflexes Rating : ")
speed = input("Enter Player's Speed Rating : ")


player_data = {
    'player_positions': [positions],
    'overall': [int(overall)],
    'potential': [int(potential)],
    'age': [int(age)],
    'height_cm': [int(height)],
    'weight_kg': [int(weight)],
    'league_name': [league_name],
    'league_level': [int(league_level)],
    'club_name': [club_name],
    'nationality_name': [nationality],
    'preferred_foot': [pref_foot],
    'weak_foot': [int(weak_foot)],
    'skill_moves': [int(skill_moves)],
    'international_reputation': [int(international_reputation)],
    'work_rate': [work_rate],
    'body_type': [body_type],
    'player_traits': [player_traits],  
    'pace': [int(pace)],
    'shooting': [int(shooting)],
    'passing': [int(passing)],
    'dribbling': [int(dribbling)],  
    'defending': [int(defending)],
    'physic': [int(physicality)],  
    'attacking_crossing': [int(crossing)],
    'attacking_finishing': [int(finishing)],
    'attacking_heading_accuracy': [int(heading_accuracy)],
    'attacking_short_passing': [int(short_passing)],
    'attacking_volleys': [int(volleys)],
    'skill_dribbling': [int(skill_dribbling)], 
    'skill_curve': [int(curve)],
    'skill_fk_accuracy': [int(fk_accuracy)],
    'skill_long_passing': [int(long_passing)],
    'skill_ball_control': [int(ball_control)],
    'movement_acceleration': [int(acceleration)],
    'movement_sprint_speed': [int(sprint_speed)],
    'movement_agility': [int(agility)],
    'movement_reactions': [int(reactions)],
    'movement_balance': [int(balance)],
    'power_shot_power': [int(shot_power)],
    'power_jumping': [int(jumping)],
    'power_stamina': [int(stamina)],
    'power_strength': [int(strength)],
    'power_long_shots': [int(long_shots)],
    'mentality_aggression': [int(aggression)],
    'mentality_interceptions': [int(interception)],
    'mentality_positioning': [int(positioning)],
    'mentality_vision': [int(vision)],
    'mentality_penalties': [int(penalties)],
    'mentality_composure': [int(composure)],
    'defending_marking_awareness': [int(marking_awareness)],
    'defending_standing_tackle': [int(standing_tackle)],
    'defending_sliding_tackle': [int(sliding_tackle)],
    'goalkeeping_diving': [int(diving)],
    'goalkeeping_handling': [int(handling)],
    'goalkeeping_kicking': [int(kicking)],
    'goalkeeping_positioning': [int(goalkeeper_positioning)], 
    'goalkeeping_reflexes': [int(reflexes)],
    'goalkeeping_speed': [int(speed)]
}

input_data = pd.DataFrame.from_dict(player_data)

categorical_features = ['player_positions', 'league_name', 'club_name', 'nationality_name', 'preferred_foot', 'body_type', 'player_traits', 'work_rate']
for column in categorical_features:
    input_data[column] = input_data[column].apply(lambda x: x.lower() if type(x) == str else x)
    
transformed_input = pipeline.transform(input_data)

predicted_value_eur = random_forest_regression_model.predict(transformed_input)

print(f'The Player : {name} \n')
print('With the following stats : \n')

for stat in player_data : 
    print(f'{stat} : {player_data[stat]}')
    
print(f'\n Has the estimated market value (EUR) ≈ €{predicted_value_eur} \n')
print("Please keep in mind that the data this model is built on was collected on September 11th, 2023\n As such, Player Values reflect what the standard was at that time.\n")
print("Through my limited testing I've found that the discrepency rate from market values listed on www.fifacm.com collected as of September 11th, 2023,\n to the model's predicted value tends to hover around 3%")
