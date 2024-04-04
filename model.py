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

from joblib import dump

data = pd.read_csv('male_players.csv')

data = data.dropna(subset=['value_eur'])

data = data.reset_index()

features = data[['player_positions', 'overall', 'potential', 'age', 'height_cm', 'weight_kg', 'league_name', 'league_level', 'club_name', 'nationality_name', 'preferred_foot', 'weak_foot', 'skill_moves', 'international_reputation', 'work_rate', 'body_type', 'player_traits', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']]
target = data['value_eur']





'''
sns.set(rc={'figure.figsize':(15,15)})
corr = features.corr()
sns.heatmap(corr[((corr >= 0) | (corr <= 0)) & (corr != 1)], annot=False, linewidths=.5, fmt='.2f')
plt.title('Corelation Matrix')
plt.show()
'''


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, shuffle=True)

# player_positions, league_name, club_name, nationality_name, preferred_foot, body_type, player_traits, work_rate
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = ['player_positions', 'league_name', 'club_name', 'nationality_name', 'preferred_foot', 'body_type', 'player_traits', 'work_rate']

for column in categorical_features:
    data[column] = data[column].apply(lambda x: x.lower() if type(x) == str else x)

column_transformer = ColumnTransformer(
    transformers=[
        ('numeric', SimpleImputer(strategy='mean'), numeric_features),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)


pipeline = Pipeline([
    ('preprocessor', column_transformer),
    ('scaler', StandardScaler(with_mean=False))
])


X_train_preprocessed = pipeline.fit_transform(X_train)


X_test_preprocessed = pipeline.transform(X_test)

'''
# Linear Regression Model

lin_reg = LinearRegression()

lin_reg.fit(X_train_preprocessed, y_train)
reg_score = lin_reg.score(X_test_preprocessed, y_test)
print(f"Linear Regression Score: {reg_score} \n")

y_pred = lin_reg.predict(X_test_preprocessed)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Linear Regression Mean Absolute Error : {mae} \n")
print(f"Linear Regression Mean Squared Error : {mse} \n")
print(f"Linear Regression Root Mean Squared Error : {rmse} \n")
'''

# Random Forest Regression Model

rand_forest_reg = RandomForestRegressor(n_estimators=100, random_state=1, oob_score=True)
rand_forest_reg.fit(X_train_preprocessed, y_train)

'''
reg_score = rand_forest_reg.score(X_test_preprocessed, y_test)
print(f"Random Forest Regression Score: {reg_score} \n")
y_pred = rand_forest_reg.predict(X_test_preprocessed)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Random Forest Regression Mean Absolute Error : {mae} \n")
print(f"Random Forest Regression Mean Squared Error : {mse} \n")
print(f"Random Forest Regression Root Mean Squared Error : {rmse} \n")
'''

dump(pipeline, 'preprocessing_pipeline.joblib')
dump(rand_forest_reg, 'random_forest_regression_model.joblib')


'''
Some notes :

    LINEAR REGRESSION : 
    
        linear regression score with out 'club_name' ≈ low 0.7's, three values obtained = 0.7179300384282666, 0.7248100414085079, 0.7129789154860866
        linear regression score with 'club_name' ≈ upper mid 0.7's, three values obtained = 0.7679399993505359, 0.7583700650541587, 0.7701310104770249
            - including club name, despite high number of values, improves quality of data
        
        Linear Regression Mean Absolute Error : 1855836.9316803238 

        Linear Regression Mean Squared Error : 13527201291774.797 

        Linear Regression Root Mean Squared Error : 3677934.378394318 
        
    RANDOM FOREST REGRESSION SCORE :

        Random Forest Regression Score: 0.9997288103706363 

        Random Forest Regression Mean Absolute Error : 9952.929046884045 

        Random Forest Regression Mean Squared Error : 15838758848.338318 

        Random Forest Regression Root Mean Squared Error : 125852.13088517141
'''