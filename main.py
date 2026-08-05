import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

data = pd.read_csv("data/results.csv")
data = data.sort_values("date").reset_index(drop=True)
data["home_avg_goals"] = np.nan
data["away_avg_goals"] = np.nan
data["home_last5_avg_goals"] = np.nan
data["away_last5_avg_goals"] = np.nan
data["home_win_percentage"] = np.nan
data["away_win_percentage"] = np.nan

team_stats = {}
last5_goals = {}
win_stats = {}

for i in range(len(data)):

    home_team = data.loc[i, "home_team"]
    away_team = data.loc[i, "away_team"]
    
    if home_team not in win_stats:
        win_stats[home_team] = {"wins": 0, "matches": 0}
        
    if away_team not in win_stats:
        win_stats[away_team] = {"wins": 0, "matches": 0}
        
    if home_team not in last5_goals:
        last5_goals[home_team] = []
        
    if away_team not in last5_goals:
        last5_goals[away_team] = []

    if home_team not in team_stats:
        team_stats[home_team] = {"goals": 0, "matches": 0}

    if away_team not in team_stats:
        team_stats[away_team] = {"goals": 0, "matches": 0}

    if team_stats[home_team]["matches"] > 0:
        data.loc[i, "home_avg_goals"] = (
            team_stats[home_team]["goals"] /
            team_stats[home_team]["matches"]
        )

    if team_stats[away_team]["matches"] > 0:
        data.loc[i, "away_avg_goals"] = (
            team_stats[away_team]["goals"] /
            team_stats[away_team]["matches"]
        )
    if len(last5_goals[home_team]) > 0:
        data.loc[i, "home_last5_avg_goals"] = np.mean(last5_goals[home_team])
        
    if len(last5_goals[away_team]) > 0:
        data.loc[i, "away_last5_avg_goals"] = np.mean(last5_goals[away_team])

    if win_stats[home_team]["matches"] > 0:
        data.loc[i, "home_win_percentage"] = (win_stats[home_team]["wins"] / win_stats[home_team]["matches"])

    if win_stats[away_team]["matches"] > 0:
        data.loc[i, "away_win_percentage"] = (win_stats[away_team]["wins"] / win_stats[away_team]["matches"])

    team_stats[home_team]["goals"] += data.loc[i, "home_score"]
    team_stats[home_team]["matches"] += 1

    team_stats[away_team]["goals"] += data.loc[i, "away_score"]
    team_stats[away_team]["matches"] += 1
    
    last5_goals[home_team].append(data.loc[i, "home_score"])
    last5_goals[away_team].append(data.loc[i, "away_score"])
    
    win_stats[home_team]["matches"] += 1
    win_stats[away_team]["matches"] += 1
    
    if len(last5_goals[home_team]) > 5:
        last5_goals[home_team].pop(0)
        
    if len(last5_goals[away_team]) > 5:
        last5_goals[away_team].pop(0)
        
    if data.loc[i, "home_score"] > data.loc[i, "away_score"]:
        win_stats[home_team]["wins"] += 1
    
    elif data.loc[i, "home_score"] < data.loc[i, "away_score"]:
        win_stats[away_team]["wins"] += 1




data["result"] = np.where(data["home_score"] > data["away_score"], "home_win", np.where(data["home_score"] < data["away_score"], "away_win", "draw"))
#print(data.head())
#print(data.tail())
data["home_last5_avg_goals"] = data["home_last5_avg_goals"].fillna(0)
data["away_last5_avg_goals"] = data["away_last5_avg_goals"].fillna(0)
data["home_win_percentage"] = data["home_win_percentage"].fillna(0)
data["away_win_percentage"] = data["away_win_percentage"].fillna(0)
X=data.drop(columns=(["result","home_score","away_score"]))
#print(X.columns)
y=data["result"]
#print(X.dtypes)
X["date"] = pd.to_datetime(X["date"])
X["year"] = X["date"].dt.year
X["month"] = X["date"].dt.month
X["day"] = X["date"].dt.day
X["dayofweek"] = X["date"].dt.dayofweek

X = X.drop(columns=["date"])
#print(X.columns)
#print(X.head())
#print(data.columns)

home_team_encoder = LabelEncoder()
away_team_encoder = LabelEncoder()
tournament_encoder = LabelEncoder()
city_encoder = LabelEncoder()
country_encoder = LabelEncoder()

X["home_team"] = home_team_encoder.fit_transform(X["home_team"])
X["away_team"] = away_team_encoder.fit_transform(X["away_team"])
X["tournament"] = tournament_encoder.fit_transform(X["tournament"])
X["city"] = city_encoder.fit_transform(X["city"])
X["country"] = country_encoder.fit_transform(X["country"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100,max_depth=20 ,min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

cm = confusion_matrix(y_test, predictions)
#print(cm)
#print(y.value_counts())
#print(le.classes_)
#print(classification_report(y_test, predictions))


joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

encoders = {
    "home_team": home_team_encoder,
    "away_team": away_team_encoder,
    "tournament": tournament_encoder,
    "city": city_encoder,
    "country": country_encoder
}
joblib.dump(encoders, "encoders.pkl")
print("Encoders saved as encoders.pkl")
print('X.columns:.', X.columns.tolist())