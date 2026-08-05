from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from feature_engineering import create_features

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Load encoders
encoders = joblib.load("encoders.pkl")

# Load dataset
data = pd.read_csv("data/results.csv")
data = data.sort_values("date").reset_index(drop=True)

# Dropdown data
teams = sorted(list(set(data["home_team"]).union(set(data["away_team"]))))
tournaments = sorted(data["tournament"].dropna().unique())
cities = sorted(data["city"].dropna().unique())
countries = sorted(data["country"].dropna().unique())


@app.route("/")
def home():
    return render_template(
        "index.html",
        teams=teams,
        tournaments=tournaments,
        cities=cities,
        countries=countries
    )


@app.route("/predict", methods=["POST"])
def predict():

    home_team = request.form["homeTeam"]
    away_team = request.form["awayTeam"]
    tournament = request.form["tournament"]
    city = request.form["city"]
    country = request.form["country"]

    neutral = request.form["neutralVenue"] == "yes"

    # Create all features
    features = create_features(
        data,
        home_team,
        away_team,
        tournament,
        city,
        country,
        neutral
    )

    input_data = pd.DataFrame([features])

    # Encode categorical columns
    input_data["home_team"] = encoders["home_team"].transform(
        input_data["home_team"]
    )

    input_data["away_team"] = encoders["away_team"].transform(
        input_data["away_team"]
    )

    input_data["tournament"] = encoders["tournament"].transform(
        input_data["tournament"]
    )

    input_data["city"] = encoders["city"].transform(
        input_data["city"]
    )

    input_data["country"] = encoders["country"].transform(
        input_data["country"]
    )

    # Arrange columns exactly as training data
    input_data = input_data[
        [
            "home_team",
            "away_team",
            "tournament",
            "city",
            "country",
            "neutral",
            "home_avg_goals",
            "away_avg_goals",
            "home_last5_avg_goals",
            "away_last5_avg_goals",
            "home_win_percentage",
            "away_win_percentage",
            "year",
            "month",
            "day",
            "dayofweek",
        ]
    ]

    # Prediction
    prediction = model.predict(input_data)[0]
    
    if prediction == "home_win":
        prediction = f"{home_team} Wins"

    elif prediction == "away_win":
        prediction = f"{away_team} Wins"

    else:
        prediction = "Draw"
    probabilities = model.predict_proba(input_data)[0]
    confidence = np.max(model.predict_proba(input_data)) * 100
    

    return render_template(
        "index.html",
        teams=teams,
        tournaments=tournaments,
        cities=cities,
        countries=countries,
        prediction=prediction,
        confidence=round(confidence, 2),
        home_team=home_team,
        away_team=away_team,
    )


if __name__ == "__main__":
    app.run(debug=True)