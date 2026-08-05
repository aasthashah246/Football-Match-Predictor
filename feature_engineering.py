import pandas as pd
import numpy as np


def create_features(
    data,
    home_team,
    away_team,
    tournament,
    city,
    country,
    neutral
):
    """
    Creates all features required by the trained model.
    """

    # -----------------------------
    # Date features
    # -----------------------------
    today = pd.Timestamp.today()

    year = today.year
    month = today.month
    day = today.day
    dayofweek = today.dayofweek

    # -----------------------------
    # Historical Average Goals
    # -----------------------------
    team_stats = {}

    for _, row in data.iterrows():

        team = row["home_team"]

        if team not in team_stats:
            team_stats[team] = {"goals": 0, "matches": 0}

        team_stats[team]["goals"] += row["home_score"]
        team_stats[team]["matches"] += 1

        team = row["away_team"]

        if team not in team_stats:
            team_stats[team] = {"goals": 0, "matches": 0}

        team_stats[team]["goals"] += row["away_score"]
        team_stats[team]["matches"] += 1

    if home_team in team_stats and team_stats[home_team]["matches"] > 0:
        home_avg_goals = (
            team_stats[home_team]["goals"]
            / team_stats[home_team]["matches"]
        )
    else:
        home_avg_goals = 0

    if away_team in team_stats and team_stats[away_team]["matches"] > 0:
        away_avg_goals = (
            team_stats[away_team]["goals"]
            / team_stats[away_team]["matches"]
        )
    else:
        away_avg_goals = 0

    # -----------------------------
    # Last 5 Average Goals
    # -----------------------------
    home_matches = data[
        (data["home_team"] == home_team) |
        (data["away_team"] == home_team)
    ].tail(5)

    home_goals = []

    for _, row in home_matches.iterrows():

        if row["home_team"] == home_team:
            home_goals.append(row["home_score"])
        else:
            home_goals.append(row["away_score"])

    if len(home_goals) > 0:
        home_last5_avg_goals = np.mean(home_goals)
    else:
        home_last5_avg_goals = 0

    away_matches = data[
        (data["home_team"] == away_team) |
        (data["away_team"] == away_team)
    ].tail(5)

    away_goals = []

    for _, row in away_matches.iterrows():

        if row["home_team"] == away_team:
            away_goals.append(row["home_score"])
        else:
            away_goals.append(row["away_score"])

    if len(away_goals) > 0:
        away_last5_avg_goals = np.mean(away_goals)
    else:
        away_last5_avg_goals = 0

    # -----------------------------
    # Win Percentage
    # -----------------------------
    home_matches = data[
        (data["home_team"] == home_team) |
        (data["away_team"] == home_team)
    ]

    home_wins = 0

    for _, row in home_matches.iterrows():

        if (
            row["home_team"] == home_team
            and row["home_score"] > row["away_score"]
        ):
            home_wins += 1

        elif (
            row["away_team"] == home_team
            and row["away_score"] > row["home_score"]
        ):
            home_wins += 1

    if len(home_matches) > 0:
        home_win_percentage = home_wins / len(home_matches)
    else:
        home_win_percentage = 0

    away_matches = data[
        (data["home_team"] == away_team) |
        (data["away_team"] == away_team)
    ]

    away_wins = 0

    for _, row in away_matches.iterrows():

        if (
            row["home_team"] == away_team
            and row["home_score"] > row["away_score"]
        ):
            away_wins += 1

        elif (
            row["away_team"] == away_team
            and row["away_score"] > row["home_score"]
        ):
            away_wins += 1

    if len(away_matches) > 0:
        away_win_percentage = away_wins / len(away_matches)
    else:
        away_win_percentage = 0

    # -----------------------------
    # Return Features
    # -----------------------------
    return {
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "city": city,
        "country": country,
        "neutral": neutral,
        "home_avg_goals": home_avg_goals,
        "away_avg_goals": away_avg_goals,
        "home_last5_avg_goals": home_last5_avg_goals,
        "away_last5_avg_goals": away_last5_avg_goals,
        "home_win_percentage": home_win_percentage,
        "away_win_percentage": away_win_percentage,
        "year": year,
        "month": month,
        "day": day,
        "dayofweek": dayofweek
    }