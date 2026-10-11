import numpy as np

GOAL_X, GOAL_Y = 120, 40          # StatsBomb-banen er 120 x 80
FEATURES = ["dist", "angle", "is_header", "set_piece", "under_pressure", "first_time"]


def add_features(df):
    df = df.copy()
    dx = GOAL_X - df["x"]
    dy = df["y"] - GOAL_Y
    df["dist"] = np.hypot(dx, dy)                                 # avstand til målmidten
    df["angle"] = np.arctan2(8 * dx, dx**2 + dy**2 - 16)          # vinkel mellom stolpene
    return df


def clean_shots(df):
    df = df[df["shot_type"] != "Penalty"].dropna(subset=["x", "y"]).copy()
    df["goal"] = (df["shot_outcome"] == "Goal").astype(int)
    df["is_header"] = (df["shot_body_part"] == "Head").astype(int)
    df["set_piece"] = df["play_pattern"].isin(["From Free Kick", "From Corner"]).astype(int)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["first_time"] = df["shot_first_time"].fillna(False).astype(int)
    return add_features(df)