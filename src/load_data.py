from pathlib import Path
import pandas as pd
from statsbombpy import sb

COLS = ["match_id", "location", "shot_body_part", "shot_type", "play_pattern",
        "under_pressure", "shot_first_time", "shot_outcome", "shot_statsbomb_xg"]


def load_shots(max_matches=None):
    match_ids = []
    for _, c in sb.competitions().iterrows():
        m = sb.matches(competition_id=c.competition_id, season_id=c.season_id)
        match_ids += list(m.match_id)
    if max_matches:
        match_ids = match_ids[:max_matches]

    shots = []
    for i, match_id in enumerate(match_ids):
        try:
            ev = sb.events(match_id=match_id)
        except Exception as e:
            print("Hoppet over", match_id, e)
            continue
        s = ev[ev["type"] == "Shot"].copy()
        s["match_id"] = match_id
        s = s.reindex(columns=COLS)        # sikrer at alle kolonner finnes
        s["x"] = s["location"].str[0]
        s["y"] = s["location"].str[1]
        shots.append(s.drop(columns="location"))
        if i % 50 == 0:
            print(f"{i}/{len(match_ids)} kamper")
    return pd.concat(shots, ignore_index=True)


if __name__ == "__main__":
    df = load_shots(max_matches=None)       # sett None for alle kamper
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/shots.csv", index=False)
    print("Lagret", len(df), "skudd")