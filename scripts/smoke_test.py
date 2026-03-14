import sys

print("Step 1: importing api...", flush=True)
from nhl_predictor import api
print("Step 2: api imported OK", flush=True)

print("Step 3: fetching schedule for 2025-01-01...", flush=True)
games = api.get_games_for_date("2025-01-01")
print(f"Step 4: got {len(games)} games", flush=True)

if games:
    g = games[0]
    print(f"Step 5: fetching first goalscorer for game {g['game_id']}...", flush=True)
    scorer = api.get_first_goalscorer(g["game_id"])
    print(f"Step 6: scorer = {scorer}", flush=True)
else:
    print("Step 5: no games on that date, skipping goalscorer test", flush=True)

print("Smoke test PASSED", flush=True)
