import json
import sys

target = sys.argv[1]
with open("/tmp/schedule.json") as f:
    data = json.load(f)

games = []
for day in data.get("gameWeek", []):
    if day.get("date") == target:
        games = [g for g in day.get("games", []) if g.get("gameType", 2) in (2, 3)]

print(f"Games found: {len(games)}")
for g in games:
    print(f"  {g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}")
