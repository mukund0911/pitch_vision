# pitch_vision

soccer scouting from tactical cam video. wip.

goal: take a full-match tactical cam clip and spit out per-player stats + movement intent.

rough plan:
- detect players/ball per frame
- track them across frames
- map pixels → pitch coords via homography
- derive on-ball events (pass, shot, dribble)
- train a small transformer over tracking tokens for off-ball intent
