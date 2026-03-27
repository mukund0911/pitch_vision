"""
ScoutingAgent — natural-language query layer over a finalized MatchAccumulator.

Handles a small set of structured queries with regex. Anything it can't parse
falls back to an LLM call (if a client is provided); otherwise returns a hint.
"""

import json
import re
from typing import Optional

from pitchvision.inference.accumulator import MatchAccumulator


STAT_ALIASES = {
    "passes": "passes_attempted",
    "pass": "passes_attempted",
    "shots": "shots",
    "shot": "shots",
    "dribbles": "dribbles",
    "dribble": "dribbles",
    "receptions": "receptions",
    "sprints": "sprints",
    "presses": "pressing_actions",
    "pressing": "pressing_actions",
    "support": "support_runs",
}


class ScoutingAgent:
    def __init__(self, accumulator: MatchAccumulator, llm_client=None,
                 llm_model: str = "claude-sonnet-4-5"):
        self.accumulator = accumulator
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.profiles = accumulator.finalize()

    def query(self, question: str) -> str:
        answer = self._try_structured(question)
        if answer is not None:
            return answer
        if self.llm_client is None:
            return ("I couldn't parse that as a structured query and no LLM "
                    "client is configured.")
        return self._llm_query(question)

    # ------------------------------------------------- structured queries
    def _try_structured(self, question: str) -> Optional[str]:
        q = question.lower()

        # "how many <stat> did (player) #X make/have?"
        m = re.search(r"how many (\w+).*?#?\s*(\d+)", q)
        if m:
            stat = self._resolve_stat(m.group(1))
            player_id = int(m.group(2))
            if stat is None:
                return None
            profile = self.profiles.get(player_id)
            if profile is None:
                return f"No player with id {player_id} in the match data."
            val = getattr(profile, stat)
            return f"Player #{player_id} had {val} {m.group(1)}."

        # "who had/made the most <stat>?"
        m = re.search(r"who .*?most (\w+)", q)
        if m:
            stat = self._resolve_stat(m.group(1))
            if stat is None:
                return None
            best_id, best_val = None, -1
            for pid, prof in self.profiles.items():
                val = getattr(prof, stat, 0)
                if val > best_val:
                    best_val, best_id = val, pid
            if best_id is None:
                return "No players in the match data."
            return f"Player #{best_id} with {best_val} {m.group(1)}."

        return None

    @staticmethod
    def _resolve_stat(word: str) -> Optional[str]:
        word = word.rstrip("s") + "s"  # crude plural normalize
        # Try direct + aliases
        if word in STAT_ALIASES:
            return STAT_ALIASES[word]
        if word[:-1] in STAT_ALIASES:
            return STAT_ALIASES[word[:-1]]
        return None

    # ------------------------------------------------- LLM fallback
    def _llm_query(self, question: str) -> str:
        context = {
            "profiles": {pid: vars(p) for pid, p in self.profiles.items()},
            "recent_events": self.accumulator.event_log()[-50:],
        }
        prompt = (
            "You are an elite football scout answering questions from match "
            "data. Use only the data provided. Be concise and cite numbers.\n\n"
            f"Data:\n{json.dumps(context, default=str, indent=2)}\n\n"
            f"Question: {question}"
        )
        response = self.llm_client.messages.create(
            model=self.llm_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
