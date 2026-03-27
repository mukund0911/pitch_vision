"""
FastAPI server — REST + query endpoints over a loaded match.

Call set_match(accumulator) at startup to populate the globals, then the
endpoints work over that match.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pitchvision.inference.accumulator import MatchAccumulator
from server.agent import ScoutingAgent


app = FastAPI(title="PitchVision Scouting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


_state = {"accumulator": None, "agent": None}


def set_match(accumulator: MatchAccumulator, llm_client=None):
    _state["accumulator"] = accumulator
    _state["agent"] = ScoutingAgent(accumulator, llm_client=llm_client)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok", "match_loaded": _state["accumulator"] is not None}


@app.get("/players")
def list_players():
    acc = _state["accumulator"]
    if acc is None:
        return {"error": "no match loaded"}
    profiles = acc.finalize()
    return {str(pid): vars(p) for pid, p in profiles.items()}


@app.get("/players/{player_id}")
def get_player(player_id: int):
    acc = _state["accumulator"]
    if acc is None:
        return {"error": "no match loaded"}
    return acc.player(player_id)


@app.get("/events")
def get_events(event_type: str = None, player_id: int = None):
    acc = _state["accumulator"]
    if acc is None:
        return {"error": "no match loaded"}
    events = acc.event_log()
    if event_type:
        events = [e for e in events if e["type"] == event_type]
    if player_id is not None:
        events = [e for e in events if e["player_id"] == player_id]
    return {"events": events, "count": len(events)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    agent = _state["agent"]
    if agent is None:
        return QueryResponse(answer="No match loaded.")
    return QueryResponse(answer=agent.query(req.question))
