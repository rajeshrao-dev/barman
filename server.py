#!/usr/bin/env python3
"""Cocktail Recipe Agent backend.

Receives selections from the Cocktail Recipe page, asks Claude (with web search)
for recipes in the agreed JSON schema, checks the reply, and returns it.

Standard library only.
  export ANTHROPIC_API_KEY=...
  python3 server.py              # serves POST /api/recipes on 127.0.0.1:8787
  python3 server.py --selftest   # offline checks with a stubbed model call
"""
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("COCKTAIL_MODEL", "claude-haiku-5-5")
API_URL = "https://api.anthropic.com/v1/messages"
PORT = int(os.environ.get("PORT", "8787"))
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
# Verify this tool version against the current Anthropic docs before deploying.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}

SPIRITS = {"gin", "vodka", "white-rum", "bourbon", "tequila", "mezcal", "scotch", "brandy"}
INGREDIENTS = {"lime", "lemon", "mint", "sugar", "bitters", "orange", "tonic", "soda",
               "ginger-beer", "ginger-ale", "coke", "sprite", "fanta", "egg", "cucumber", "salt"}
STYLES = {"any", "dry", "balanced", "sweet"}
STRENGTHS = {"any", "light", "medium", "strong"}
OCCASIONS = {"any", "brunch", "party", "nightcap", "hot"}

SYSTEM_PROMPT = """You are the Cocktail Recipe Agent. You receive a request with:
- spirits: one or more base spirits
- ingredients: what the user has on hand
- style: any | dry | balanced | sweet
- strength: any | light | medium | strong
- occasion: any | brunch | party | nightcap | hot
- restrictions: noSugar, noEgg, lowAbv (booleans)

For each spirit, return up to 2 recipes. Keep the whole reply short: at most 6 method steps per recipe,
and keep "notes", "changes_from_source", and "why_it_works" under 25 words each:
1. Scouted: an established recipe found by web search that fits the request.
   Adapt it to the ingredients on hand and say what changed. Give its source URL.
2. Invented: one original recipe from the spirit and the user's ingredients,
   with a one-line reason for the pairings.

Rules:
- Use only the user's ingredients, plus water, ice, and these pantry staples:
  simple syrup, salt, bitters. Flag every staple you add with "pantry": true.
- Never break a restriction. noSugar excludes sugar, syrups, and sugary mixers
  (Coke, Sprite, Fanta, ginger ale, ginger beer). Sugar-free sweeteners are allowed
  only if the source says so. noEgg excludes egg white. lowAbv excludes strong drinks.
- If the restrictions leave little to work with, say so plainly in "notes".
- Measure in US oz and ml. Each recipe balances spirit, a sour or sweet modifier,
  and a dilutor or aromatic.
- Do not copy source text beyond a few words.
- Return ONLY a JSON object, with no prose before or after it, in this shape:
{"results":[{"spirit":"string","recipes":[{"name":"string","origin":"scouted|invented",
"source":"URL or null","glass":"string","ice":"string","style":"dry|balanced|sweet",
"strength":"light|medium|strong","ingredients":[{"text":"string","optional":false,"pantry":false}],
"method":["string"],"garnish":"string","changes_from_source":"string or null",
"why_it_works":"string or null"}]}],"notes":"string"}
"""

FORBIDDEN = {
    "noSugar": ["sugar", "syrup", "coke", "sprite", "fanta", "ginger ale", "ginger beer"],
    "noEgg": ["egg"],
}


def validate_request(body):
    """Return (clean_request, None) or (None, error_message)."""
    if not isinstance(body, dict):
        return None, "Body must be a JSON object."
    spirits = body.get("spirits")
    if not isinstance(spirits, list) or not spirits or not set(spirits) <= SPIRITS:
        return None, "spirits must be a non-empty list of known spirits."
    ings = body.get("ingredients", [])
    if not isinstance(ings, list) or not set(ings) <= INGREDIENTS:
        return None, "ingredients must be a list of known ingredients."
    for key, allowed in (("style", STYLES), ("strength", STRENGTHS), ("occasion", OCCASIONS)):
        value = body.get(key, "any")
        if value not in allowed:
            return None, f"{key} must be one of {sorted(allowed)}."
    restrictions = body.get("restrictions", {})
    if not isinstance(restrictions, dict):
        return None, "restrictions must be an object."
    clean_restr = {k: bool(restrictions.get(k, False)) for k in ("noSugar", "noEgg", "lowAbv")}
    return {
        "spirits": sorted(set(spirits)),
        "ingredients": sorted(set(ings)),
        "style": body.get("style", "any"),
        "strength": body.get("strength", "any"),
        "occasion": body.get("occasion", "any"),
        "restrictions": clean_restr,
    }, None


def call_model(system, user_text):
    """Send one request to the Messages API with web search and return the text reply."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    messages = [{"role": "user", "content": user_text}]
    for _ in range(4):  # web search can pause mid-turn; resume until the reply is complete
        payload = {
            "model": MODEL,
            "max_tokens": 16000,
            "system": system,
            "tools": [WEB_SEARCH_TOOL],
            "messages": messages,
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        content = data.get("content", [])
        if data.get("stop_reason") == "pause_turn":
            messages.append({"role": "assistant", "content": content})
            continue
        return "".join(b.get("text", "") for b in content if b.get("type") == "text")
    return ""


def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Model reply contained no JSON object.")
    return json.loads(text[start:end + 1])


def check_schema(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("results"), list):
        raise ValueError("Reply is missing a results list.")
    for res in obj["results"]:
        if not isinstance(res.get("spirit"), str) or not isinstance(res.get("recipes"), list):
            raise ValueError("Each result needs a spirit and a recipes list.")
        for rec in res["recipes"]:
            if not isinstance(rec.get("name"), str) or rec.get("origin") not in {"scouted", "invented"}:
                raise ValueError("Each recipe needs a name and a valid origin.")
            if not isinstance(rec.get("ingredients"), list) or not isinstance(rec.get("method"), list):
                raise ValueError("Each recipe needs ingredients and a method.")
            if rec["origin"] == "scouted" and not (isinstance(rec.get("source"), str)
                                                    and rec["source"].startswith("http")):
                raise ValueError("A scouted recipe must include a source URL.")
    return obj


def enforce_restrictions(obj, restrictions):
    """Drop any recipe that breaks a restriction, and say so in the notes."""
    dropped = 0
    for res in obj["results"]:
        kept = []
        for rec in res["recipes"]:
            text = " ".join(i.get("text", "").lower() for i in rec["ingredients"])
            bad = any(restrictions.get(r) and any(term in text and "sugar-free" not in text
                                                  and "sugar free" not in text for term in terms)
                      for r, terms in FORBIDDEN.items())
            if restrictions.get("lowAbv") and rec.get("strength") == "strong":
                bad = True
            if bad:
                dropped += 1
            else:
                kept.append(rec)
        res["recipes"] = kept
    if dropped:
        obj["notes"] = (obj.get("notes", "") + f" {dropped} recipe(s) removed for breaking a restriction.").strip()
    return obj


def generate(request, call=call_model):
    user_text = "Request:\n" + json.dumps(request, indent=2)
    try:
        obj = extract_json(call(SYSTEM_PROMPT, user_text))
    except ValueError:  # one retry with a stricter instruction
        retry = user_text + "\n\nReply with only the JSON object described in the instructions. No other text."
        obj = extract_json(call(SYSTEM_PROMPT, retry))
    obj = check_schema(obj)
    return enforce_restrictions(obj, request["restrictions"])


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_HEAD(self):
        # Health checks send HEAD requests; answer them so the platform sees the app as up.
        self.send_response(200 if self.path in ("/", "/index.html") else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            return self._send(404, {"error": "Not found."})
        page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        with open(page, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/recipes":
            return self._send(404, {"error": "Not found."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16_384:
                return self._send(413, {"error": "Request too large."})
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "Body must be valid JSON."})
        request, err = validate_request(body)
        if err:
            return self._send(400, {"error": err})
        try:
            return self._send(200, generate(request))
        except Exception as exc:  # surface a clear error instead of a bare 500
            return self._send(502, {"error": f"Agent failed: {exc}"})


def selftest():
    fixture = json.dumps({"results": [{"spirit": "Mezcal", "recipes": [
        {"name": "Mezcal Mule", "origin": "scouted", "source": "https://example.com/mule",
         "glass": "Copper mug", "ice": "Cubed ice", "style": "dry", "strength": "light",
         "ingredients": [{"text": "2 oz mezcal", "optional": False, "pantry": False},
                         {"text": "4 oz soda water", "optional": False, "pantry": False}],
         "method": ["Build over ice."], "garnish": "Lime wheel"},
        {"name": "Coke Smoke", "origin": "invented", "source": None, "glass": "Highball",
         "ice": "Cubed ice", "style": "sweet", "strength": "light",
         "ingredients": [{"text": "2 oz mezcal", "optional": False, "pantry": False},
                         {"text": "4 oz Coke", "optional": False, "pantry": False}],
         "method": ["Build over ice."], "garnish": "Lime wedge"}]}],
        "notes": ""})

    def stub(system, user_text):
        assert "Request:" in user_text and "SYSTEM" not in user_text
        return "Here you go:\n" + fixture

    req, err = validate_request({"spirits": ["mezcal"], "ingredients": ["ginger-ale", "lime"],
                                 "restrictions": {"noSugar": True}})
    assert err is None, err
    out = generate(req, call=stub)
    names = [r["name"] for r in out["results"][0]["recipes"]]
    assert names == ["Mezcal Mule"], names            # Coke recipe dropped under noSugar
    assert "removed" in out["notes"]

    assert validate_request({"spirits": ["rum-ish"]})[1] is not None
    assert validate_request({"spirits": ["gin"], "style": "bitter"})[1] is not None

    bad = json.dumps({"results": [{"spirit": "Gin", "recipes": [
        {"name": "X", "origin": "scouted", "source": None, "ingredients": [], "method": []}]}]})
    try:
        generate(req, call=lambda s, u: bad)
        raise AssertionError("scouted recipe without source should fail")
    except ValueError:
        pass
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        # On Render, PORT is always set, so listen on all addresses unless HOST says otherwise.
        host = os.environ.get("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
        print(f"Listening on http://{host}:{PORT}/")
        ThreadingHTTPServer((host, PORT), Handler).serve_forever()
