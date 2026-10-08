"""Tiny wrapper around the FREE Groq API (OpenAI-compatible).
Set MOCK=1 in your environment to test the whole pipeline without any API key."""
import os, re, json, time, random, requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://api.groq.com/openai/v1/chat/completions"
GEN_MODEL = os.getenv("GEN_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
MOCK = os.getenv("MOCK") == "1"
PAUSE = float(os.getenv("PAUSE_SECONDS", "2"))  # small pause to respect free-tier limits


def chat(messages, model, temperature=0.0, max_tokens=600, json_mode=False):
    if MOCK:
        return _mock(messages, json_mode)
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "paste_your_key_here":
        raise SystemExit("GROQ_API_KEY missing. Copy .env.example to .env and paste your free key.")
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if "gpt-oss" in model:          # reasoning models: keep thinking short + leave room for the answer
        body["reasoning_effort"] = "low"
        body["max_tokens"] = max(max_tokens, 2000)
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(6):
        r = requests.post(API_URL, json=body, timeout=90,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 200:
            time.sleep(PAUSE)
            return (r.json()["choices"][0]["message"].get("content") or "").strip()
        if r.status_code in (429, 500, 502, 503):      # rate limit / temporary error -> wait, retry
            wait = float(r.headers.get("retry-after", 5 * (attempt + 1)))
            print(f"   (API busy [{r.status_code}], waiting {wait:.0f}s...)")
            time.sleep(wait + 1)
            continue
        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("API kept failing after 6 tries")


def parse_json(text):
    """Judge models sometimes wrap JSON in extra words - pull out the JSON part."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def _mock(messages, json_mode):
    """Fake answers so you can test the code with no internet / no key."""
    if not json_mode:
        return "MOCK ANSWER: this is a fake answer used only for testing."
    p = messages[-1]["content"]
    if '"claims"' in p:
        return json.dumps({"claims": [{"claim": "c1", "supported": True},
                                      {"claim": "c2", "supported": random.random() > .4}],
                           "reason": "mock"})
    if '"relevant"' in p:
        n = len(re.findall(r"\[Chunk \d+\]", p))
        return json.dumps({"relevant": [random.random() > .5 for _ in range(n)], "reason": "mock"})
    return json.dumps({"score": round(random.random(), 2), "reason": "mock"})
