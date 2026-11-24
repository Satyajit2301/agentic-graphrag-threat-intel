"""A small client for OpenAI-style chat APIs (Groq in the deployed app, Ollama or any compatible server locally).

The notebook used the same request format (Block 9 / Block 12). Differences for the live app:
  * rate limits (HTTP 429) are waited out for a short time and then reported in plain words,
  * there is no Drive cache; answers are not stored.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

GROQ_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"      # the same model family that was evaluated locally (gpt-oss:20b)


class LLMError(Exception):
    """a call failed in a way the user should be told about"""


class QuotaExhausted(LLMError):
    """the daily token budget of the free plan is used up"""


class LLM:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None, effort: str | None = "low"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self.base = (base_url or os.environ.get("LLM_BASE_URL", GROQ_BASE)).rstrip("/")
        self.effort = effort                  # reasoning effort of gpt-oss models: "low" keeps the token bill small
        self.max_wait = 45                    # longest single wait for a rate limit, in seconds

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or "localhost" in self.base or "127.0.0.1" in self.base

    def chat(self, messages, tools=None, max_tokens=700, temperature=0.0, _retries=4) -> dict:
        """one call -> {'content', 'tool_calls', 'finish', 'pt', 'ct', 'lat', 'error'}"""
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        if self.effort and "gpt-oss" in self.model:
            body["reasoning_effort"] = self.effort
        headers = {"Authorization": f"Bearer {self.api_key or 'none'}", "Content-Type": "application/json"}
        err = ""
        for attempt in range(_retries + 1):
            t0 = time.time()
            try:
                resp = requests.post(f"{self.base}/chat/completions", headers=headers, json=body, timeout=120)
            except requests.RequestException as e:
                err = f"network problem ({type(e).__name__})"
                time.sleep(min(10, 2 * (attempt + 1)))
                continue
            lat = time.time() - t0
            if resp.status_code == 200:
                j = resp.json()
                ch = j["choices"][0]
                m = ch["message"]
                u = j.get("usage", {})
                rec = {"content": m.get("content") or "", "tool_calls": m.get("tool_calls") or None, "finish": ch.get("finish_reason"),
                       "pt": int(u.get("prompt_tokens", 0)), "ct": int(u.get("completion_tokens", 0)), "lat": round(lat, 2), "error": ""}
                if rec["finish"] == "length" and not rec["content"].strip() and not rec["tool_calls"] and max_tokens < 2400:
                    return self.chat(messages, tools, max_tokens * 2, temperature, _retries)     # the model used all tokens for thinking: ask again with more room
                return rec
            msg = ""
            try:
                msg = resp.json().get("error", {}).get("message", "")
            except Exception:
                msg = resp.text[:300]
            if resp.status_code == 401:
                raise LLMError("The API key was rejected (HTTP 401). Check GROQ_API_KEY in the app secrets.")
            if resp.status_code == 429:
                if re.search(r"per day|\bTPD\b|\bRPD\b", msg):
                    raise QuotaExhausted("The daily token limit of the free LLM plan is used up. Try again later, or use a recorded example.")
                wait = self._wait_from(resp, msg)
                if attempt >= _retries or wait > self.max_wait:
                    raise LLMError(f"The LLM service is rate-limited (free plan). Wait about {int(wait)} seconds and ask again.")
                time.sleep(wait + 0.5)
                continue
            if resp.status_code == 400 and "reasoning_effort" in msg and "reasoning_effort" in body:
                body.pop("reasoning_effort"); self.effort = None
                continue
            if resp.status_code >= 500 or resp.status_code in (408, 409):
                err = f"server problem (HTTP {resp.status_code})"
                time.sleep(min(10, 2 * (attempt + 1)))
                continue
            # 400-type errors such as a tool call the server could not parse: report to the caller, it may retry with another temperature
            return {"content": "", "tool_calls": None, "finish": "error", "pt": 0, "ct": 0, "lat": round(lat, 2), "error": f"HTTP {resp.status_code}: {msg[:200]}"}
        raise LLMError(f"The LLM service could not be reached ({err or 'unknown error'}). Check the internet connection or try again.")

    @staticmethod
    def _wait_from(resp, msg: str) -> float:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
        m = re.search(r"try again in ([\d.]+)\s*(ms|s|m)", msg or "")
        if m:
            v = float(m.group(1))
            return v / 1000 if m.group(2) == "ms" else v * 60 if m.group(2) == "m" else v
        return 10.0
