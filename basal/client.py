"""Minimal Python client.

  from basal.client import Basal
  b = Basal("http://127.0.0.1:8000")
  b.choice(state, "Which department should handle this?", {"billing": "Billing", "tech": "Technical support"})
"""
import httpx


class Basal:
    def __init__(self, url="http://127.0.0.1:8000", timeout=60, early_exit=None, facts=None, api_key=None):
        self.url, self.early_exit, self.facts = url.rstrip("/") + "/v1/systemone", early_exit, facts
        self.http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    def decide(self, state, questions):
        body = {"state": state, "questions": questions}
        if self.early_exit:
            body["early_exit"] = self.early_exit
        if self.facts:
            body["facts"] = self.facts  # "auto": computed calendar / arithmetic facts appended to the state
        r = self.http.post(self.url, json=body)
        if r.status_code != 200:
            raise RuntimeError(r.text)
        return r.json()["answers"]

    def choice(self, state, question, criteria):
        return self.decide(state, {"q": {"type": "choice", "instructions": question, "criteria": criteria}})["q"]

    def yes_no(self, state, question, true=None, false=None):
        crit = {k: v for k, v in (("true", true), ("false", false)) if v}
        return self.decide(state, {"q": {"type": "noul", "instructions": question, "criteria": crit}})["q"]

    def score(self, state, question, levels):
        return self.decide(state, {"q": {"type": "score", "instructions": question, "criteria": levels}})["q"]
