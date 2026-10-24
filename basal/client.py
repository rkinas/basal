"""Minimal Python client.

  from basal.client import Basal
  b = Basal("http://127.0.0.1:8000")
  b.choice(state, "Which department should handle this?", {"billing": "Billing", "tech": "Technical support"})

It also talks to other servers of the same /v1/systemone request shape, such as vLLM's structured-diffusion example
(examples/features/structured_diffusion): `api_key` sends "Authorization: Bearer <key>", choice criteria given as a
list of keys are sent as {key: null} (what basal-serve reads a list as), and a noul answer that carries only "noul"
gets "probabilities" and "confidence" from it, as basal-serve computes them. The confidences of another server are
not calibrated by basal: basal's certified thresholds (max_error) do not apply to them.
"""
import httpx


def complete(answer):
    """A noul answer with only "noul" (p of true) -> also "probabilities" {"true", "false"} and "confidence"; every
    other answer (basal-serve's, or null for a question another server skipped) as it is."""
    if isinstance(answer, dict) and answer.get("type") == "noul" and "probabilities" not in answer:
        p = float(answer["noul"])
        answer = dict(answer, probabilities={"true": p, "false": 1.0 - p}, confidence=max(p, 1.0 - p))
    return answer


class Basal:
    def __init__(self, url="http://127.0.0.1:8000", timeout=60, early_exit=None, facts=None, transport=None,
                 api_key=None):
        self.url, self.early_exit, self.facts = url.rstrip("/") + "/v1/systemone", early_exit, facts
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.http = httpx.Client(timeout=timeout, transport=transport, headers=headers)

    def decide(self, state, questions):
        body = {"state": state, "questions": questions}
        if self.early_exit:
            body["early_exit"] = self.early_exit
        if self.facts:
            body["facts"] = self.facts  # "auto": computed calendar / arithmetic facts appended to the state
        r = self.http.post(self.url, json=body)
        if r.status_code != 200:
            raise RuntimeError(r.text)
        return {k: complete(a) for k, a in r.json()["answers"].items()}

    def choice(self, state, question, criteria):
        if isinstance(criteria, list) and all(isinstance(k, str) for k in criteria):
            criteria = dict.fromkeys(criteria)  # basal-serve reads a list of keys so; other servers need the map
        return self.decide(state, {"q": {"type": "choice", "instructions": question, "criteria": criteria}})["q"]

    def yes_no(self, state, question, true=None, false=None):
        crit = {k: v for k, v in (("true", true), ("false", false)) if v}
        return self.decide(state, {"q": {"type": "noul", "instructions": question, "criteria": crit}})["q"]

    def score(self, state, question, levels):
        return self.decide(state, {"q": {"type": "score", "instructions": question, "criteria": levels}})["q"]

    def preset(self, state, name, lang=None):
        """The answers to a ready question set (basal/presets.py: triage, moderation, compliance) about state."""
        from .presets import questions
        return self.decide(state, questions(name, lang, state))
