# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OpenAI-compatible chat client s3 translates through (DeepSeek, Ollama,
LM Studio, OpenAI): one call with retry, the response_format each provider
accepts, tolerant JSON parsing, and the served-model check doctor runs.
What to ask (prompts, schemas) stays in s3."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from dubadabidu.core.config.schema import Translation

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger("dubadabidu.s3")   # same logger as before the move
# Some local servers (e.g. LM Studio) may wrap JSON in ```json fences in `text`
# mode; strip them before parsing.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _response_format(tcfg: Translation, schema: dict) -> dict:
    """OpenAI `response_format` arg. Providers diverge on structured-output support:
      json_object (default) — DeepSeek / OpenAI / Ollama
      json_schema           — LM Studio (strict; best for small local models)
      text                  — no server-side enforcement; relies on the prompt."""
    mode = tcfg.response_format
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "text":
        return {"type": "text"}
    if mode == "json_schema":
        return {"type": "json_schema",
                "json_schema": {"name": "translation", "strict": True,
                                "schema": schema}}
    raise SystemExit(f"translation.response_format '{mode}' is invalid "
                     "(use json_object | json_schema | text).")


def _loads(content: str) -> dict:
    return json.loads(_FENCE.sub("", content.strip()))


def chat(client: OpenAI, tcfg: Translation, sysmsg: str, user: str, retries: int = 4,
         *, schema: dict, model: str | None = None,
         seen: set[str] | None = None) -> dict:
    """One chat call with retry. `seen` collects the model id the server says
    ANSWERED (resp.model) — provider names are moving aliases (deepseek-flash
    moved checkpoint twice in 2026), so the requested name alone does not say
    which model produced a translation."""
    rformat = _response_format(tcfg, schema)
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model or tcfg.model, temperature=0.3, timeout=120,
                response_format=rformat,
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": user}])
            if seen is not None and getattr(resp, "model", None):
                seen.add(str(resp.model))
            return _loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == retries:
                raise
            log.warning("LLM call failed (%s); retry in %.0fs", e, delay)
            time.sleep(delay)
            delay *= 2


def configured_models(tcfg: Translation) -> list[str]:
    """Every model name the translation block will request."""
    names = [tcfg.model, tcfg.reflect_model, tcfg.adequacy_model]
    return sorted({n for n in names if n})


def missing_models(client: OpenAI, tcfg: Translation) -> list[str]:
    """Configured model names the endpoint does not list. Catches a provider
    retiring a name (deepseek-v4-flash, 2026-09-10) before a paid run silently
    gets routed to an unevaluated replacement — or fails mid-batch."""
    listed = {m.id for m in client.models.list().data}
    return [n for n in configured_models(tcfg) if n not in listed]
