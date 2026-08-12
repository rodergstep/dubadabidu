"""s3: LLM translation via OpenAI-compatible endpoint (DeepSeek / Ollama / OpenAI)
with retry+backoff, per-batch checkpointing, isometric constraint, glossary,
and n-best shorter variants. See prompts/translate_system.md."""
from __future__ import annotations
import threading
from concurrent.futures import ThreadPoolExecutor
import csv, json, logging, os, re, time
from pathlib import Path
from . import manifest as M

log = logging.getLogger("dubadabidu.s3")
LANG_NAMES = {"en": "English", "fr": "French", "de": "German",
              "es": "Spanish", "ru": "Russian", "pl": "Polish"}

# Some local servers (e.g. LM Studio) may wrap JSON in ```json fences in `text`
# mode; strip them before parsing.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")

# Strict schemas for the {"type":"json_schema"} response_format (LM Studio /
# OpenAI structured outputs). One per pass — draft/adapt emit translations,
# reflect emits critiques, terms emits the terminology base.
def _segments_schema(props: dict, required: list) -> dict:
    return {"type": "object",
            "properties": {"segments": {"type": "array", "items": {
                "type": "object", "properties": props, "required": required}}},
            "required": ["segments"]}


_JSON_SCHEMA = _segments_schema(
    {"id": {"type": "string"}, "text": {"type": "string"},
     "variants": {"type": "array", "items": {"type": "string"}}},
    ["id", "text", "variants"])
_REFLECT_SCHEMA = _segments_schema(
    {"id": {"type": "string"}, "issues": {"type": "string"}}, ["id", "issues"])
_ADEQUACY_SCHEMA = _segments_schema(
    {"id": {"type": "string"}, "score": {"type": "integer"},
     "issue": {"type": "string"}}, ["id", "score", "issue"])
_TERMS_SCHEMA = {
    "type": "object",
    "properties": {"terms": {"type": "array", "items": {
        "type": "object",
        "properties": {"uk": {"type": "string"}, "tr": {"type": "string"}},
        "required": ["uk", "tr"]}}},
    "required": ["terms"],
}
_SHORTEN_SCHEMA = {
    "type": "object",
    "properties": {"variants": {"type": "array", "items": {"type": "string"}}},
    "required": ["variants"],
}


def _response_format(tcfg: dict, schema: dict = _JSON_SCHEMA) -> dict:
    """OpenAI `response_format` arg. Providers diverge on structured-output support:
      json_object (default) — DeepSeek / OpenAI / Ollama
      json_schema           — LM Studio (strict; best for small local models)
      text                  — no server-side enforcement; relies on the prompt."""
    mode = tcfg.get("response_format", "json_object")
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


def _glossary(lang: str) -> str:
    p = Path("glossary") / f"{lang}.csv"
    if not p.exists():
        return ""
    with p.open(encoding="utf-8") as fh:
        rows = [f"  {r[0]} -> {r[1]}" for r in csv.reader(fh)
                if len(r) >= 2 and not r[0].startswith("#")]
    return ("Mandatory terminology (Ukrainian -> target):\n" + "\n".join(rows)) if rows else ""


def _prompt(lang: str, tol: float, n_var: int,
            name: str = "translate_system") -> str:
    tpl = Path(f"prompts/{name}.md").read_text(encoding="utf-8")
    return (tpl.replace("{LANG}", LANG_NAMES[lang])
               .replace("{TOL}", str(int(tol * 100)))
               .replace("{NVAR}", str(n_var)))


def _terms(client, tcfg, cfg: dict, video: str, man: dict, lang: str) -> str:
    """Video-specific terminology base, extracted once per video+lang (cached).
    Injected next to the static glossary so every batch translates domain terms
    and proper nouns identically — the document-level consistency lever."""
    cache = M.video_workdir(cfg, video) / f"terms_{lang}.json"
    if cache.exists():
        terms = json.loads(cache.read_text(encoding="utf-8"))["terms"]
    else:
        transcript = "\n".join(u["text_uk"] for u in man["utterances"])
        data = _chat(client, tcfg,
                     f"From this Ukrainian art-course transcript, extract up to "
                     f"30 domain terms and proper nouns whose {LANG_NAMES[lang]} "
                     f"translation must stay consistent throughout the course. "
                     f"Give your best dubbing-appropriate translation for each. "
                     f'Output STRICT JSON: {{"terms":[{{"uk":"...","tr":"..."}}]}}',
                     transcript, schema=_TERMS_SCHEMA)
        terms = data.get("terms", [])
        cache.write_text(json.dumps({"terms": terms}, ensure_ascii=False,
                                    indent=2), encoding="utf-8")
        log.info("%s: %d terms extracted -> %s", lang, len(terms), cache.name)
    if not terms:
        return ""
    return ("Video-specific terminology (Ukrainian -> target, mandatory):\n"
            + "\n".join(f"  {t['uk']} -> {t['tr']}" for t in terms))


def _avoid_block(cfg: dict, lang: str, mandatory: str) -> str:
    """Prompt section asking the model to route around words the TTS mis-stresses.

    THE ONE REMEDIATION THAT NEEDS NO NEW CAPABILITY. Automated Russian stress
    detection is closed (FINDINGS 2.1f-2.1h: four detectors, all at or under the
    AUC that WER already gives free) and so is selection (2.1i/2.1j: consensus
    is ANTI-correlated, because for some words qwen is reliably wrong and the
    majority placement is the wrong one). What is left is per-word control, and
    the cheapest form of it is not saying the word: the translation layer is
    ours, it already takes a mandatory glossary, and most of the marked words
    are ordinary vocabulary with ordinary synonyms.

    Measured against the first real review pass (46 ru segments, 13 marks):
    avoidance has a plausible synonym for 6-7 of the 9 marked words, where
    respelling reaches 4 and the `ё` lever reaches 0 (no marked word is missing
    a `ё`; the one that has it was mis-stressed anyway).

    A PREFERENCE, NOT A BAN, and the wording matters. A word with no natural
    synonym must survive untouched — otherwise this trades a stress error for a
    paraphrase nobody asked for, in a course where `натюрморт` is the subject.
    That trade is unmeasured: there is no evidence yet that the listener prefers
    a correctly-stressed synonym to a mis-stressed exact word, so the prompt is
    written to lose gracefully.

    PRECEDENCE IS STATED because the two sections contradict each other by
    construction: the glossary and the terms base both say "mandatory", and a
    mandatory term can also be mis-stressed. Terminology wins — consistency
    across a 20-video course outranks one word's stress — and a collision is
    logged rather than silently resolved.
    """
    tcfg = cfg["translation"]
    if not tcfg.get("avoid_mis_stressed", True):
        return ""
    from qc.stress_words import load_avoid_list
    words = load_avoid_list(lang, int(tcfg.get("avoid_min_marks", 1)))
    if not words:
        return ""
    log.info("%s: avoidance list active — %d word(s) from stress_lexicon_%s.json",
             lang, len(words), lang)
    clash = sorted({w for w, _ in words if w.lower() in mandatory.lower()})
    if clash:
        log.info("%s: %d avoid-list word(s) are mandatory terminology — "
                 "terminology wins, they will not be routed around: %s",
                 lang, len(clash), ", ".join(clash))
    listed = ", ".join(f"{w} ({n}x)" for w, n in words)
    # The three clauses below are not hedging — each one is a failure mode
    # MEASURED on a live A/B against these exact segments (2026-08-11):
    #   - it inflected instead of substituting: `натюрморте` -> `натюрморта`,
    #     `белилам` -> `белилами`. A different case of the same word is stressed
    #     on the same syllable, so that is not avoidance at all, it is the same
    #     defect wearing a different ending.
    #   - it DELETED the noun: "разместить самые светлые цвета" came back as
    #     "разместить самые светлые". Avoiding a word by dropping it is a
    #     content loss the adequacy judge might well pass, since nothing was
    #     mistranslated.
    #   - it rewrote words that were never on the list (`участков` -> `зон`),
    #     which is churn the reviewer has to re-read for no benefit.
    return (
        "Words the TTS voice mis-stresses (marked by a native listener; the "
        "count is how often).\n"
        "PREFER a natural synonym or rephrasing where one exists. This is a "
        "preference, not a rule:\n"
        "  - use a DIFFERENT WORD or restructure the phrase. Putting the same "
        "word in another case, number or tense is NOT a substitute — it is "
        "pronounced the same way;\n"
        "  - never drop the word without carrying its meaning, and never "
        "change the meaning, drop an instruction or invent terminology;\n"
        "  - if the mandatory terminology above fixes a word, KEEP IT — "
        "terminology wins;\n"
        "  - if no natural alternative exists, keep the word. An exact word "
        "beats an awkward paraphrase;\n"
        "  - change nothing else. Words not listed here must be translated "
        "exactly as you otherwise would.\n"
        f"  {listed}")


def _measured_cps(cfg: dict, lang: str, min_samples: int = 10) -> float | None:
    """Median TTS speaking rate (chars/second) measured from past synths of
    this language across all work/*/manifest.json. The prompt's char bound is
    relative to UKRAINIAN length — a poor proxy for how long the TARGET text
    takes to speak (German runs long, English short). A measured budget lets
    the LLM aim at the actual slot instead of guessing.

    Samples are filtered to the engine that will synthesize THIS run (edge
    voices pace differently from Chatterbox — Mac edge prototyping must not
    skew GPU chatterbox budgets). Legacy samples with no engine tag are used
    only as a fallback when the engine-matched pool is too small."""
    import statistics
    from .manifest import resolve_engine
    engine = resolve_engine(cfg["tts"], lang)
    rates, legacy = [], []
    for mp in Path(cfg["work_dir"]).glob("*/manifest.json"):
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for u in m.get("utterances", []):
            tr = u.get("tr", {}).get(lang, {})
            dur, text = tr.get("synth_dur"), tr.get("text", "")
            if dur and dur > 0.5 and len(text) >= 20:  # skip trivial segments
                se = tr.get("synth_engine")
                (rates if se == engine else legacy if se is None else []
                 ).append(len(text) / dur)
    if len(rates) < min_samples:
        rates += legacy
    if len(rates) < min_samples:
        return None
    return statistics.median(rates)


def shorten(cfg: dict, lang: str, text_uk: str, text: str, max_chars: int,
            n: int = 2) -> list[str]:
    """Emergency rewrite for s5 overflow: the pre-generated variants were all
    too long, so ask the LLM for n candidates under a HARD char budget derived
    from the measured synth duration. Returns [] when the endpoint is
    unavailable (no key / offline) — s5 then degrades to the overflow flag."""
    tcfg = cfg["translation"]
    key = os.environ.get(tcfg["api_key_env"])
    if not key:
        log.warning("%s not set — cannot request emergency shorter variants",
                    tcfg["api_key_env"])
        return []
    from openai import OpenAI
    client = OpenAI(base_url=tcfg["base_url"], api_key=key)
    sysmsg = (
        f"You shorten dubbing lines for a painting course voice-over.\n"
        f"Rewrite the given {LANG_NAMES[lang]} line so it keeps the exact "
        f"instructional meaning of the Ukrainian source but fits a tighter "
        f"time slot. Produce {n} candidates, EACH AT MOST {max_chars} "
        f"characters, progressively shorter. Drop filler and pleasantries "
        f"before dropping instruction. Natural spoken {LANG_NAMES[lang]}.\n"
        f'Output STRICT JSON: {{"variants":["...","..."]}}')
    user = json.dumps({"source_uk": text_uk, "current": text,
                       "max_chars": max_chars}, ensure_ascii=False)
    try:
        data = _chat(client, tcfg, sysmsg, user, retries=1,
                     schema=_SHORTEN_SCHEMA)
        return [v.strip() for v in data.get("variants", []) if v.strip()]
    except Exception as e:
        log.warning("emergency shorten failed (%s); keeping overflow", e)
        return []


def _chat(client, tcfg, sysmsg: str, user: str, retries: int = 4,
          schema: dict = _JSON_SCHEMA, model: str | None = None) -> dict:
    rformat = _response_format(tcfg, schema)
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model or tcfg["model"], temperature=0.3, timeout=120,
                response_format=rformat,
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": user}])
            return _loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == retries:
                raise
            log.warning("LLM call failed (%s); retry in %.0fs", e, delay)
            time.sleep(delay)
            delay *= 2


def _adequacy(ask, shared: str, batch: list, lang: str, flag: int,
              tcfg: dict) -> None:
    """LLM-judge each FINAL translation against its Ukrainian source; store
    {score 1-5, issue} in tr[lang]['adequacy'] and log sub-threshold segments.
    Never fatal — a judge failure must not abort a translated batch."""
    sysmsg = (
        f"You are a bilingual QA judge for Ukrainian -> {LANG_NAMES[lang]} "
        f"dubbing of a painting course. For each segment, compare the "
        f"{LANG_NAMES[lang]} translation ('tr') against the Ukrainian source "
        f"('uk') and rate FAITHFULNESS 1-5 (5 = complete, accurate, natural "
        f"dubbing; 3 = minor drift; 1 = wrong meaning or a dropped "
        f"instruction). Weigh mistranslation, omitted instructions, wrong "
        f"terminology, invented content — NOT length or word choice for timing. "
        f"'issue' = a short reason, or 'ok' when the score is 5.\n"
        f'Output STRICT JSON: {{"segments":[{{"id":"..","score":5,"issue":"ok"}}]}}'
        + (("\n\n" + shared) if shared else ""))
    payload = [{"id": u["id"], "uk": u["text_uk"], "tr": u["tr"][lang]["text"]}
               for u in batch]
    try:
        scored = ask(sysmsg, payload, schema=_ADEQUACY_SCHEMA,
                     model=tcfg.get("adequacy_model"))
    except Exception as e:                     # judge is best-effort
        log.warning("%s: adequacy judge failed (%s); skipping batch", lang, e)
        return
    flagged = []
    for u in batch:
        s = scored.get(u["id"])
        if not s:
            continue
        score = int(s.get("score", 0))
        u["tr"][lang]["adequacy"] = {"score": score,
                                     "issue": s.get("issue", "").strip()}
        if score < flag:
            flagged.append((u["id"], score, s.get("issue", "")))
    if flagged:
        log.warning("%s: %d/%d below adequacy %d: %s", lang, len(flagged),
                    len(batch), flag,
                    ", ".join(f"{i}({sc}:{iss[:40]})" for i, sc, iss in flagged))


def run(cfg: dict, video: str, langs: list[str]) -> None:
    from openai import OpenAI
    tcfg = cfg["translation"]
    key = os.environ.get(tcfg["api_key_env"])
    if not key:
        raise SystemExit(f"Set {tcfg['api_key_env']} in your environment "
                         f"(export {tcfg['api_key_env']}=sk-...).")
    client = OpenAI(base_url=tcfg["base_url"], api_key=key)
    man = M.load(cfg, video)

    passes = int(tcfg.get("passes", 1))
    # LANGUAGES RUN CONCURRENTLY. Measured 2026-08-01 on a 62-utterance lesson:
    # 32 minutes elapsed for 1.3 languages while the process used 3.5 SECONDS of
    # CPU — it is round-trip latency almost end to end, so five languages
    # sequentially is ~5x longer than it needs to be for no extra work done.
    # Threads, not processes: the work is I/O, and a second process would load
    # its own copy of `man` and clobber the first on save.
    # Each language writes only u["tr"][<its own lang>], so the segment data is
    # disjoint; `man["stages"]` and the save itself are NOT, hence _save_lock.
    # API calls stay OUTSIDE the lock or this would be sequential again.
    save_lock = threading.Lock()

    def _one_lang(lang: str) -> None:
        todo = [u for u in man["utterances"] if "text" not in u["tr"].get(lang, {})]
        if not todo:
            log.info("%s: cached", lang)
            return
        tol, nvar = tcfg["isometric_tolerance"], tcfg["n_short_variants"]
        terms = _terms(client, tcfg, cfg, video, man, lang)
        transcript = "\n".join(u["text_uk"] for u in man["utterances"])
        # local servers (LM Studio / Ollama) have small context windows; a 1h
        # transcript in the system prompt of every batch overflows them
        # silently. Remote providers (DeepSeek 1M + context caching) keep the
        # full transcript — that shape is what makes cached batches ~free.
        local = any(h in tcfg["base_url"] for h in ("127.0.0.1", "localhost"))
        max_ctx = int(tcfg.get("max_context_chars", 12000))
        if local and len(transcript) > max_ctx:
            log.warning("local endpoint: transcript context capped "
                        "%d -> %d chars", len(transcript), max_ctx)
            transcript = transcript[:max_ctx] + "\n[... transcript truncated]"
        context = ("Full source transcript (context only, translate ONLY the "
                   "segments in the request):\n" + transcript)
        cps = _measured_cps(cfg, lang)
        pace = ""
        if cps:
            pace = (f"Measured speaking pace of the TTS voice in "
                    f"{LANG_NAMES[lang]}: ~{cps:.1f} characters per second. "
                    f"Hard duration budget: keep each segment's \"text\" under "
                    f"seconds × {cps:.1f} characters; make each variant "
                    f"progressively shorter than that.")
            log.info("%s: measured TTS pace %.1f chars/s -> duration budget "
                     "in prompt", lang, cps)
        # adequacy judge sees terminology only (built here so the avoid block
        # can check it for collisions before the prompts are assembled)
        adeq_shared = "\n\n".join(x for x in [_glossary(lang), terms] if x)
        # words the TTS mis-stresses — ru/uk only, and deliberately NOT part of
        # adeq_shared: the judge grades faithfulness, and knowing which words we
        # would rather avoid could bias it toward approving a paraphrase.
        avoid = _avoid_block(cfg, lang, adeq_shared)
        # pace goes BEFORE the long transcript context so it isn't buried
        shared = "\n\n".join(x for x in [_glossary(lang), terms, avoid, pace,
                                         context] if x)
        sys_draft = _prompt(lang, tol, nvar) + "\n\n" + shared
        sys_reflect = _prompt(lang, tol, nvar, "translate_reflect") + "\n\n" + shared
        sys_adapt = _prompt(lang, tol, nvar, "translate_adapt") + "\n\n" + shared
        # (adeq_shared is built above, before the avoid block, because that
        # block needs it to detect terminology collisions. The judge still sees
        # terminology only — not the transcript, the pace budget or the avoid
        # list — because it grades faithfulness, not length or word choice.)
        flag = int(tcfg.get("adequacy_flag", 3))

        def _ask(sysmsg, payload, schema=_JSON_SCHEMA, model=None):
            data = _chat(client, tcfg, sysmsg,
                         json.dumps({"segments": payload}, ensure_ascii=False),
                         schema=schema, model=model)
            by_id = {s["id"]: s for s in data.get("segments", [])}
            dropped = [p for p in payload if p["id"] not in by_id]
            if dropped:  # small models drop segments; a singleton request is
                # near-impossible to drop — rescue instead of failing the run
                log.warning("LLM dropped %d segments (%s); retrying one by one",
                            len(dropped), [p["id"] for p in dropped])
                for p in dropped:
                    data = _chat(client, tcfg, sysmsg,
                                 json.dumps({"segments": [p]}, ensure_ascii=False),
                                 schema=schema, model=model)
                    by_id.update({s["id"]: s for s in data.get("segments", [])})
            missing = [p["id"] for p in payload if p["id"] not in by_id]
            if missing:
                raise RuntimeError(f"LLM dropped segments {missing}; re-run s3.")
            return by_id

        bs = tcfg["batch_size"]
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            payload = [{"id": u["id"], "uk": u["text_uk"], "chars": len(u["text_uk"]),
                        "seconds": round(u["end"] - u["start"], 1)} for u in batch]
            final = _ask(sys_draft, payload)
            if passes >= 3:  # translate -> reflect -> adapt
                drafted = [dict(p, draft=final[p["id"]]["text"]) for p in payload]
                # a stronger critic on the reflect pass only (~1/3 of tokens) is
                # the cheapest known quality lever; unset => same model
                critique = _ask(sys_reflect, drafted, schema=_REFLECT_SCHEMA,
                                model=tcfg.get("reflect_model"))
                adapted = [dict(d, issues=critique[d["id"]]["issues"])
                           for d in drafted]
                n_ok = sum(1 for d in adapted if d["issues"].strip().lower() == "ok")
                log.info("%s: reflect pass — %d/%d drafts clean",
                         lang, n_ok, len(adapted))
                final = _ask(sys_adapt, adapted)
            # Build every value first, then splice under the lock. The mutation
            # is `u["tr"][lang] = ...`, i.e. an INSERT into a dict that another
            # language's thread may be inside json.dumps() over. The lock only
            # ever covered the save, not the writes it serialises against.
            # I could not force a failure in 400 attempts, so this is a hazard
            # rather than an observed bug — but it is unsynchronised access on a
            # 32-minute paid path, and doing it correctly costs nothing.
            done = {u["id"]: {
                "text": final[u["id"]]["text"].strip(),
                "variants": [v.strip()
                             for v in final[u["id"]].get("variants", [])]}
                for u in batch}
            with save_lock:
                for u in batch:
                    u["tr"].setdefault(lang, {}).update(done[u["id"]])
            # adequacy gate: judge the FINAL text against the source. QC only
            # ever checked the AUDIO (WER/sim/MOS) — a mistranslation passes all
            # of those and only a human reading the review page caught it. Here
            # a cheap LLM-judge flags unfaithful segments BEFORE synthesis.
            # OUTSIDE the lock: it is an API round trip, and holding the lock
            # across it would serialise the languages this function exists to
            # overlap. It writes tr[lang]["adequacy"] into a key this thread
            # already created above, so no new insert races a concurrent save.
            if tcfg.get("adequacy_check", True):
                _adequacy(_ask, adeq_shared, batch, lang, flag, tcfg)
            with save_lock:
                man["stages"][f"s3_{lang}"] = f"{min(i+bs, len(todo))}/{len(todo)}"
                M.save(cfg, video, man)
            log.info("%s: %d/%d", lang, min(i + bs, len(todo)), len(todo))
        with save_lock:
            man["stages"][f"s3_{lang}"] = "done"
            M.save(cfg, video, man)

    workers = max(1, int(tcfg.get("lang_workers", len(langs))))
    if workers > 1 and len(langs) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(langs)),
                                thread_name_prefix="tr") as ex:
            # list() so an exception in any language surfaces here rather than
            # being swallowed by the context manager
            list(ex.map(_one_lang, langs))
    else:
        for lang in langs:
            _one_lang(lang)
