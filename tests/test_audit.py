import ast
import sys
from pathlib import Path
import yaml


def test_audit():
    """Mechanical audit for the failure classes that actually bit us.

Every recent bug had the same shape: a change that reads correctly at the call
site and is never checked against the thing it is meant to affect.
  - tts.engine never set in the GPU profile -> production ran chatterbox
  - _remote_stages read "from"/"to" instead of from_stage/to_stage -> silent
    fallback to the whole pipeline
  - a local `import shlex` shadowed the module one -> UnboundLocalError
  - config keys that no code reads, and code keys no config sets

So: check names against their definitions, not against intent.
"""



    ROOT = Path(__file__).resolve().parents[1]
    FAIL = []


    def bad(msg):
        FAIL.append(msg)
        print("  FAIL " + msg)


    def ok(msg):
        print("  ok   " + msg)


    print("\n=== 1. local imports shadowing module-level names ===")
    for py in sorted(ROOT.glob("pipeline/*.py")) + sorted(ROOT.glob("qc/*.py")) + [Path("dub.py")]:
        tree = ast.parse(py.read_text())
        top = {a.asname or a.name.split(".")[0]
               for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            inner = {a.asname or a.name.split(".")[0]
                     for n in ast.walk(fn) if isinstance(n, ast.Import) for a in n.names}
            clash = top & inner
            if clash:
                bad(f"{py}:{fn.name} re-imports {sorted(clash)} already at module "
                    f"scope -> the name becomes function-local for the WHOLE "
                    f"function (this is the shlex bug)")
    if not FAIL:
        ok("no function-local import shadows a module-level one")

    print("\n=== 2. argparse dests vs how they are read ===")
    src = (ROOT / "dub.py").read_text()
    tree = ast.parse(src)
    dests = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument":
            d = next((k.value.value for k in n.keywords if k.arg == "dest"), None)
            if d:
                dests.add(d)
            else:
                for a in n.args:
                    if isinstance(a, ast.Constant):
                        # positionals ("cmd", "rest") are dests too — missing them
                        # made the audit cry wolf on its first run
                        dests.add(a.value[2:].replace("-", "_")
                                  if a.value.startswith("--") else a.value)
    # every getattr(a, "...") / a.<attr> on the parsed namespace must be a real dest
    reads = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr" \
                and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant):
            if getattr(n.args[0], "id", "") == "a":
                reads.add(n.args[1].value)
        if isinstance(n, ast.Attribute) and getattr(n.value, "id", "") == "a":
            reads.add(n.attr)
    unknown = {r for r in reads if r not in dests and not r.startswith("_")}
    if unknown:
        bad(f"dub.py reads namespace attrs that are not argparse dests: "
            f"{sorted(unknown)} (this is the from_stage bug)")
    else:
        ok(f"all {len(reads)} namespace reads match a declared dest")

    print("\n=== 3. tts config keys: set-but-unread and read-but-unset ===")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    gpu = yaml.safe_load((ROOT / "config.gpu.yaml").read_text())
    merged_tts = {**cfg.get("tts", {}), **(gpu.get("tts") or {})}
    code = "\n".join(p.read_text() for p in
                     list(ROOT.glob("pipeline/*.py")) + list(ROOT.glob("qc/*.py")))
    unread = [k for k in merged_tts
              if f'"{k}"' not in code and f"'{k}'" not in code]
    if unread:
        bad(f"tts keys in config that NO code reads: {sorted(unread)}")
    else:
        ok(f"all {len(merged_tts)} tts config keys are read somewhere")

    print("\n=== 4. engine routing: bakeoff vs production ===")
    sys.path.insert(0, str(ROOT))
    from pipeline.logic import deep_merge                      # noqa: E402
    from pipeline.manifest import resolve_engine               # noqa: E402
    m = deep_merge(cfg, gpu)
    bo_engines = set(m.get("bakeoff", {}).get("engines", []))
    prod = {lang: resolve_engine(m["tts"], lang) for lang in m["languages"]}
    if set(prod.values()) - bo_engines - {"edge"}:
        bad(f"production resolves to {sorted(set(prod.values()))} but the bake-off "
            f"validated {sorted(bo_engines)} — they must agree")
    else:
        ok(f"production engine {sorted(set(prod.values()))} == bake-off roster")

    print("\n=== 5. engines referenced in code vs installable ===")
    import pipeline.runpod_infra as R                          # noqa: E402
    # the dispatch dict now lives in tts_engine.synthesize (engine_worker was
    # deleted with the venv isolation, 2026-08-02). Only that dict maps names to
    # _synth_* functions; every other literal in the file is something else.
    dispatch = set()
    for n in ast.walk(ast.parse((ROOT / "pipeline/tts_engine.py").read_text())):
        if isinstance(n, ast.Dict) and n.values and all(
                isinstance(v, ast.Name) and v.id.startswith("_synth_")
                for v in n.values):
            dispatch |= {k.value for k in n.keys if isinstance(k, ast.Constant)}
    setup = set((gpu.get("runpod") or {}).get("engine_setup", {}))
    need_install = dispatch - {"chatterbox", "edge"}
    missing = need_install - setup
    if missing:
        bad(f"engines the worker can dispatch but no install recipe exists: "
            f"{sorted(missing)}")
    else:
        ok(f"dispatch {sorted(dispatch)} — all git-clone engines have recipes")

    print("\n=== 6. config keys named in COMMENTS actually exist ===")
    # The chatterbox bug was a comment asserting behaviour that was never
    # wired ("qwen serves every language; engine_by_lang is left empty on
    # purpose"). Comments that name a `key:` are documentation of a setting —
    # if the setting is absent, the comment is a claim, not a description.
    import re as _re
    for f in ("config.yaml", "config.gpu.yaml"):
        text = (ROOT / f).read_text()
        live = set(_re.findall(r"^\s{0,6}([a-z_][a-z0-9_]*):", text, _re.M))
        # "# key: value" — a commented-out setting is fine (it is a default),
        # but a comment REFERRING to tts.foo / bakeoff.bar must resolve
        refs = set(_re.findall(r"#[^\n]*\b(?:tts|bakeoff|runpod|qc)\.([a-z_][a-z0-9_]*)",
                               text))
        ghosts = sorted(r for r in refs if r not in live and r not in code)
        if ghosts:
            bad(f"{f}: comments reference settings that exist nowhere: {ghosts}")
    if not any("exist nowhere" in x for x in FAIL):
        ok("every dotted config reference in comments resolves")

    print("\n=== 7. stage graph: dub.py STAGES vs the stage modules ===")
    # ORDER = list(STAGES), so read the dict literal rather than the alias.
    import importlib
    order = None
    for n in ast.walk(ast.parse((ROOT / "dub.py").read_text())):
        if (isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "STAGES"
                and isinstance(n.value, ast.Dict)):
            order = [k.value for k in n.value.keys if isinstance(k, ast.Constant)]
    if not order:
        bad("dub.py STAGES not found — the stage graph is unverifiable")
    else:
        missing = [st for st in order
                   if not (ROOT / "pipeline" / f"{st}.py").exists()]
        if missing:
            bad(f"STAGES lists stages with no module: {missing}")
        else:
            ok(f"all {len(order)} stages in STAGES have a pipeline module")
        norun = [st for st in order
                 if not hasattr(importlib.import_module(f"pipeline.{st}"), "run")]
        if norun:
            bad(f"stage modules without run(): {norun}")
        else:
            ok("every stage module exposes run()")

    print("\n=== 8. synth_hash covers every output-changing tts knob ===")
    from pipeline.manifest import synth_hash                   # noqa: E402
    base = {"engine": "qwen", "reference_wav": "r.wav", "cfg_weight": 0.0,
            "exaggeration": 0.5}
    h0 = synth_hash("hi", "en", base)
    # qwen_x_vector_only is deliberately absent: without a reference_text the
    # adapter forces x_only=True regardless, so the hash correctly does not move.
    OUTPUT_CHANGING = ["qwen_fast", "qwen_model_dir",
                       "reference_max_s", "qwen_gen_kwargs"]
    for k in OUTPUT_CHANGING:
        v = {"qwen_model_dir": "other",
             "reference_max_s": 12, "qwen_fast": True,
             "qwen_gen_kwargs": {"temperature": 0.5}}[k]
        if synth_hash("hi", "en", {**base, k: v}) == h0:
            bad(f"synth_hash ignores {k} — the cache would serve audio made with a "
                f"different setting")
    if not any("synth_hash ignores" in f for f in FAIL):
        ok(f"all {len(OUTPUT_CHANGING)} output-changing knobs salt the cache key")

    print("\n=== 9. functions reading names that are defined nowhere ===")
    # `setup_check` read an `overlays` global copy-pasted from remote_run, which
    # takes it as a PARAMETER. Python resolves unknown names at call time, so it
    # imported, passed every other check, and raised NameError only once the
    # command was actually invoked. Checks 1-8 all compare a name to its
    # definition; this one asks whether a definition exists at all.
    import builtins
    import symtable
    for py in (sorted(ROOT.glob("pipeline/*.py")) + sorted(ROOT.glob("qc/*.py"))
               + [ROOT / "dub.py"]):
        top = symtable.symtable(py.read_text(), str(py), "exec")
        # module scope + builtins; a name assigned anywhere at module level
        # (including inside `try:`/`if`) counts as defined.
        known = {s.get_name() for s in top.get_symbols()} | set(dir(builtins))

        def scan(tbl):
            for ch in tbl.get_children():
                if ch.get_type() == "function":
                    for s in ch.get_symbols():
                        # is_global + never assigned in this scope == a free
                        # name that must resolve at module level or in builtins
                        if (s.is_global() and not s.is_assigned()
                                and s.get_name() not in known):
                            bad(f"{py.name}:{ch.get_lineno()} {ch.get_name()}() "
                                f"reads {s.get_name()!r}, which is not defined at "
                                f"module scope, not a parameter, and not a "
                                f"builtin -> NameError at call time "
                                f"(this is the setup_check/overlays bug)")
                scan(ch)
        scan(top)
    if not any("NameError at call time" in f for f in FAIL):
        ok("every free name in every function resolves to a definition")

    print(f"\n{'='*60}\n{len(FAIL)} problem(s)" if FAIL else
          f"\n{'='*60}\nclean")
    assert not FAIL, "\n".join(FAIL)
