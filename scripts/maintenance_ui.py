#!/usr/bin/env python3
"""Standalone maintenance watcher UI for the Atlas maintenance lifecycle.

Serves a self-refreshing HTML page (plus a tiny JSON status endpoint) that tails
``maintenance-events.jsonl`` written by the MaintenanceCoordinator, so the
operator can watch a live drain->produce->restore window from a browser without
needing the agent to survive the drain.

Pure stdlib; no third-party deps. Run on the cluster:

    python3 scripts/maintenance_ui.py --journal-dir <dir> --port 8899 \
        --dashboard /home/kristianaaron/tmp/model-atlas/atlas_dashboard.html

The Atlas app is served at ``/atlas`` (same origin), so its embedded Maintenance
modal polls ``/api/status`` and tails the window live. Tunnel to it from the Mac:

    ssh -L 8899:127.0.0.1:8899 kristianaaron@100.96.194.44
    open http://127.0.0.1:8899/atlas
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PHASE_LABEL = {
    "drain": "Draining services",
    "produce": "Producing derivative",
    "restore": "Restoring / loading services",
    "maintenance": "Completed",
}
PHASE_ORDER = ("drain", "produce", "restore", "maintenance")
PHASE_DURATION_S = {"drain": 60, "produce": 1200, "restore": 420}


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


def _ts(ts: object) -> float | None:
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _resolve_journal(journal_arg: Path) -> Path:
    if (journal_arg / "maintenance-events.jsonl").exists():
        return journal_arg
    hits = sorted(
        (p.parent for p in journal_arg.rglob("maintenance-events.jsonl")),
        key=lambda p: (p / "maintenance-events.jsonl").stat().st_mtime,
        reverse=True,
    )
    return hits[0] if hits else journal_arg


def _status(journal_arg: Path) -> dict:
    journal = _resolve_journal(journal_arg)
    events = _read_events(journal / "maintenance-events.jsonl")
    now = datetime.now(UTC).timestamp()

    def latest(phase: str, status: str) -> dict | None:
        found = None
        for e in events:
            if e.get("phase") == phase and e.get("status") == status:
                found = e
        return found

    current = next(
        (p for p in reversed(PHASE_ORDER) if any(e.get("phase") == p for e in events)),
        "idle",
    )
    released = sorted(
        {e.get("service") for e in events
         if e.get("phase") == "drain" and e.get("status") == "release"}
    )
    loaded = sorted(
        {e.get("service") for e in events
         if e.get("phase") == "restore" and e.get("status") == "load"}
    )
    shard_cur, shard_tot = 0, 0
    for e in events:
        if e.get("phase") == "restore" and e.get("status") == "shard_loaded":
            if isinstance(e.get("shard_current"), int) and isinstance(e.get("shard_total"), int):
                shard_cur, shard_tot = e["shard_current"], e["shard_total"]
        if e.get("phase") == "restore" and e.get("status") == "shard_plan":
            if isinstance(e.get("shard_total"), int):
                shard_tot = max(shard_tot, e["shard_total"])

    starts = [t for t in (_ts(e.get("ts")) for e in events) if t is not None]
    run_started = min(starts, default=now)
    phase_starts = [
        t for t in (_ts(e.get("ts")) for e in events if e.get("phase") == current)
        if t is not None
    ]
    phase_start = min(phase_starts, default=run_started)
    elapsed = max(0.0, now - run_started)
    phase_elapsed = max(0.0, now - phase_start)
    phase_dur = PHASE_DURATION_S.get(current, 120)

    # derive preflight blockers (if any) so the UX can show a BLOCKED state
    blockers: list[dict] = []
    preflight_ev = [e for e in events if e.get("phase") == "preflight" and e.get("status") == "blocked"]
    if preflight_ev:
        detail = preflight_ev[-1].get("detail") or ""
        for part in detail.split(";"):
            if ">" in part:
                kind, _, rest = part.partition(">")
                blockers.append({"kind": kind.strip(), "detail": rest.strip()})
    return {
        "present": bool(events),
        "phase": current,
        "phase_label": PHASE_LABEL.get(current, current),
        "status": _status_text(events, current),
        "released": released,
        "loaded": loaded,
        "shard_current": shard_cur,
        "shard_total": shard_tot,
        "produce_method": (latest("produce", "start") or {}).get("method"),
        "result": (latest("maintenance", "complete") or {}).get("detail"),
        "run_started_epoch": round(run_started, 3),
        "elapsed_seconds": int(elapsed),
        "phase_remaining_seconds": int(max(0.0, phase_dur - phase_elapsed)),
        "phase_duration_s": phase_dur,
        "journal": str(journal),
    }


def _status_text(events: list[dict], current: str) -> str:
    if not events:
        return "no maintenance events yet — waiting for a window to start"
    if current == "drain":
        return "draining services"
    if current == "produce":
        m = next((e for e in events if e.get("phase") == "produce"
                  and e.get("status") == "start"), None)
        return f"producing ({m.get('method', 'unknown')})" if m else "producing"
    if current == "restore":
        return "restoring / loading services"
    if current == "maintenance":
        return "complete"
    return "idle"


WATCHER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atlas · Pipeline</title>
<style>
  :root{--bg:#0b0e14;--card:#12161f;--line:#232a38;--tx:#e6edf7;--mut:#8a94a6;
        --blue:#58a6ff;--amber:#d29922;--green:#3fb950;--red:#f85149;--hot:#f4f4f5}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
    font-family:'Inter',ui-sans-serif,system-ui,sans-serif}
  .wrap{max-width:760px;margin:0 auto;padding:36px 22px}
  h1{font-size:19px;letter-spacing:.4px;margin:0 0 4px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}
  .sub{color:var(--mut);font-size:13px;margin-bottom:22px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:22px;margin-bottom:14px}
  .state{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:13px;color:var(--mut);margin-bottom:6px}
  .status{color:var(--mut);font-size:13px;margin-bottom:10px}
  .bar{height:6px;border-radius:3px;background:#1d2532;overflow:hidden;border:1px solid var(--line);margin:12px 0 6px}
  .bar>div{height:100%;width:0%;background:var(--hot);transition:width .4s}
  .meta{display:flex;justify-content:space-between;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--mut)}
  .steps{display:flex;flex-direction:column;gap:2px}
  .step{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--line);font-size:14px}
  .step:last-child{border-bottom:none}
  .dot{width:10px;height:10px;border-radius:50%;background:#2b3443;flex:0 0 auto}
  .dot.on{background:#fff;box-shadow:0 0 8px #fff}
  .dot.done{background:var(--green)}
  .lbl{flex:1}.lbl small{color:var(--mut);margin-left:8px;font-weight:400}
  .eta{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12px;color:var(--amber);margin-top:10px}
  .resume{color:var(--mut);font-size:13px;margin-top:10px}
  details{background:#0e131b;border:1px solid var(--line);border-radius:8px;padding:10px;margin-top:12px}
  summary{cursor:pointer;color:var(--blue);font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:13px;outline:none}
  pre{background:#0c1016;border:1px solid var(--line);border-radius:6px;padding:10px;font-size:11px;color:#9aa4b2;max-height:220px;overflow:auto;white-space:pre-wrap;word-break:break-all}
  button{background:#262626;color:var(--tx);border:1px solid #3a3a3a;border-radius:5px;padding:4px 12px;font-size:11px;cursor:pointer;margin-top:6px}
  button:hover{background:#333}
</style></head><body>
<div class="wrap">
  <h1>Atlas · Pipeline</h1>
  <div class="sub">Live lifecycle watcher — stays up even while the agent is offline during a drain.
    <a href="/atlas" style="color:#58a6ff">Open Atlas app →</a></div>
  <div class="card">
    <div class="state" id="im-state">IDLE</div>
    <div class="status" id="im-status">No window running.</div>
    <div class="bar"><div id="im-fill"></div></div>
    <div class="meta"><span id="im-shards"></span><span id="im-time"></span></div>
  </div>
  <div class="card">
    <div class="steps">
      <div class="step"><span class="dot" id="im-s-drain"></span><span class="lbl">1 · Drain<small id="im-l-drain"></small></span></div>
      <div class="step"><span class="dot" id="im-s-produce"></span><span class="lbl">2 · Produce<small id="im-l-produce"></small></span></div>
      <div class="step"><span class="dot" id="im-s-restore"></span><span class="lbl">3 · Restore / reload<small id="im-l-restore"></small></span></div>
    </div>
    <div class="eta" id="im-eta"></div>
    <div class="resume" id="im-resume"></div>
  </div>
  <div class="card" id="im-debug-wrap" style="display:none">
    <details><summary>See debugging details</summary>
      <pre id="im-debug"></pre>
      <button id="im-copy">Copy error to clipboard</button></details>
  </div>
</div>
<script>
(function(){
  function $(id){return document.getElementById(id);}
  function fmt(s){s=Math.max(0,Math.floor(s||0));var m=Math.floor(s/60),r=s%60;return m+':'+(r<10?'0':'')+r;}
  function col(st){return {DRAIN:'#58a6ff',PRODUCE:'#d29922',RESTORE:'#3fb950',COMPLETE:'#3fb950',FAILED:'#f85149',BLOCKED:'#f85149',PREFLIGHT:'#58a6ff',IDLE:'#8a94a6'}[st]||'#8a94a6';}
  function orderOf(st){return {IDLE:0,PREFLIGHT:1,DRAIN:2,PRODUCE:3,RESTORE:4,COMPLETE:5,FAILED:5,BLOCKED:1}[st]||0;}
  function phaseOf(M){
    if(!M||!M.present) return 'IDLE';
    if(M.blocked) return 'BLOCKED';
    var p=M.phase;
    if(p==='preflight') return 'PREFLIGHT';
    if(p==='drain') return 'DRAIN';
    if(p==='produce') return 'PRODUCE';
    if(p==='restore') return 'RESTORE';
    if(p==='maintenance') return (M.result&&String(M.result).indexOf('success=False')>=0)?'FAILED':'COMPLETE';
    return p?String(p).toUpperCase():'IDLE';
  }
  function render(M){
    var st=phaseOf(M); if(st==='IDLE'){ $('im-state').textContent='IDLE'; $('im-status').textContent='No window running.'; return; }
    $('#im-state').textContent=st; $('#im-state').style.color=col(st);
    $('#im-status').textContent=(M.status||M.phase_label||st);
    var tot=M.shard_total||0,pct=0;
    if(st==='RESTORE'||st==='COMPLETE'){ pct=tot?Math.round(100*(M.shard_current||0)/tot):(st==='COMPLETE'?100:0); }
    else if(st==='DRAIN'){ pct=Math.min(100,Math.round(((M.released&&M.released.length)||0)/4*100)); }
    else if(st==='PRODUCE'){ pct=45; }
    $('#im-fill').style.width=Math.min(100,Math.max(0,pct))+'%';
    $('#im-shards').textContent=(tot&&(st==='RESTORE'||st==='COMPLETE'))?('DSV4 shards '+(M.shard_current||0)+'/'+tot+' ('+pct+'%)'):'';
    $('#im-time').textContent='elapsed '+fmt(M.elapsed_seconds);
    function step(k,idx,label){var o=orderOf(st),el=$('im-s-'+k),lco=$('im-l-'+k);el.className='dot'+(st===label?' on':(o>idx?' done':''));lco.textContent=st===label?'(running)':(o>idx?'\u2713':'');}
    step('drain',2,'DRAIN'); step('produce',3,'PRODUCE'); step('restore',4,'RESTORE');
    var eta=$('im-eta');
    if(st==='DRAIN') eta.textContent='Services going offline — the agent will drop; this panel keeps running.';
    else if(st==='PRODUCE') eta.textContent='Capturing & evaluating — the agent stays offline until restore. Expect minutes.';
    else if(st==='RESTORE') eta.textContent='Restoring DSV4 + prior services… agent comes back online after this.';
    else eta.textContent='';
    $('#im-resume').innerHTML=M.result||'';
    var dbg=$('im-debug-wrap');
    if(st==='BLOCKED'||st==='FAILED'){
      dbg.style.display='block';
      var reason=(M.blockers&&M.blockers.length)?M.blockers.map(function(b){return b.kind+': '+b.detail;}).join('\n'):(M.result||M.status||st);
      $('#im-debug').textContent=reason;
      $('#im-copy').onclick=function(){var ta=document.createElement('textarea');ta.value=reason;document.body.appendChild(ta);ta.select();try{document.execCommand('copy');}catch(e){}document.body.removeChild(ta);$('#im-copy').textContent='Copied!';};
    } else dbg.style.display='none';
  }
  function poll(){
    if(location.protocol.indexOf('http')!==0) return;
    fetch('/api/status',{cache:'no-store'}).then(function(r){return r.json();}).then(render).catch(function(){}).then(function(){setTimeout(poll,1500);});
  }
  poll();
})();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, ctype: str, body: bytes, no_store: bool = True):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if no_store:
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", "public, max-age=60")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/status":
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(_status(SERVER_JOURNAL)).encode())
        elif self.path in ("/", "/watcher", "/index.html"):
            self._send(200, "text/html; charset=utf-8", WATCHER_HTML.encode())
        elif self.path == "/atlas":
            if SERVER_DASHBOARD and SERVER_DASHBOARD.exists():
                body = SERVER_DASHBOARD.read_bytes()
                self._send(200, "text/html; charset=utf-8", body, no_store=False)
            else:
                self._send(404, "text/plain; charset=utf-8",
                           b"dashboard not found (pass --dashboard)")
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def log_message(self, *a):
        pass


SERVER_JOURNAL: Path
SERVER_DASHBOARD: Path | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    global SERVER_JOURNAL, SERVER_DASHBOARD
    SERVER_JOURNAL = args.journal_dir
    SERVER_DASHBOARD = args.dashboard
    args.journal_dir.mkdir(parents=True, exist_ok=True)
    if args.dashboard and not args.dashboard.exists():
        print(f"WARNING: dashboard not found at {args.dashboard}", flush=True)

    print(f"serving Atlas watcher on http://{args.host}:{args.port}", flush=True)
    print(f"  watcher : http://127.0.0.1:{args.port}/", flush=True)
    print(f"  atlas app: http://127.0.0.1:{args.port}/atlas  (journal: {args.journal_dir})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
