#!/usr/bin/env python3
"""Standalone maintenance watcher UI for the Atlas maintenance lifecycle.

Serves a self-refreshing HTML page (plus a tiny JSON status endpoint) that tails
``maintenance-events.jsonl`` written by the MaintenanceCoordinator, so the
operator can watch a live drain->produce->restore window from a browser without
needing the agent to survive the drain.

Pure stdlib; no third-party deps. Run on the cluster:

    python3 scripts/maintenance_ui.py --journal-dir <dir> --port 8899

Then tunnel to it from the Mac (see `--help` / the SSH tunnel note below).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
PHASE_DURATION_S = {"drain": 60, "produce": 1200, "restore": 420}  # honest estimate


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


def _status(journal: Path) -> dict:
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
        {e.get("service") for e in events if e.get("phase") == "drain" and e.get("status") == "release"}
    )
    loaded = sorted(
        {e.get("service") for e in events if e.get("phase") == "restore" and e.get("status") == "load"}
    )
    shard_latest = {"current": 0, "total": 0}
    for e in events:
        if e.get("phase") == "restore" and e.get("status") == "shard_loaded":
            if isinstance(e.get("shard_current"), int) and isinstance(e.get("shard_total"), int):
                shard_latest = {"current": e["shard_current"], "total": e["shard_total"]}
    for e in events:
        if e.get("phase") == "restore" and e.get("status") == "shard_plan":
            if isinstance(e.get("shard_total"), int):
                shard_latest["total"] = max(shard_latest["total"], e["shard_total"])

    starts = [t for t in (_ts(e.get("ts")) for e in events) if t is not None]
    run_started = min(starts, default=now)
    phase_starts = [
        t
        for t in (_ts(e.get("ts")) for e in events if e.get("phase") == current)
        if t is not None
    ]
    phase_start = min(phase_starts, default=run_started)
    elapsed = max(0.0, now - run_started)
    phase_elapsed = max(0.0, now - phase_start)
    phase_dur = PHASE_DURATION_S.get(current, 120)

    resume = {
        "services": loaded or [],
        "shards": shard_latest,
        "note": "prior runtime + model restored" if loaded else None,
    }

    return {
        "present": bool(events),
        "phase": current,
        "phase_label": PHASE_LABEL.get(current, current),
        "status": latest("drain", "complete") and "all consumers quiesced"
        or status_text(events, current),
        "released": released,
        "loaded": loaded,
        "shard_current": shard_latest["current"],
        "shard_total": shard_latest["total"],
        "produce_method": (latest("produce", "start") or {}).get("method"),
        "result": (latest("maintenance", "complete") or {}).get("detail"),
        "run_started_epoch": round(run_started, 3),
        "elapsed_seconds": int(elapsed),
        "phase_remaining_seconds": int(max(0.0, phase_dur - phase_elapsed)),
        "phase_duration_s": phase_dur,
        "resume": resume,
        "journal": str(journal),
    }


def status_text(events: list[dict], current: str) -> str:
    if not events:
        return "no maintenance events yet — waiting for a window to start"
    if current == "drain":
        return "draining services"
    if current == "produce":
        m = next((e for e in events if e.get("phase") == "produce" and e.get("status") == "start"), None)
        return f"producing ({m.get('method', 'unknown')})" if m else "producing"
    if current == "restore":
        return "restoring / loading services"
    if current == "maintenance":
        return "complete"
    return "idle"


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atlas · Maintenance</title>
<style>
  :root{--bg:#0b0e14;--card:#12161f;--line:#232a38;--tx:#dbe3f0;--mut:#7c8798;
        --blue:#58a6ff;--amber:#d29922;--green:#3fb950;--red:#f85149;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--tx);
    font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:760px;margin:0 auto;padding:40px 22px}
  h1{font-size:20px;letter-spacing:.4px;margin:0 0 4px}
  .sub{color:var(--mut);font-size:13px;margin-bottom:26px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:22px;margin-bottom:16px}
  .phase{display:flex;flex-direction:column;gap:2px}
  .phase-name{font-size:15px;font-weight:600;color:var(--blue)}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}
  .pulse{animation:pulse 1.1s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
  .bar{height:14px;border-radius:8px;background:#1a212d;overflow:hidden;margin:12px 0 6px;
       border:1px solid var(--line)}
  .bar>div{height:100%;width:0%;background:linear-gradient(90deg,var(--blue),#79c0ff);
       transition:width .5s}
  .meta{display:flex;justify-content:space-between;font-family:ui-monospace,Menlo,monospace;
        font-size:12.5px;color:var(--mut)}
  .k{color:var(--green)} .tag{display:inline-block;background:#1a212d;border:1px solid var(--line);
        border-radius:20px;padding:3px 11px;font-size:12px;color:var(--mut);margin:2px 4px 2px 0}
  .muted{color:var(--mut)} .big{font-size:15px;font-weight:600}
  .spinner{width:34px;height:34px;border:4px solid rgba(255,255,255,.12);
        border-top-color:var(--blue);border-radius:50%;animation:spin 1s linear infinite;
        display:inline-block}
  @keyframes spin{to{transform:rotate(360deg)}}
  .row{display:flex;align-items:center;gap:14px}
  #notify{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(4,6,10,.55);
        display:none;align-items:center;justify-content:center;z-index:50}
  #notify .box{background:var(--card);border:1px solid var(--green);border-radius:14px;
        padding:26px;max-width:420px;text-align:center}
</style></head><body>
<div class="wrap">
  <h1>Atlas · Maintenance Window</h1>
  <div class="sub">Live lifecycle watcher — auto-refreshes. This panel stays up even while the
    agent/gateway is offline during the drain.</div>

  <div class="card">
    <div class="phase">
      <span class="phase-name" id="phase"><span class="dot pulse"></span>Waiting for a window…</span>
      <span class="muted" id="substatus" style="font-size:13px">No maintenance run recorded yet.</span>
    </div>
    <div class="bar"><div id="fill"></div></div>
    <div class="meta">
      <span id="shards"></span><span id="time"></span>
    </div>
  </div>

  <div class="card">
    <div style="font-size:13px;color:var(--mut);margin-bottom:10px">Pipeline phases</div>
    <div style="display:flex;flex-direction:column;gap:8px">
      <div class="row"><span class="spinner" id="sp-drain"></span>
        <span class="big" id="lbl-drain">1 · Drain</span></div>
      <div class="row"><span class="spinner" id="sp-produce"></span>
        <span class="big" id="lbl-produce">2 · Produce</span></div>
      <div class="row"><span class="spinner" id="sp-restore"></span>
        <span class="big" id="lbl-restore">3 · Restore / reload</span></div>
    </div>
  </div>

  <div class="card" id="resume-card" style="display:none">
    <div style="font-weight:600;margin-bottom:8px">Resumed what was running before</div>
    <div id="services"></div>
    <div id="resume-shards" class="muted" style="font-size:13px;margin-top:8px"></div>
  </div>
</div>
<div id="notify"><div class="box"><div style="font-size:40px">&#9989;</div>
  <div class="big" style="margin:10px 0">All services restored.</div>
  <div class="muted" id="n-det"></div></div></div>

<script>
let lastShards="", lastPhase="";
async function tick(){
  let d; try{ d=await (await fetch('/api/status',{cache:'no-store'})).json(); }
  catch(e){ setStatus("watcher offline","",0,0,0); return; }
  const ph=d.phase;
  const active=["drain","produce","restore"].includes(ph);
  document.getElementById('phase').innerHTML=
    '<span class="dot '+(active?'pulse':'')+'" style="background:'+col(ph)+'"></span>'+
    (d.phase_label||ph);
  document.getElementById('substatus').textContent=d.status||'';
  const tot=d.shard_total||0;
  const fill=document.getElementById('fill');
  if(ph==='restore'||ph==='maintenance'){
    const pct=tot?Math.round(100*d.shard_current/tot):(ph==='maintenance'?100:0);
    fill.style.width=Math.min(100, Math.max(0,pct))+'%';
    document.getElementById('shards').innerHTML=tot?
      ('DSV4 shards <span style="color:'+col(ph)+'"><b>'+d.shard_current+'</b>/'+tot+'</span> '+'('+pct+'%)')
      :'';
  } else {
    document.getElementById('shards').textContent='';
  }
  // phase spinners
  const mark=id=>{const el=document.getElementById(id);el.style.visibility='visible'};
  ['sp-drain','sp-produce','sp-restore'].forEach(id=>{document.getElementById(id).style.visibility='hidden'});
  if(ph==='drain')mark('sp-drain'); if(ph==='produce')mark('sp-produce'); if(ph==='restore')mark('sp-restore');
  document.getElementById('lbl-drain').innerHTML='1 · Drain'+(ph==='drain'?' <span class="muted">(active)</span>':'')+(d.released&&ph==='drain'?' <span class="muted">· released: '+d.released.join(', ')+'</span>':'');
  document.getElementById('lbl-produce').innerHTML='2 · Produce'+(ph==='produce'?' <span class="muted">(active)</span>':'')+(d.produce_method?' <span class="muted">· '+d.produce_method+'</span>':'');
  document.getElementById('lbl-restore').innerHTML='3 · Restore / reload'+(ph==='restore'?' <span class="muted">(active)</span>':'')+(d.loaded&&ph==='restore'?' <span class="muted">· '+d.loaded.join(', ')+'</span>':'');
  // timer
  document.getElementById('time').innerHTML='elapsed <b>'+fmt(d.elapsed_seconds)+'</b>'+(active?' · remaining <b>'+fmt(d.phase_remaining_seconds)+'</b>':'');
  // resume summary
  const rc=document.getElementById('resume-card');
  if(d.result){
    rc.style.display='block';
    document.getElementById('services').innerHTML=(d.loaded&&d.loaded.length?d.loaded.map(s=>'<span class="tag">'+s+'</span>').join(''):'<span class="tag">none</span>')+
      '<div class="muted" style="margin-top:6px">'+esc(d.result)+'</div>';
    if(tot) document.getElementById('resume-shards').textContent='DSV4 '+d.shard_current+'/'+tot+' shards loaded.';
    if(lastPhase!=='maintenance'){ showNotify(); }
  } else if(ph==='restore'){
    rc.style.display='block';
    document.getElementById('services').innerHTML='<span class="tag">DSV4</span>'+(d.loaded&&d.loaded.length?d.loaded.map(s=>'<span class="tag">'+s+'</span>').join(''):'');
  }
  lastPhase=ph;
}
function col(p){return p==='drain'?'#58a6ff':p==='produce'?'#d29922':p==='restore'?'#3fb950':p==='maintenance'?'#a5d6ff':'#7c8798';}
function fmt(s){s=Math.max(0,Math.floor(s||0));const m=Math.floor(s/60),r=s%60;return m+':'+(r<10?'0':'')+r;}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function setStatus(a,b,c,d,e){document.getElementById('phase').innerHTML='<span class="dot"></span>'+a;document.getElementById('substatus').textContent=b||'';}
function showNotify(){const n=document.getElementById('notify');n.style.display='flex';
  document.getElementById('n-det').textContent=document.getElementById('services').textContent||'DSV4 reloaded';}
document.getElementById('notify').addEventListener('click',()=>{document.getElementById('notify').style.display='none';});
tick(); setInterval(tick,2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/api/status":
            body = json.dumps(_status(SERVER_JOURNAL)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):  # silence
        pass


SERVER_JOURNAL: Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    global SERVER_JOURNAL
    SERVER_JOURNAL = args.journal_dir
    args.journal_dir.mkdir(parents=True, exist_ok=True)

    print(f"serving Atlas maintenance watcher on http://{args.host}:{args.port} "
          f"(journal: {args.journal_dir})", flush=True)
    print("SSH tunnel from the Mac:  ssh -L " + str(args.port) + ":127.0.0.1:"
          + str(args.port) + " kristianaaron@100.96.194.44", flush=True)
    print("Then open http://127.0.0.1:" + str(args.port) + " in a browser.", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
