"""openai4s daemon — stdlib http.server, zero external deps.

Endpoints:
  GET  /         -> minimal web UI (single page)
  GET  /health   -> {"status":"ok",...}
  POST /run      -> {"task": "..."} -> runs the Code-as-Action agent, returns
                    {stop_reason, submitted_output, final_message, transcript}

Bind to 127.0.0.1 by default; expose via SSH port-forward, never 0.0.0.0 on an
untrusted network (see / deployment notes).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai4s.agent import Agent
from openai4s.config import Config, get_config

_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>openai4s</title>
<style>
:root{color-scheme:light dark}
body{font:15px/1.5 system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.3rem;margin:0 0.3rem}
.sub{color:#888;margin:0 0 1.2rem;font-size:.85rem}
textarea{width:100%;min-height:90px;padding:.6rem;border:1px solid #8884;border-radius:8px;font:inherit;box-sizing:border-box}
button{margin-top:.6rem;padding:.5rem 1.1rem;border:0;border-radius:8px;background:#5b5bd6;color:#fff;font:inherit;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
pre{white-space:pre-wrap;word-break:break-word;background:#8881;padding:.8rem;border-radius:8px;font-size:.82rem}
.turn{border-left:3px solid #8884;padding:.2rem.7rem;margin:.5rem 0}
.turn.assistant{border-color:#5b5bd6}
.turn.observation{border-color:#2a9d8f}
.role{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:#888}
</style></head><body>
<h1>openai4s</h1>
<p class="sub">Code-as-Action agent · ark / doubao-seed-2.0-pro</p>
<textarea id="task" placeholder="Describe a task, e.g. Compute the mean of [1,2,3,4] and submit it."></textarea>
<br><button id="go">Run</button>
<div id="out"></div>
<script>
const btn=document.getElementById('go'),ta=document.getElementById('task'),out=document.getElementById('out');
btn.onclick=async()=>{
  const task=ta.value.trim(); if(!task)return;
  btn.disabled=true; out.innerHTML='<p class="sub">running…</p>';
  try{
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task})});
    const j=await r.json();
    let html='';
    if(j.error){html='<pre>ERROR: '+j.error+'</pre>';}
    else{
      html+='<p class="sub">stop: '+j.stop_reason+' · turns: '+(j.transcript?j.transcript.length:0)+'</p>';
      if(j.submitted_output)html+='<div class="turn"><div class="role">submitted output</div><pre>'+JSON.stringify(j.submitted_output,null,2)+'</pre></div>';
      (j.transcript||[]).forEach(t=>{html+='<div class="turn '+t.role+'"><div class="role">'+t.role+'</div><pre>'+t.content.replace(/</g,'&lt;')+'</pre></div>';});
    }
    out.innerHTML=html;
  }catch(e){out.innerHTML='<pre>'+e+'</pre>';}
  btn.disabled=false;
};
</script></body></html>"""


def _make_handler(cfg: Config):
    class Handler(BaseHTTPRequestHandler):
        server_version = "openai4s/0.1"

        def log_message(self, *a):  # quieter default logging
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(
                code,
                json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, _INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/health":
                self._json(
                    200,
                    {
                        "status": "ok",
                        "model": cfg.llm.model,
                        "data_dir": str(cfg.data_dir),
                    },
                )
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/run":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                task = (payload.get("task") or "").strip()
                if not task:
                    self._json(400, {"error": "missing 'task'"})
                    return
                result = Agent(cfg=cfg).run(task)
                self._json(200, result)
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": str(e)})

    return Handler


def build_server(cfg: Config | None = None) -> ThreadingHTTPServer:
    cfg = cfg or get_config()
    handler = _make_handler(cfg)
    return ThreadingHTTPServer((cfg.host, cfg.port), handler)


def serve(cfg: Config | None = None, *, block: bool = True) -> ThreadingHTTPServer:
    cfg = cfg or get_config()
    httpd = build_server(cfg)
    if block:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.shutdown()
    else:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
    return httpd
