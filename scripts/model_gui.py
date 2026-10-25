"""Local HTML GUI for the EXISTING MMA3001 downforce_model.py.

On Windows, double-click `Open M25 Predictor.bat` in the project root.

"""

from __future__ import annotations

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import runpy
import sys
import webbrowser
from urllib.parse import parse_qs, urlparse

# This is a local browser UI, not a plotting session. Prevent the model's
# existing diagnostic plt.show() calls from opening many blocking windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_FILE = HERE / "downforce_model.py"
HOST = "127.0.0.1"
PORT = 0  # Let Windows assign an unused port; avoids reusing an older GUI server.
MODEL_NAMES = ("Linear", "Quadratic", "Gaussian RBF")

HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMA3001 | Monash Motorsport M25 Downforce Predictor</title>
<style>
:root{color-scheme:light;--mms:#0876b5;--mms-deep:#075d96;--cyan:#20a2de;--navy:#14283f;--orange:#ee9050;--bg:#f4f7fa;--paper:#fff;--edge:#dce5ec;--muted:#667c8d;--ink:#16283b;--font:Arial,Helvetica,system-ui,sans-serif;--display:'Arial Narrow',Impact,'Helvetica Neue',Arial,sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--bg);font-family:var(--font);line-height:1.46}
.topline{height:10px;background:linear-gradient(108deg,#0674b6 0 71%,#1598da 71% 80%,#fff 80% 81%,#0674b6 81% 100%)}
.page{max-width:1240px;padding:0 28px 60px;margin:auto}
.heading{position:relative;overflow:hidden;background:var(--mms);color:#fff;margin:26px 0 23px;padding:30px 36px 32px;min-height:194px;display:flex;align-items:center;justify-content:space-between;gap:20px}
.heading::before,.heading::after{content:"";position:absolute;pointer-events:none;transform:skew(-19deg);top:-90px;bottom:-90px}.heading::before{background:#158bcc;width:132px;right:113px;opacity:.67}.heading::after{background:#22a4e0;width:34px;right:99px;opacity:.92}
.head-text{position:relative;z-index:1;max-width:850px}.eyebrow{letter-spacing:2.8px;font-weight:750;font-size:12px;margin-bottom:10px;opacity:.92}.heading h1{font:900 clamp(29px,3vw,45px)/1.04 var(--display);text-transform:uppercase;letter-spacing:.1px;margin:0;max-width:850px}.heading h1 span{display:block}.heading p{color:#d7eefa;font-size:14px;margin:15px 0 0}.heading p::before{content:"";display:block;width:63px;height:4px;background:var(--cyan);margin:0 0 12px}
.head-right{z-index:2;position:relative;align-self:flex-start;display:flex;flex-direction:column;align-items:flex-end;gap:28px}.team{font-weight:900;font-style:italic;font-size:17px;letter-spacing:1px;white-space:nowrap}.status{background:#fff;border:1px solid #fff;padding:7px 12px;border-radius:4px;color:var(--mms-deep);font-size:11px;white-space:nowrap;font-weight:700}.status.err{color:#a42626;border-color:#f7b5aa;background:#fff5f1}
.layout{display:grid;grid-template-columns:355px minmax(0,1fr);gap:19px;align-items:start}.panel{background:var(--paper);border:1px solid var(--edge);padding:24px;box-shadow:0 3px 12px rgba(21,51,74,.055)}
.panel h2{color:var(--mms-deep);text-transform:uppercase;font:900 21px/1.15 var(--display);letter-spacing:.2px;margin:0 0 23px;border-left:5px solid var(--cyan);padding:1px 0 1px 12px}
.field{margin-bottom:22px}.field-head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px}label{color:var(--navy);font-size:13px;font-weight:700}.range-note{font-size:11px;color:var(--muted);white-space:nowrap}
.field input[type=number],select{width:100%;border:1px solid #bfced9;background:#fbfdff;color:var(--navy);border-radius:3px;padding:11px 12px;outline:none;font:500 15px var(--font);transition:border-color .16s}.field input:focus,select:focus{border-color:var(--mms);box-shadow:0 0 0 3px #1ca1db26}.slider{display:block;width:100%;accent-color:var(--mms);margin:13px 0 0;cursor:pointer}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.info{border-top:1px solid var(--edge);margin-top:7px;padding-top:16px;color:#5b7182;font-size:12px;line-height:1.65}.info strong{color:var(--navy)}.info .data-label{display:inline-block;font-size:10px;text-transform:uppercase;letter-spacing:1px;font-weight:800;background:var(--navy);color:#fff;padding:4px 9px;margin:0 0 9px}.info p{margin:0 0 5px}.info b{color:var(--mms-deep)}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:14px}.metric{border:1px solid var(--edge);background:#f7fafc;padding:18px;min-height:135px;border-top:4px solid #b8c9d6}.metric.primary{background:#eff8fe;border-top-color:var(--mms)}.metric:nth-child(2){border-top-color:var(--cyan)}.metric .meta{color:#566b7b;text-transform:uppercase;letter-spacing:.95px;font-size:11px;font-weight:800;margin:0 0 9px}.metric .value{font-size:clamp(26px,2.9vw,36px);font-weight:800;color:var(--navy);letter-spacing:-1px;white-space:nowrap;font-variant-numeric:tabular-nums}.metric.primary .value{color:var(--mms)}.metric .unit{font-size:12px;color:#597385;margin-left:5px}.metric .detail{color:#617789;font-size:11px;line-height:1.35;margin:8px 0 0}
.smallbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:27px 0 9px}.smallbar h2{margin:0}.smallbar span{font-size:11px;color:#6a7f90}.legend{display:inline-flex;gap:6px;align-items:center}.legend b{width:16px;height:3px;background:var(--mms);display:inline-block}.legend.uncert b{height:9px;background:var(--cyan);opacity:.3}.plot{width:100%;height:auto;display:block;margin-top:8px;border:1px solid #dce7ef;background:#fff}.note{border-left:3px solid var(--orange);background:#fff8f3;color:#626f7a;font-size:11px;line-height:1.65;padding:12px 14px;margin-top:14px}.note strong{color:var(--navy)}.error{display:none;background:#fff4f2;border:1px solid #eaa198;color:#a12f27;padding:12px;margin:10px 0;font-size:12px}.error.active{display:block}
.footer{display:flex;justify-content:space-between;gap:12px;margin-top:18px;border-top:2px solid var(--mms);padding-top:12px;font-size:11px;color:#718390}.footer strong{color:var(--mms-deep)}
@media(max-width:900px){.layout{grid-template-columns:1fr}.heading{align-items:flex-start;min-height:180px}.head-right{gap:30px}.page{padding:0 16px 44px}}@media(max-width:660px){.heading{padding:23px 20px;display:block}.heading::before{right:-12px;opacity:.25}.heading::after{right:-8px;opacity:.32}.head-right{align-items:flex-start;margin-top:14px;gap:10px}.team{display:none}.heading h1{font-size:31px}.metrics{grid-template-columns:1fr 1fr}.panel{padding:18px}.smallbar{align-items:flex-start;flex-direction:column}.footer{flex-direction:column}}@media(max-width:420px){.metrics{grid-template-columns:1fr}.twocol{grid-template-columns:1fr 1fr}.heading h1{font-size:27px}}
</style>
</head>
<body>
<div class="topline"></div>
<div class="page">
<header class="heading">
  <div class="head-text"><div class="eyebrow">MMA3001</div><h1><span>Monash Motorsport M25</span><span>Aerodynamic Downforce Predictor</span></h1><p>Christina Deva Doss</p></div>
  <div class="head-right"><div class="team">MONASH MOTORSPORT</div><div class="status" id="status">Connecting to model…</div></div>
</header>
<div class="layout">
  <section class="panel">
    <h2>Operating Conditions</h2>
    <div class="field"><div class="field-head"><label for="model">Regression model</label></div><select id="model"><option value="Linear">Linear regression</option><option value="Quadratic">Quadratic polynomial</option><option value="Gaussian RBF">Gaussian RBF · optimised</option></select></div>
    <div class="field"><div class="field-head"><label for="speed">Airspeed · m/s</label><span class="range-note" id="speed-range"></span></div><input type="number" id="speed" value="17" step="0.1"><input class="slider" type="range" id="speed-slider" step="0.1" value="17"></div>
    <div class="field"><div class="field-head"><label for="yaw">Yaw angle · degrees</label><span class="range-note" id="yaw-range"></span></div><input type="number" id="yaw" value="0" step="0.5"><input class="slider" type="range" id="yaw-slider" step="0.5" value="0"></div>
    <div class="twocol">
      <div class="field"><div class="field-head"><label for="density">Air density · kg/m³</label></div><input type="number" id="density" value="1.225" min="0.1" max="3" step="0.001"></div>
      <div class="field"><div class="field-head"><label for="coverage">Interval coverage</label></div><select id="coverage"><option value="0.8">80%</option><option value="0.9" selected>90%</option><option value="0.95">95%</option></select></div>
    </div>
    <div class="info"><span class="data-label">Experimental data</span>
      <p><strong>Training:</strong> M25 wind tunnel testing, <b>24 February 2026</b>, Australian Centre for Sports Aerodynamics (<b>ACSA</b>).</p>
      <p><strong>Test conditions:</strong> nominal airspeeds <b>11, 17 and 19 m/s</b>; yaw <b>−12° to +12°</b>.</p>
      <p>Actual measured input bounds are shown alongside the sliders.</p>
    </div>
  </section>
  <section class="panel">
    <h2>Prediction</h2>
    <div class="error" id="error"></div>
    <div class="metrics">
      <div class="metric primary"><div class="meta">Downforce area · C<sub>L</sub>A</div><div><span class="value" id="cla">—</span><span class="unit">m²</span></div><div class="detail" id="cla-range">Independent wind tunnel error interval</div></div>
      <div class="metric"><div class="meta">Downforce · D</div><div><span class="value" id="force">—</span><span class="unit">N</span></div><div class="detail" id="force-range">D = ½ρV²C<sub>L</sub>A</div></div>
      <div class="metric"><div class="meta">Independent WT RMSE</div><div><span class="value" id="rmse">—</span><span class="unit">m²</span></div><div class="detail">Separate pressure-tap configuration</div></div>
      <div class="metric"><div class="meta">Validation mean bias</div><div><span class="value" id="bias">—</span><span class="unit">m²</span></div><div class="detail">Prediction − measurement</div></div>
    </div>
    <div class="smallbar"><h2>Predicted Yaw Response</h2><span><span class="legend"><b></b> Model</span>&nbsp; <span class="legend uncert"><b></b> Error band</span></span></div>
    <svg class="plot" id="chart" viewBox="0 0 720 310" role="img" aria-label="Predicted downforce area versus yaw angle"></svg>
    <div class="note"><strong>Uncertainty interpretation:</strong> These are empirical error bands from 21 independent wind tunnel measurements, not verified per-prediction confidence probabilities. The validation used a different pressure-tap configuration, which may cause a systematic offset. Prediction outside the measured speed and yaw range is disabled.</div>
  </section>
</div>
<div class="footer"><span><strong>© 2026 Monash Motorsport</strong> · Wind tunnel test data: 24 February 2026 · ACSA · Interface v2.1</span><span id="details">Local predictor</span></div>
</div>
<script>
const E=id=>document.getElementById(id);
let bounds=null, debounce=null, revision=0;
const fmt=(n,d=3)=>Number(n).toFixed(d);
function pair(number, slider){
  E(number).addEventListener('input',()=>{E(slider).value=E(number).value; schedule();});
  E(slider).addEventListener('input',()=>{E(number).value=E(slider).value; schedule();});
}
pair('speed','speed-slider'); pair('yaw','yaw-slider');
for(const id of ['density','coverage','model']) E(id).addEventListener('input',schedule);
function schedule(){clearTimeout(debounce);debounce=setTimeout(update,110);}
function draw(result){
  const data=result.curve, p=result.prediction, band=result.error_band_m2, svg=E('chart');
  const W=720,H=310,L=67,R=22,T=24,B=51;
  const xMin=data[0].yaw_deg,xMax=data[data.length-1].yaw_deg;
  let vals=data.flatMap(d=>[d.cla_m2-band,d.cla_m2+band]);
  vals.push(p.cla_m2-band,p.cla_m2+band);
  let yMin=Math.min(...vals),yMax=Math.max(...vals);let pad=(yMax-yMin)*.14||.05;
  yMin-=pad;yMax+=pad;
  const sx=x=>L+(x-xMin)/(xMax-xMin)*(W-L-R);
  const sy=y=>H-B-(y-yMin)/(yMax-yMin)*(H-T-B);
  const upper=data.map(d=>`${sx(d.yaw_deg).toFixed(1)},${sy(d.cla_m2+band).toFixed(1)}`).join(' ');
  const lower=[...data].reverse().map(d=>`${sx(d.yaw_deg).toFixed(1)},${sy(d.cla_m2-band).toFixed(1)}`).join(' ');
  const line=data.map(d=>`${sx(d.yaw_deg).toFixed(1)},${sy(d.cla_m2).toFixed(1)}`).join(' ');
  let s=`<rect x="0" y="0" width="${W}" height="${H}" fill="#fff"/>`;
  for(let i=0;i<=5;i++){
    const y=yMin+(yMax-yMin)*i/5,yy=sy(y);
    s+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#e2eaf0" stroke-width="1"/><text x="${L-12}" y="${yy+4}" text-anchor="end" font-family="Arial,system-ui" font-size="11" fill="#627889">${fmt(y,2)}</text>`;
  }
  for(let i=0;i<=6;i++){
    const x=xMin+(xMax-xMin)*i/6,xx=sx(x);
    s+=`<line x1="${xx}" x2="${xx}" y1="${T}" y2="${H-B}" stroke="#edf1f5"/><text x="${xx}" y="${H-B+22}" text-anchor="middle" font-family="Arial,system-ui" font-size="11" fill="#627889">${fmt(x,0)}°</text>`;
  }
  s+=`<polygon points="${upper} ${lower}" fill="#3aaae3" opacity=".22"/><polyline points="${line}" fill="none" stroke="#0876b5" stroke-width="3" stroke-linejoin="round"/>`;
  const px=sx(p.yaw_deg),py=sy(p.cla_m2);
  s+=`<line x1="${px}" x2="${px}" y1="${T}" y2="${H-B}" stroke="#ee9050" stroke-dasharray="5 5" opacity=".8"/><circle cx="${px}" cy="${py}" r="6" fill="#ee9050" stroke="#fff" stroke-width="2"/><text x="${W/2}" y="${H-8}" text-anchor="middle" font-family="Arial,system-ui" font-size="12" fill="#344f62">Yaw angle β (degrees)</text><text x="17" y="${H/2}" text-anchor="middle" transform="rotate(-90 17 ${H/2})" font-family="Arial,system-ui" font-size="12" fill="#344f62">Downforce area CₗA (m²)</text>`;
  svg.innerHTML=s;
}
async function update(){
  if(!bounds)return;
  const current=++revision;
  const params=new URLSearchParams({model:E('model').value,speed:E('speed').value,yaw:E('yaw').value,density:E('density').value,coverage:E('coverage').value});
  try{
    const response=await fetch('/api/predict?'+params.toString(),{cache:'no-store'}),result=await response.json();
    if(current!==revision)return;
    if(!response.ok)throw Error(result.error||'Unable to predict');
    E('error').classList.remove('active');E('status').className='status';E('status').textContent='● Model connected';
    const p=result.prediction,b=result.error_band_m2,f=result.force_error_band_n;
    E('cla').textContent=fmt(p.cla_m2,3);
    E('force').textContent=Math.round(p.downforce_n).toLocaleString();
    E('cla-range').textContent=`${Math.round(result.coverage*100)}% empirical band: ${fmt(p.cla_m2-b,3)} – ${fmt(p.cla_m2+b,3)} m²`;
    E('force-range').textContent=`Corresponding range: ${Math.round(p.downforce_n-f).toLocaleString()} – ${Math.round(p.downforce_n+f).toLocaleString()} N`;
    E('rmse').textContent=fmt(result.validation.rmse_m2,3);
    E('bias').textContent=(result.validation.bias_m2>=0?'+':'')+fmt(result.validation.bias_m2,3);
    E('details').textContent=`${result.validation.count} independent validation points · ${result.model}`;
    draw(result);
  }catch(err){if(current!==revision)return;E('error').textContent=err.message;E('error').classList.add('active');E('status').className='status err';E('status').textContent='Check inputs';}
}
async function start(){
  try{
    const response=await fetch('/api/meta');if(!response.ok)throw Error('Cannot reach the Python model');
    bounds=await response.json();
    for(const [input,slider,lower,upper,rangeId] of [['speed','speed-slider',bounds.speed_min,bounds.speed_max,'speed-range'],['yaw','yaw-slider',bounds.yaw_min,bounds.yaw_max,'yaw-range']]){
      for(const id of [input,slider]){E(id).min=lower;E(id).max=upper;}
      E(rangeId).textContent=`${fmt(lower,1)} to ${fmt(upper,1)}`;
    }
    E('speed').value=E('speed-slider').value=Math.min(bounds.speed_max,Math.max(bounds.speed_min,17));
    E('yaw').value=E('yaw-slider').value=0;
    E('model').value=bounds.default_model;
    await update();
  }catch(err){E('error').textContent=err.message;E('error').classList.add('active');E('status').className='status err';E('status').textContent='Model unavailable';}
}
start();
</script></body></html>'''


def load_existing_model():
    """Execute the user's own script once and reuse its fitted predict_model()."""
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Cannot find {MODEL_FILE}. Place model_gui.py in scripts/ next to downforce_model.py."
        )
    sys.path.insert(0, str(HERE))
    os.chdir(ROOT)  # Original script reads data/m25_re_yaw_sweep.csv from root.

    original_show = plt.show
    try:
        plt.show = lambda *args, **kwargs: plt.close("all")
        with contextlib.redirect_stdout(sys.stdout):
            fitted = runpy.run_path(str(MODEL_FILE), run_name="__main__")
    finally:
        plt.show = original_show
        plt.close("all")

    if not callable(fitted.get("predict_model")):
        raise RuntimeError("Your downforce_model.py must define predict_model(model, X) in Part 9.")
    if "x_train" not in fitted or "X_val1" not in fitted or "Y_val1" not in fitted:
        raise RuntimeError("Training inputs and independent WT validation arrays must exist in downforce_model.py.")
    return fitted


def build_app_state(ns):
    """Use the existing validation residuals; do not fit a second predictor."""
    predict = ns["predict_model"]
    X_val = np.asarray(ns["X_val1"], dtype=float)
    Y_val = np.asarray(ns["Y_val1"], dtype=float)
    x_train = np.asarray(ns["x_train"], dtype=float)
    validation = {}
    for model in MODEL_NAMES:
        predicted = np.asarray(predict(model, X_val), dtype=float)
        residuals = predicted - Y_val
        if len(residuals) < 1 or not np.all(np.isfinite(residuals)):
            raise RuntimeError(f"No valid independent WT residuals for {model}.")
        validation[model] = {
            "abs_errors": np.sort(np.abs(residuals)),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "bias": float(np.mean(residuals)),
            "count": int(len(residuals)),
        }
    return {
        "predict": predict,
        "validation": validation,
        "speed_min": float(x_train[:, 0].min()),
        "speed_max": float(x_train[:, 0].max()),
        "yaw_min": float(x_train[:, 1].min()),
        "yaw_max": float(x_train[:, 1].max()),
        "default_model": ns.get("best_model") if ns.get("best_model") in MODEL_NAMES else "Quadratic",
    }


def finite_value(params, name):
    raw = params.get(name, [None])[0]
    if raw is None:
        raise ValueError(f"Missing {name}.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"Invalid {name}.") from ex
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def calculate_view(state, params):
    """Only invokes the original prediction function and formats its results."""
    model = params.get("model", ["Quadratic"])[0]
    if model not in MODEL_NAMES:
        raise ValueError("Unknown regression model.")
    speed = finite_value(params, "speed")
    yaw = finite_value(params, "yaw")
    density = finite_value(params, "density")
    coverage = finite_value(params, "coverage")
    if not state["speed_min"] <= speed <= state["speed_max"]:
        raise ValueError(f"Airspeed outside measured range: {state['speed_min']:.2f}–{state['speed_max']:.2f} m/s.")
    if not state["yaw_min"] <= yaw <= state["yaw_max"]:
        raise ValueError(f"Yaw outside measured range: {state['yaw_min']:.1f}–{state['yaw_max']:.1f}°.")
    if not 0 < density <= 3:
        raise ValueError("Air density must be greater than 0 and at most 3 kg/m³.")
    if coverage not in (0.8, 0.9, 0.95):
        raise ValueError("Select 80%, 90%, or 95% empirical coverage.")

    # The only downforce-area prediction call: the fitted model in Part 9.
    point = float(state["predict"](model, np.array([[speed, yaw]], dtype=float))[0])
    yaw_samples = np.linspace(state["yaw_min"], state["yaw_max"], 121)
    inputs = np.column_stack((np.full_like(yaw_samples, speed), yaw_samples))
    curve_predictions = np.asarray(state["predict"](model, inputs), dtype=float)
    if not math.isfinite(point) or not np.all(np.isfinite(curve_predictions)):
        raise RuntimeError("The existing model returned a non-finite prediction.")

    # Empirical quantile of INDEPENDENT validation absolute errors, not a
    # new model. Finite-sample ceil((n+1)*coverage) order statistic.
    val = state["validation"][model]
    n = val["count"]
    k = min(n, math.ceil((n + 1) * coverage))
    band = float(val["abs_errors"][k - 1])
    pressure = 0.5 * density * speed ** 2
    return {
        "model": model,
        "coverage": coverage,
        "prediction": {
            "speed_mps": speed,
            "yaw_deg": yaw,
            "cla_m2": point,
            "dynamic_pressure_pa": pressure,
            "downforce_n": pressure * point,
        },
        "error_band_m2": band,
        "force_error_band_n": pressure * band,
        "validation": {
            "rmse_m2": val["rmse"],
            "bias_m2": val["bias"],
            "count": n,
        },
        "curve": [
            {"yaw_deg": float(beta), "cla_m2": float(pred)}
            for beta, pred in zip(yaw_samples, curve_predictions)
        ],
    }


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path in ("/", "/index.html"):
                    self.send_bytes(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
                    return
                if parsed.path == "/api/meta":
                    keys = ("speed_min", "speed_max", "yaw_min", "yaw_max", "default_model")
                    self.send_json(200, {key: state[key] for key in keys})
                    return
                if parsed.path == "/api/predict":
                    params = parse_qs(parsed.query, keep_blank_values=True)
                    self.send_json(200, calculate_view(state, params))
                    return
                self.send_json(404, {"error": "Not found."})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                self.send_json(500, {"error": f"Model evaluation failed: {error}"})

        def send_json(self, status, obj):
            self.send_bytes(status, "application/json; charset=utf-8", json.dumps(obj).encode("utf-8"))

        def send_bytes(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    print("Loading your existing downforce_model.py (and its validations)…")
    namespace = load_existing_model()
    state = build_app_state(namespace)
    server = ThreadingHTTPServer((HOST, PORT), make_handler(state))
    url = f"http://{HOST}:{server.server_address[1]}/?interface=v2_1"
    print(f"\nM25 predictor: {url}\nPress Ctrl+C to close.\n")
    with contextlib.suppress(Exception):
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
