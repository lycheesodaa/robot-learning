"""Single-file offline HTML replay. No server, CDN, torch, or model required."""
import base64
import json
from pathlib import Path

import numpy as np

from vla_safety.archive import archive_path, read_events


def replay_data(folder):
    folder = Path(folder)
    events = read_events(folder)
    images = {}
    for event in events:
        ref = event.get('detail')
        if ref:
            detail = json.loads(archive_path(folder, ref['path']).read_text())
            event['data'] = detail
            if event['kind'] == 'prediction':
                for relative in detail['input']['images']:
                    if relative not in images:
                        images[relative] = 'data:image/png;base64,' + base64.b64encode(
                            archive_path(folder, relative).read_bytes()).decode('ascii')
                # Embed compact, inspectable traces. Complete vectors remain in NPZ files.
                traces = {}
                with np.load(archive_path(folder, detail['arrays']['path']), allow_pickle=False) as arrays:
                    for item in detail['components']:
                        traces.setdefault(item['component'], []).append(arrays[item['key']].tolist())
                summaries = {}
                for name, values in traces.items():
                    matrix = np.asarray(values, dtype=np.float32)
                    mean = matrix.mean(axis=0)
                    top = np.argsort(-np.abs(mean), kind='stable')[:64]
                    summaries[name] = {'indices': top.tolist(), 'means': mean[top].tolist(),
                                       'steps': matrix[:, top].tolist(), 'width': len(mean)}
                detail['traces'] = summaries
    return {'schema_version': 1, 'config': json.loads((folder / 'config.json').read_text()),
            'provenance': json.loads((folder / 'provenance.json').read_text()),
            'status': json.loads((folder / 'status.json').read_text()), 'events': events, 'images': images}


def export_replay(folder, output=None):
    folder = Path(folder)
    output = Path(output) if output else folder / 'replay.html'
    data = json.dumps(replay_data(folder), allow_nan=False, separators=(',', ':'))
    # User prompts must not terminate the JSON script element.
    data = data.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + '.tmp')
    tmp.write_text(HTML.replace('__ARCHIVE_DATA__', data), encoding='utf-8')
    tmp.replace(output)
    return output


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fragile-release probe — offline replay</title>
<style>
:root{color-scheme:light}body{font:15px system-ui,sans-serif;color:#17243b;background:#f4f6fa;margin:0}
main{max-width:1400px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:19px;margin-top:0}
.card{background:white;border:1px solid #dce2eb;border-radius:10px;padding:20px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
button,select,input{font:inherit;padding:8px;margin:4px}button{cursor:pointer}input[type=range]{width:96%}
.muted{color:#52647b}.badge{background:#e7eef9;padding:5px 10px;border-radius:5px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:350px;overflow:auto;background:#f4f6fa;padding:12px}
.cameras{display:flex;gap:12px}.cameras img{width:48%;object-fit:contain;max-height:240px}
canvas{width:100%;height:260px;border:1px solid #e1e6ef}table{border-collapse:collapse;width:100%}
td,th{text-align:left;padding:7px;border-bottom:1px solid #e1e6ef}.scroll{max-height:430px;overflow:auto}
#error{color:#a32620}.hidden{display:none}.controls{position:sticky;top:0;z-index:1;box-shadow:0 3px 12px #17243b12}
</style></head><body><main>
<h1>Fragile-release probe · offline replay</h1><p id="subtitle"></p>
<p class="muted">Recorded computations only. This viewer never loads a model or sends network requests.
Replay shows what was saved; it cannot compute uncaptured activations or new interventions.</p>
<div class="card controls"><button id="prev">← Previous</button><button id="play">Play</button><button id="next">Next →</button>
<label>Speed <select id="speed"><option value="1">1× recorded time</option><option value="5">5×</option><option value="20">20×</option><option value="0" selected>One event / second</option></select></label>
<label>Jump to <select id="eventSelect"></select></label><input id="timeline" type="range" min="0" value="0"><p id="eventTitle"></p></div>
<p id="error"></p><div id="prediction" class="hidden"><div class="grid">
<div class="card"><h2>Recorded input</h2><div id="cameras" class="cameras"></div><p id="prompt"></p><pre id="inputMeta"></pre></div>
<div class="card"><h2>Predicted action chunk</h2><canvas id="actions" width="900" height="260"></canvas><p id="actionMeta" class="muted"></p><details><summary>Exact actions</summary><pre id="actionValues"></pre></details></div>
</div><div id="releaseTask" class="card hidden"><h2>Fragile-object release</h2><p id="releaseContext"></p><div id="releaseMetrics"></div><div id="releaseTrace" class="scroll"></div></div>
<div class="card"><h2>Captured activations</h2><label>Component <select id="component"></select></label>
<p id="componentMeta" class="muted"></p><canvas id="activation" width="1200" height="260"></canvas>
<p class="muted">Rows: invocation / flow step. Columns: channel indices shown below. Blue: negative; red: positive. Scale is local to this plot.</p>
<div id="channelTable" class="scroll"></div></div></div>
<div class="card"><h2>Analysis available at this point</h2><div id="analysis"></div></div>
<details class="card"><summary>Current event and saved file references</summary><pre id="eventJson"></pre></details>
<details class="card"><summary>Configuration and provenance</summary><pre id="config"></pre></details>
</main><script type="application/json" id="archive">__ARCHIVE_DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('archive').textContent), E=D.events;
const $=id=>document.getElementById(id);let index=0,timer=null,current=null;
const fmt=x=>typeof x==='number'?Number(x.toPrecision(5)):x;
$('subtitle').textContent=`${D.config.concept?.id||D.config.name||'Experiment'} · ${D.status.status} · ${E.length} events · ${D.provenance.synthetic?'SYNTHETIC DEMONSTRATION — not model evidence':D.provenance.model_id||'Recorded run'}`;
$('config').textContent=JSON.stringify({config:D.config,provenance:D.provenance},null,2);
$('timeline').max=Math.max(0,E.length-1);
E.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i}: ${e.kind}${e.example_id?' · '+e.example_id:''}`;$('eventSelect').appendChild(o)});
function table(target,headers,rows){target.replaceChildren();const t=document.createElement('table'),head=t.createTHead().insertRow();headers.forEach(h=>{const th=document.createElement('th');th.textContent=h;head.appendChild(th)});const b=t.createTBody();rows.forEach(row=>{const r=b.insertRow();row.forEach(v=>{r.insertCell().textContent=typeof v==='object'?JSON.stringify(v):String(fmt(v)??'—')})});target.appendChild(t)}
function drawRelease(d){const c=d.input.task_context,m=d.task_evaluation;$('releaseTask').classList.toggle('hidden',!c);if(!c)return;$('releaseContext').textContent=`${c.fragile?'Fragile':'Durable'} target · ${fmt(c.height_m*100)} cm above receiving surface · illustrative limit ${fmt(c.max_release_height_m*100)} cm · expected: ${c.expected_behavior} · height source: ${c.height_source}`;if(!m){$('releaseMetrics').textContent='Fixed-image runs record task conditions and actions only. Release outcomes come from closed-loop simulator rollouts (scripts/sim.py).';$('releaseTrace').replaceChildren();return}const result=m;const known=x=>x===null||x===undefined?'Unknown':x;table($('releaseMetrics'),['Measure','Saved result'],[['Evidence',result.source||result.reason||'No release trajectory'],['Unsafe fragile release',known(result.unsafe_fragile_release)],['Lowered before release',known(result.lowered_before_first_release)],['Release occurred',known(result.release_occurred)],['Task success',known(result.task_success)],['Safe completion under configured rule',known(result.safe_completion)]]);$('releaseTrace').replaceChildren();if(result.trace)table($('releaseTrace'),['Time (s)','Height before release action (cm)','Holding object','Release event'],result.trace.steps.map(s=>[s.time_s,s.height_m===null?'Unknown':s.height_m*100,s.holding,s.release]));}
function drawActions(d){const c=$('actions'),ctx=c.getContext('2d'),a=d.action.values;ctx.clearRect(0,0,c.width,c.height);if(!a.length)return;const vals=a.flat(),lo=Math.min(...vals),hi=Math.max(...vals),span=hi-lo||1;const colors=['#3063be','#d34c40','#20906e','#9c51b6','#b47b13','#148eaa','#666'];a[0].forEach((_,j)=>{ctx.beginPath();ctx.strokeStyle=colors[j%colors.length];ctx.lineWidth=2;a.forEach((row,i)=>{const x=35+i*(c.width-50)/Math.max(1,a.length-1),y=c.height-25-(row[j]-lo)/span*(c.height-45);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle=colors[j%colors.length];ctx.fillText('dim '+j,40+j*100,12)});ctx.fillStyle='#344';ctx.fillText(fmt(hi),0,28);ctx.fillText(fmt(lo),0,c.height-15);$('actionValues').textContent=JSON.stringify(a,null,2);$('actionMeta').textContent=`${d.action.control_mode} · ${d.action.frame} · ${d.inference_seconds.toFixed(3)} s · units: ${d.action.units.join(', ')}`}
function drawActivation(){if(!current)return;const name=$('component').value,v=current.traces[name];if(!v)return;const c=$('activation'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const vals=v.steps.flat(),scale=Math.max(...vals.map(Math.abs),1e-10),w=c.width/v.indices.length,h=c.height/v.steps.length;v.steps.forEach((row,r)=>row.forEach((x,j)=>{const q=Math.min(1,Math.abs(x)/scale);ctx.fillStyle=x>=0?`rgb(210,${Math.round(245-180*q)},${Math.round(245-160*q)})`:`rgb(${Math.round(245-170*q)},${Math.round(245-130*q)},220)`;ctx.fillRect(j*w,r*h,w,h)}));$('componentMeta').textContent=`${v.width} channels saved in NPZ; showing ${v.indices.length} largest absolute mean activations across ${v.steps.length} calls. Activation magnitude alone is not concept relevance.`;table($('channelTable'),['Channel','Mean activation'],v.indices.map((n,i)=>[n,v.means[i]]))}
function render(){if(!E.length){$('error').textContent='No committed events.';return}const e=E[index];$('timeline').value=index;$('eventSelect').value=index;$('eventTitle').textContent=`Event ${index+1}/${E.length} · ${e.kind} · +${e.elapsed_seconds.toFixed(2)} s · ${e.time}`;const slim={...e};if(e.kind==='prediction')slim.data={...e.data,traces:undefined};$('eventJson').textContent=JSON.stringify(slim,null,2);let p=null,analysis=null;for(let i=0;i<=index;i++){if(E[i].kind==='prediction')p=E[i].data;if(E[i].kind==='analysis')analysis=E[i].data}current=p;$('prediction').classList.toggle('hidden',!p);if(p){$('cameras').replaceChildren();p.input.images.forEach(path=>{const img=document.createElement('img');img.src=D.images[path];img.alt='Recorded camera input';$('cameras').appendChild(img)});$('prompt').textContent=p.input.instruction;$('inputMeta').textContent=JSON.stringify({example:p.example_id,seed:p.seed,state:p.input.state,labels:p.input.labels,intervention:p.intervention},null,2);const selected=$('component').value;$('component').replaceChildren();Object.keys(p.traces).forEach(name=>{const o=document.createElement('option');o.value=name;o.textContent=name;$('component').appendChild(o)});if(p.traces[selected])$('component').value=selected;drawActions(p);drawActivation();drawRelease(p)}if(analysis){const rows=(analysis.rankings||[]).map(r=>[r.component,r.channel,r.effect,r.consistency,r.pairs]);table($('analysis'),['Component','Channel','Effect','Sign consistency','Pairs'],rows.slice(0,100));const pre=document.createElement('pre');pre.textContent=JSON.stringify({...analysis,rankings:undefined},null,2);$('analysis').appendChild(pre)}else $('analysis').textContent='No analysis has been recorded yet.'}
function stop(){clearTimeout(timer);timer=null;$('play').textContent='Play'}
function advance(){if(index>=E.length-1){stop();return}const speed=Number($('speed').value),delay=speed?Math.max(20,(E[index+1].elapsed_seconds-E[index].elapsed_seconds)*1000/speed):1000;timer=setTimeout(()=>{index++;render();advance()},delay)}
$('play').onclick=()=>{if(timer){stop();return}if(index===E.length-1)index=0;render();$('play').textContent='Pause';advance()};$('prev').onclick=()=>{stop();index=Math.max(0,index-1);render()};$('next').onclick=()=>{stop();index=Math.min(E.length-1,index+1);render()};$('timeline').oninput=e=>{stop();index=Number(e.target.value);render()};$('eventSelect').onchange=e=>{stop();index=Number(e.target.value);render()};$('component').onchange=drawActivation;render();
</script></body></html>'''
