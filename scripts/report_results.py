#!/usr/bin/env python3
"""
report_results.py
-----------------
Đọc tất cả JSON kết quả từ thư mục results, sinh HTML report với:
  - Summary table (avg ± std, 5 seeds)
  - Bar chart Overall vs Cold (HR@20, MRR@20)
  - Training history (loss + val HR@20 theo epoch, nếu có)
  - Per-seed breakdown

Usage:
  python scripts/report_results.py
      --results_dir results
      --output results/report_cellphones_fulllen.html
      --title "CellPhones cold_20 — Full Session Length"

  # Chỉ đọc 1 subfolder:
  python scripts/report_results.py
      --results_dir results/baselines
      --output results/report_baselines.html
"""

import argparse
import json
import os
import glob
import statistics
from collections import defaultdict


# ─── Load + normalize ─────────────────────────────────────────────────────────

def load_result(fpath):
    with open(fpath) as f:
        d = json.load(f)

    r = {
        'file': os.path.basename(fpath),
        'model': None, 'dataset': None, 'seed': None,
        'overall': {}, 'cold': {},
        'n_total': 0, 'n_cold': 0,
        'best_epoch': None, 'n_epochs_run': None,
        'training_history': [],
    }

    if 'results' in d:
        # DC2R format
        r['model']    = d.get('model', 'DC2R')
        r['dataset']  = d.get('dataset', '')
        r['seed']     = d.get('seed')
        r['best_epoch']   = d.get('best_epoch')
        r['n_epochs_run'] = d.get('n_epochs_run')
        r['training_history'] = d.get('training_history', [])
        res = d['results']
        r['overall']  = res.get('overall', {})
        r['cold']     = res.get('cold', {})
        r['n_total']  = res.get('n_total', 0)
        r['n_cold']   = res.get('n_cold', 0)
    elif 'test' in d:
        # A11 / NirGNN / M2TRec format
        r['model']    = d.get('model', d.get('ablation', 'Unknown'))
        r['dataset']  = d.get('dataset', '')
        r['seed']     = d.get('seed')
        r['best_epoch']   = d.get('best_epoch')
        r['n_epochs_run'] = d.get('n_epochs', d.get('total_epochs'))
        r['training_history'] = d.get('training_history', [])
        t = d['test']
        r['n_total']  = t.get('n_overall', t.get('n_total', 0))
        r['n_cold']   = t.get('n_cold', 0)
        r['overall']  = {k: t.get(k, 0) for k in ['HR@10','HR@20','MRR@10','MRR@20']}
        r['cold']     = {
            'HR@10':  t.get('Cold_HR@10', 0),
            'HR@20':  t.get('Cold_HR@20', 0),
            'MRR@10': t.get('Cold_MRR@10', 0),
            'MRR@20': t.get('Cold_MRR@20', 0),
        }
    return r


def load_all(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, '**', '*.json'), recursive=True))
    rows = []
    for f in files:
        try:
            rows.append(load_result(f))
        except Exception as e:
            print(f'  SKIP {f}: {e}')
    print(f'Loaded {len(rows)} result files from {results_dir}')
    return rows


# ─── Aggregate across seeds ───────────────────────────────────────────────────

def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r['model'], r['dataset'])
        groups[key].append(r)

    agg = {}
    for key, seeds in groups.items():
        metrics = ['HR@10','HR@20','MRR@10','MRR@20']
        avg, std = {}, {}
        for m in metrics:
            vals_o = [s['overall'].get(m, 0) for s in seeds]
            vals_c = [s['cold'].get(m, 0) for s in seeds]
            avg[f'overall_{m}'] = statistics.mean(vals_o)
            avg[f'cold_{m}']    = statistics.mean(vals_c)
            std[f'overall_{m}'] = statistics.stdev(vals_o) if len(vals_o) > 1 else 0
            std[f'cold_{m}']    = statistics.stdev(vals_c) if len(vals_c) > 1 else 0

        n_total = seeds[0]['n_total']
        n_cold  = seeds[0]['n_cold']

        # training history: average loss + val_hr20 per epoch across seeds
        hist_by_epoch = defaultdict(list)
        for s in seeds:
            for ep in s['training_history']:
                e = ep['epoch']
                hist_by_epoch[e].append({
                    'loss':     ep.get('loss', 0),
                    'val_hr20': ep.get('val_hr20', 0),
                    'val_mrr20': ep.get('val_mrr20', 0),
                })
        avg_history = []
        for ep in sorted(hist_by_epoch):
            entries = hist_by_epoch[ep]
            avg_history.append({
                'epoch':      ep,
                'loss':       round(statistics.mean(e['loss'] for e in entries), 6),
                'val_hr20':   round(statistics.mean(e['val_hr20'] for e in entries), 6),
                'val_mrr20':  round(statistics.mean(e['val_mrr20'] for e in entries), 6),
            })

        best_epochs = [s['best_epoch'] for s in seeds if s['best_epoch'] is not None]

        agg[key] = {
            'model':   key[0], 'dataset': key[1],
            'n_seeds': len(seeds),
            'n_total': n_total, 'n_cold': n_cold,
            'avg': avg, 'std': std,
            'best_epoch_avg': round(statistics.mean(best_epochs), 1) if best_epochs else None,
            'avg_history': avg_history,
            'seeds': seeds,
        }
    return agg


# ─── HTML generation ──────────────────────────────────────────────────────────

COLORS = {
    0: '#2a78d6', 1: '#eb6834', 2: '#1baf7a', 3: '#eda100',
    4: '#e87ba4', 5: '#4a3aa7', 6: '#008300', 7: '#e34948',
}

def pct(v): return f'{v*100:.2f}%'
def pm(v):  return f'±{v*100:.2f}'


def generate_html(agg, title):
    models = sorted(agg.keys(), key=lambda k: -agg[k]['avg']['overall_HR@20'])
    model_list = [agg[k] for k in models]
    model_names = [m['model'] for m in model_list]
    colors = [COLORS[i % len(COLORS)] for i in range(len(model_list))]

    # ── data for bar charts ──
    def bar_data(metric_key):
        return [round(m['avg'][metric_key] * 100, 4) for m in model_list]

    # ── summary table rows ──
    def table_rows():
        rows = []
        for i, m in enumerate(model_list):
            a, s = m['avg'], m['std']
            color = colors[i]
            rows.append(f"""
            <tr>
              <td style="font-weight:500;color:{color}">{m['model']}</td>
              <td>{m['n_seeds']}</td>
              <td>{m['n_total']:,}</td>
              <td>{m['n_cold']:,} ({m['n_cold']/m['n_total']*100:.1f}%)</td>
              <td>{pct(a['overall_HR@10'])} {pm(s['overall_HR@10'])}</td>
              <td>{pct(a['overall_HR@20'])} {pm(s['overall_HR@20'])}</td>
              <td>{pct(a['overall_MRR@10'])} {pm(s['overall_MRR@10'])}</td>
              <td>{pct(a['overall_MRR@20'])} {pm(s['overall_MRR@20'])}</td>
              <td>{pct(a['cold_HR@10'])} {pm(s['cold_HR@10'])}</td>
              <td>{pct(a['cold_HR@20'])} {pm(s['cold_HR@20'])}</td>
              <td>{pct(a['cold_MRR@10'])} {pm(s['cold_MRR@10'])}</td>
              <td>{pct(a['cold_MRR@20'])} {pm(s['cold_MRR@20'])}</td>
              <td>{'ep '+str(m['best_epoch_avg']) if m['best_epoch_avg'] else '—'}</td>
            </tr>""")
        return '\n'.join(rows)

    # ── per-seed tables ──
    def seed_tables():
        blocks = []
        for i, m in enumerate(model_list):
            color = colors[i]
            seed_rows = []
            for s in sorted(m['seeds'], key=lambda x: x['seed'] or 0):
                a = s['overall']
                c = s['cold']
                best = f"ep {s['best_epoch']}" if s['best_epoch'] else '—'
                seed_rows.append(f"""
                <tr>
                  <td>{s['seed']}</td>
                  <td>{pct(a.get('HR@10',0))}</td><td>{pct(a.get('HR@20',0))}</td>
                  <td>{pct(a.get('MRR@10',0))}</td><td>{pct(a.get('MRR@20',0))}</td>
                  <td>{pct(c.get('HR@10',0))}</td><td>{pct(c.get('HR@20',0))}</td>
                  <td>{pct(c.get('MRR@10',0))}</td><td>{pct(c.get('MRR@20',0))}</td>
                  <td>{best}</td>
                </tr>""")
            blocks.append(f"""
            <div class="card" style="margin-bottom:1.5rem">
              <h3 style="color:{color};margin:0 0 0.75rem">{m['model']}
                <span style="font-size:13px;font-weight:400;color:#888"> — {m['n_seeds']} seeds</span>
              </h3>
              <div style="overflow-x:auto">
              <table class="tbl">
                <thead><tr>
                  <th>Seed</th>
                  <th>HR@10</th><th>HR@20</th><th>MRR@10</th><th>MRR@20</th>
                  <th>Cold HR@10</th><th>Cold HR@20</th><th>Cold MRR@10</th><th>Cold MRR@20</th>
                  <th>Best ep</th>
                </tr></thead>
                <tbody>{''.join(seed_rows)}</tbody>
              </table>
              </div>
            </div>""")
        return '\n'.join(blocks)

    # ── training history charts ──
    def history_charts():
        charts = []
        chart_id = 0
        for i, m in enumerate(model_list):
            if not m['avg_history']:
                continue
            color = colors[i]
            epochs  = [h['epoch'] for h in m['avg_history']]
            losses  = [h['loss'] for h in m['avg_history']]
            val_hr  = [round(h['val_hr20']*100,4) for h in m['avg_history']]
            val_mrr = [round(h['val_mrr20']*100,4) for h in m['avg_history']]
            best_ep = m['best_epoch_avg']

            cid_loss = f'hist_loss_{chart_id}'
            cid_val  = f'hist_val_{chart_id}'
            chart_id += 1

            charts.append(f"""
            <div class="card" style="margin-bottom:2rem">
              <h3 style="color:{color};margin:0 0 1rem">{m['model']} — Training History
                <span style="font-size:13px;font-weight:400;color:#888"> (avg {m['n_seeds']} seeds
                  {', best ep≈'+str(best_ep) if best_ep else ''})</span>
              </h3>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem">
                <div>
                  <p style="font-size:13px;color:#666;margin:0 0 0.5rem">Train Loss</p>
                  <div style="position:relative;height:220px">
                    <canvas id="{cid_loss}" role="img" aria-label="Training loss curve for {m['model']}"></canvas>
                  </div>
                </div>
                <div>
                  <p style="font-size:13px;color:#666;margin:0 0 0.5rem">Val HR@20 & MRR@20 (%)</p>
                  <div style="position:relative;height:220px">
                    <canvas id="{cid_val}" role="img" aria-label="Validation HR@20 curve for {m['model']}"></canvas>
                  </div>
                </div>
              </div>
            </div>
            <script>
            (function(){{
              var epochs = {json.dumps(epochs)};
              var losses = {json.dumps(losses)};
              var val_hr = {json.dumps(val_hr)};
              var val_mrr = {json.dumps(val_mrr)};
              var bestEp = {json.dumps(best_ep)};
              var color = '{color}';

              new Chart(document.getElementById('{cid_loss}'), {{
                type:'line', data:{{
                  labels: epochs,
                  datasets:[{{label:'loss',data:losses,borderColor:color,backgroundColor:color+'22',
                    borderWidth:2,pointRadius:2,fill:true,tension:0.3}}]
                }},
                options:{{responsive:true,maintainAspectRatio:false,
                  plugins:{{legend:{{display:false}}}},
                  scales:{{x:{{title:{{display:true,text:'Epoch',font:{{size:11}}}}}},
                           y:{{title:{{display:true,text:'Loss',font:{{size:11}}}}}}}}
                }}
              }});

              new Chart(document.getElementById('{cid_val}'), {{
                type:'line', data:{{
                  labels: epochs,
                  datasets:[
                    {{label:'Val HR@20',data:val_hr,borderColor:color,borderWidth:2,pointRadius:2,tension:0.3}},
                    {{label:'Val MRR@20',data:val_mrr,borderColor:color+'88',borderWidth:2,
                      pointRadius:2,tension:0.3,borderDash:[4,3]}}
                  ]
                }},
                options:{{responsive:true,maintainAspectRatio:false,
                  plugins:{{legend:{{labels:{{boxWidth:12,font:{{size:11}}}}}}}},
                  scales:{{x:{{title:{{display:true,text:'Epoch',font:{{size:11}}}}}},
                           y:{{title:{{display:true,text:'%',font:{{size:11}}}}}}}}
                }}
              }});
            }})();
            </script>""")
        return '\n'.join(charts) if charts else '<p style="color:#888">Không có training history.</p>'

    # ── bar chart data ──
    bar_labels   = json.dumps(model_names)
    bar_colors   = json.dumps(colors)
    o_hr20  = json.dumps(bar_data('overall_HR@20'))
    c_hr20  = json.dumps(bar_data('cold_HR@20'))
    o_mrr20 = json.dumps(bar_data('overall_MRR@20'))
    c_mrr20 = json.dumps(bar_data('cold_MRR@20'))
    o_hr10  = json.dumps(bar_data('overall_HR@10'))
    c_hr10  = json.dumps(bar_data('cold_HR@10'))

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  body{{font-family:Arial,sans-serif;margin:0;padding:2rem;background:#f8f8f6;color:#222;font-size:15px}}
  h1{{font-size:22px;font-weight:600;margin:0 0 0.25rem}}
  h2{{font-size:17px;font-weight:600;margin:2rem 0 1rem;border-bottom:2px solid #e0dfd8;padding-bottom:0.4rem}}
  h3{{font-size:15px;font-weight:600}}
  .card{{background:#fff;border-radius:10px;border:0.5px solid #e0dfd8;padding:1.25rem 1.5rem}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;margin-bottom:1.5rem}}
  .tbl{{width:100%;border-collapse:collapse;font-size:13px}}
  .tbl th,.tbl td{{padding:6px 10px;text-align:right;border-bottom:0.5px solid #eee}}
  .tbl th{{background:#f4f3ef;font-weight:600;text-align:center}}
  .tbl td:first-child,.tbl th:first-child{{text-align:left}}
  .meta{{font-size:13px;color:#888;margin-bottom:2rem}}
  .legend{{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:1rem;font-size:13px}}
  .legend-dot{{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px}}
  .tabs{{display:flex;gap:6px;margin-bottom:1rem}}
  .tab{{padding:5px 14px;border:0.5px solid #ccc;border-radius:6px;background:transparent;
        cursor:pointer;font-size:13px;color:#555}}
  .tab.active{{background:#222;color:#fff;border-color:#222}}
</style>
</head>
<body>

<h1>{title}</h1>
<p class="meta">Tự động sinh từ {sum(len(m['seeds']) for m in model_list)} file JSON — {len(model_list)} models</p>

<!-- ── Legend ── -->
<div class="legend">
  {''.join(f'<span><span class="legend-dot" style="background:{colors[i]}"></span>{m["model"]}</span>' for i,m in enumerate(model_list))}
</div>

<!-- ── Bar charts ── -->
<h2>So sánh Overall vs Cold</h2>
<div class="tabs" id="metricTabs">
  <button class="tab active" onclick="switchMetric('hr20')">HR@20</button>
  <button class="tab" onclick="switchMetric('mrr20')">MRR@20</button>
  <button class="tab" onclick="switchMetric('hr10')">HR@10</button>
</div>
<div class="grid2">
  <div class="card">
    <p style="font-size:13px;color:#666;margin:0 0 0.5rem">Overall</p>
    <div style="position:relative;height:260px">
      <canvas id="barOverall" role="img" aria-label="Overall metric comparison"></canvas>
    </div>
  </div>
  <div class="card">
    <p style="font-size:13px;color:#666;margin:0 0 0.5rem">Cold-start</p>
    <div style="position:relative;height:260px">
      <canvas id="barCold" role="img" aria-label="Cold-start metric comparison"></canvas>
    </div>
  </div>
</div>

<!-- ── Summary table ── -->
<h2>Summary Table (avg ± std, {model_list[0]['n_seeds'] if model_list else 5} seeds)</h2>
<div class="card" style="overflow-x:auto">
<table class="tbl">
  <thead><tr>
    <th>Model</th><th>Seeds</th><th>N total</th><th>N cold</th>
    <th>HR@10</th><th>HR@20</th><th>MRR@10</th><th>MRR@20</th>
    <th>Cold HR@10</th><th>Cold HR@20</th><th>Cold MRR@10</th><th>Cold MRR@20</th>
    <th>Best epoch</th>
  </tr></thead>
  <tbody>{table_rows()}</tbody>
</table>
</div>

<!-- ── Training History ── -->
<h2>Training History (avg across seeds)</h2>
{history_charts()}

<!-- ── Per-seed details ── -->
<h2>Per-seed Breakdown</h2>
{seed_tables()}

<script>
var LABELS  = {bar_labels};
var COLORS  = {bar_colors};
var DATA = {{
  hr20:  {{ overall: {o_hr20},  cold: {c_hr20}  }},
  mrr20: {{ overall: {o_mrr20}, cold: {c_mrr20} }},
  hr10:  {{ overall: {o_hr10},  cold: {c_hr10}  }},
}};
var YLABEL = {{hr20:'HR@20 (%)',mrr20:'MRR@20 (%)',hr10:'HR@10 (%)'}};

function makeBar(canvasId, data, label) {{
  var ctx = document.getElementById(canvasId);
  if (ctx._chart) ctx._chart.destroy();
  var ch = new Chart(ctx, {{
    type:'bar',
    data:{{labels:LABELS, datasets:[{{
      label:label, data:data,
      backgroundColor:COLORS,
      borderRadius:4, barPercentage:0.6
    }}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}}}},
      scales:{{
        x:{{grid:{{display:false}},ticks:{{font:{{size:11}}}}}},
        y:{{ticks:{{callback:v=>v.toFixed(2)+'%',font:{{size:11}}}},
            title:{{display:true,text:YLABEL[curMetric],font:{{size:11}}}}}}
      }}
    }}
  }});
  ctx._chart = ch;
}}

var curMetric = 'hr20';
function switchMetric(m) {{
  curMetric = m;
  document.querySelectorAll('.tab').forEach((b,i)=>
    b.classList.toggle('active',['hr20','mrr20','hr10'][i]===m));
  makeBar('barOverall', DATA[m].overall, 'Overall');
  makeBar('barCold',    DATA[m].cold,    'Cold');
}}

switchMetric('hr20');
</script>
</body>
</html>"""
    return html


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results',
                    help='Thư mục chứa JSON kết quả (tìm recursive)')
    ap.add_argument('--output', default='results/report_results.html',
                    help='Đường dẫn file HTML output')
    ap.add_argument('--title', default='CatPro-CL — Benchmark Results',
                    help='Tiêu đề report')
    args = ap.parse_args()

    rows = load_all(args.results_dir)
    if not rows:
        print('Không tìm thấy file JSON nào!')
        return

    agg = aggregate(rows)
    html = generate_html(agg, args.title)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report saved → {args.output}')


if __name__ == '__main__':
    main()
