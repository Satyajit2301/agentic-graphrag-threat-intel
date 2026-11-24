# ============ BLOCK 16 : Package the data files of the Streamlit app (read-only for everything else; makes one new folder and one zip) ============
# Needs: Block 1 run in this session. Everything else is read from Drive.
import zipfile, glob
PROC, RES = DIRS['proc'], DIRS['results']
OUT = ROOT / 'app_package' / 'data'; OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob('*'): f.unlink()
def sec(t): print('\n' + '=' * 6 + ' ' + t + ' ' + '=' * 6)
def size_mb(p): return os.path.getsize(p) / 1e6

# ---------- 1. the graph ----------
sec('Knowledge graph')
nodes = pd.read_parquet(PROC / 'kg_nodes.parquet'); edges = pd.read_parquet(PROC / 'kg_edges.parquet')
need_n, need_e = {'node_id', 'node_type', 'ext_id', 'name', 'text', 'attrs'}, {'src', 'rel', 'dst', 'src_type', 'dst_type', 'attr'}
assert need_n <= set(nodes.columns), f'kg_nodes.parquet lacks columns {need_n - set(nodes.columns)}'
assert need_e <= set(edges.columns), f'kg_edges.parquet lacks columns {need_e - set(edges.columns)}'
nodes[sorted(need_n | ({'source'} & set(nodes.columns)), key=list(nodes.columns).index)].to_parquet(OUT / 'kg_nodes.parquet', index=False)
edges[[c for c in edges.columns if c in need_e or c == 'source']].to_parquet(OUT / 'kg_edges.parquet', index=False)
print(f'kg_nodes.parquet {len(nodes):,} rows ({size_mb(OUT / "kg_nodes.parquet"):.1f} MB) | kg_edges.parquet {len(edges):,} rows ({size_mb(OUT / "kg_edges.parquet"):.1f} MB)')

# ---------- 2. link prediction ----------
sec('Link-prediction files')
pr = pd.read_parquet(PROC / 'lp_predictions.parquet')
assert {'cve_id', 'rank', 'technique_id', 'score', 'technique_name', 'has_expert_labels', 'model'} <= set(pr.columns), 'lp_predictions.parquet has unexpected columns'
pr.to_parquet(OUT / 'lp_predictions.parquet', index=False)
Z = np.load(PROC / 'lp_features.npz', allow_pickle=True)
info = pd.DataFrame({'cve_id': Z['cve_ids'].astype(str), 'split': Z['split'].astype(str)}); info.to_parquet(OUT / 'lp_cve_info.parquet', index=False)
shutil.copy(RES / 'lp_test_results.json', OUT / 'lp_test_results.json')
print(f'lp_predictions.parquet {len(pr):,} rows | lp_cve_info.parquet {len(info):,} CVEs, split counts: {info.split.value_counts().to_dict()} | lp_test_results.json copied')

# ---------- 3. benchmark, example questions and recorded runs ----------
sec('Benchmark, example questions, recorded runs')
BENCH_ = _load(PROC / 'benchmark.json')
shutil.copy(PROC / 'benchmark.json', OUT / 'benchmark.json')
ex = []
for t in sorted({q['template'] for q in BENCH_}):
    ex += sorted((q for q in BENCH_ if q['template'] == t and q['split'] == 'test'), key=lambda q: q['id'])[:2]       # two TEST questions per question type
KEEP = ('id', 'template', 'split', 'question', 'answer_kind', 'answer_spec', 'gold', 'entities', 'hops')
(OUT / 'example_questions.json').write_text(json.dumps([{k: q[k] for k in KEEP if k in q} for q in ex], ensure_ascii=False))
def find(prefix):
    tag = globals().get('MODEL_TAG')
    c = sorted(glob.glob(str(RES / f'answers_{prefix}__*.json')))
    pick = [p for p in c if tag and p.endswith(f'__{tag}.json')] or c
    return _load(Path(pick[0])) if pick else {}
AG, GR = find('agentic_final'), find('graphrag')
print(f'answer files found: agentic {len(AG)} answers, GraphRAG {len(GR)} answers')
rec = {}
for q in ex:
    r = {}
    if q['id'] in AG:
        a = AG[q['id']]; m = a.get('meta', {})
        r['agentic'] = {'raw': a.get('raw', ''), 'trace': [{'tool': t[0], 'args': t[1], 'result': t[2]} for t in m.get('trace', [])], 'handle': m.get('handle'), 'forced': bool(m.get('forced')),
                        'tool_errors': int(m.get('tool_errors', 0)), 'pt': int(a.get('pt', 0)), 'ct': int(a.get('ct', 0)), 'calls': int(a.get('calls', 0)), 'lat': float(a.get('lat', 0)), 'strict': bool(a.get('strict', False))}
    if q['id'] in GR:
        g = GR[q['id']]; r['graphrag'] = {'raw': g.get('raw', ''), 'pt': int(g.get('pt', 0)), 'ct': int(g.get('ct', 0)), 'lat': float(g.get('lat', 0)), 'strict': bool(g.get('strict', False))}
    if r: rec[q['id']] = r
(OUT / 'recorded_runs.json').write_text(json.dumps(rec, ensure_ascii=False))
print(f'{len(ex)} example questions ({", ".join(sorted({q["template"] for q in ex}))}) | recorded runs for {len(rec)} of them')
print('   ', pd.DataFrame([{'id': q['id'], 'question': q['question'][:95], 'agent exact': rec.get(q['id'], {}).get('agentic', {}).get('strict'), 'GraphRAG exact': rec.get(q['id'], {}).get('graphrag', {}).get('strict')} for q in ex]).to_string(index=False))

# ---------- 4. check that every file reads back, then zip ----------
sec('Check and zip')
for n in ('kg_nodes', 'kg_edges', 'lp_predictions', 'lp_cve_info'): assert len(pd.read_parquet(OUT / f'{n}.parquet')) > 0, n
for n in ('example_questions', 'recorded_runs', 'benchmark', 'lp_test_results'): json.loads((OUT / f'{n}.json').read_text())
tot = 0; big = []
ZIP = ROOT / 'app_package' / 'app_data.zip'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.glob('*')):
        z.write(f, f'data/{f.name}'); s = size_mb(f); tot += s; big += [f.name] if s > 90 else []
        print(f'  data/{f.name:28s} {s:7.2f} MB')
print(f'\ntotal {tot:.1f} MB (zip {size_mb(ZIP):.1f} MB)' + (f' | WARNING: GitHub refuses files over 100 MB: {big}' if big else ' | every file is far below the GitHub limit of 100 MB'))
print('\nfolder on Drive: MyDrive/GraphRAG_ThreatIntel/app_package/   ->  download app_data.zip from there and unzip it into the repository folder (the zip already contains the data/ folder).')
print('\nBLOCK 16 DONE')
