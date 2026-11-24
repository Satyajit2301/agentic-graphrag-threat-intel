# ============ BLOCK 17b : check the figures and tables of the report and pack them into ONE zip (read-only; makes one zip) ============
# Needs: Block 1 run in this session. Run it after Block 15b (and 16). It lists every figure/table that main.tex expects,
# says which ones exist on Drive, and zips report_assets/figures + report_assets/tables into report_assets/report_assets.zip.
import zipfile
FIGS = [
   'fig_dataset_cvss_hist', 'fig_dataset_pub_year', 'fig_dataset_severity', 'fig_dataset_top_cwe', 'fig_dataset_top_products',
   'fig_eda_correlation', 'fig_eda_cvss_vs_epss', 'fig_eda_epss_by_kev', 'fig_eda_missing_values', 'fig_graph_degree_ccdf',
   'fig_graph_example_subgraph', 'fig_graph_label_coverage', 'fig_graph_schema', 'fig_graph_techniques_per_cve',
   'fig_graph_top_techniques', 'fig_lp_confusion', 'fig_lp_feature_ablation', 'fig_lp_hits_at_k', 'fig_lp_labels_by_tactic',
   'fig_lp_learning_curve', 'fig_lp_mlp_training', 'fig_lp_pr', 'fig_lp_roc', 'fig_lp_similarity', 'fig_lp_technique_frequency',
   'fig_lp_tuning', 'fig_nvd_cves_per_year', 'fig_qa_cost_accuracy', 'fig_qa_score_by_hops', 'fig_qa_score_by_type',
   'fig_rag_recall_by_type', 'fig_rag_recall_vs_k', 'fig_rag_retriever_comparison',
]
TABS = [
   'table_benchmark', 'table_dataset_composition', 'table_eda_describe', 'table_eda_missing', 'table_lp_classification',
   'table_lp_confusion', 'table_lp_examples', 'table_lp_feature_ablation', 'table_lp_features', 'table_lp_learning_curve',
   'table_lp_per_technique', 'table_lp_ranking', 'table_lp_similarity', 'table_lp_split', 'table_lp_tuning', 'table_qa_by_type',
   'table_qa_outcomes', 'table_qa_summary', 'table_rag_corpus', 'table_rag_retrieval',
]
FD, TD = DIRS['figures'], DIRS['tables']
miss_f = [n for n in FIGS if not (FD / f'{n}.png').exists()]
miss_t = [n for n in TABS if not (TD / f'{n}.png').exists()]
print(f'figures expected by main.tex: {len(FIGS)} | found {len(FIGS) - len(miss_f)} | missing {len(miss_f)}')
print(f'tables  expected by main.tex: {len(TABS)} | found {len(TABS) - len(miss_t)} | missing {len(miss_t)}')
if miss_f: print('  MISSING figures:', ', '.join(miss_f))
if miss_t: print('  MISSING tables :', ', '.join(miss_t))
extra = sorted({p.stem for p in FD.glob('*.png')} - set(FIGS)) + sorted({p.stem for p in TD.glob('*.png')} - set(TABS))
print('  files on Drive that main.tex does not use (harmless):', ', '.join(extra) if extra else 'none')
ZIP = ROOT / 'report_assets' / 'report_assets.zip'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(FD.glob('*.png')): z.write(p, f'figures/{p.name}')
    for p in sorted(TD.glob('*.png')): z.write(p, f'tables/{p.name}')
print(f'\nzip written: {ZIP}  ({os.path.getsize(ZIP) / 1e6:.1f} MB)')
print('Download it from Drive (MyDrive/GraphRAG_ThreatIntel/report_assets/report_assets.zip) and unzip it NEXT TO main.tex:')
print('   report/main.tex   report/figures/<name>.png   report/tables/<name>.png      (the names in the tex are the names of the files)')
print('\nBLOCK 17b DONE')
