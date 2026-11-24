# ============ BLOCK 15b : Link prediction - extra results (curves, confusion matrix, examples, near-duplicates, feature-group ablation, learning curve) ============
# Needs: Block 1 AND Block 15 run in THIS Colab session (it re-uses their scores, models and helper functions). No language model needed. About 3-6 min on CPU.
# Rules kept: parts 1-4 only REPORT on the test CVEs (nothing is tuned on them); parts 6-7 use the VALIDATION CVEs only, so the test CVEs are not used for any decision.
from sklearn.metrics import roc_curve, precision_recall_curve
need = ['TEST_S', 'TEST_O', 'TEST_M', 'MODELS', 'LABEL', 'SEL', 'BEST', 'BEST_EP', 'FIT', 'prep_for', 'per_cve', 'Ymat', 'label_space', 'XG', 'TR', 'VA', 'TE', 'TRVA',
        'L_TR', 'L_TRVA', 'ids', 'true_sets', 'TNAME', 'G', 'popularity_scores', 'TT', 'save_table_wrapped', 'style', 'COLORS', 'MARK', 'INK', 'INK2']
miss = [v for v in need if v not in globals()]
assert not miss, f'Run Block 15 first in this session (missing: {miss})'
T0_ = time.time(); PROC = DIRS['proc']; EXTRA = {}
def section(t): print('\n' + '=' * 6, t, '=' * 6)
SHORT = {'popularity': 'popularity', 'LR': 'LR', 'kNN': 'kNN', 'SVM': 'SVM', 'RF': 'RF', 'MLP': 'MLP'}

# ---------------------------------------------------------------------------------------------------------
section('1. ROC and precision-recall curves (test CVEs; every pair CVE x technique is one case)')
Ytest = TEST_O[SEL]['_Y'].ravel().astype(int)
print(f'{len(TE)} test CVEs x {len(L_TRVA)} techniques = {Ytest.size:,} pairs, of which {int(Ytest.sum())} are true links ({100 * Ytest.mean():.2f}%)  (true links to techniques outside the label space are not in this grid)')
fig, ax = plt.subplots(figsize=(6.4, 5.4))
for m in MODELS:
    fpr, tpr, _ = roc_curve(Ytest, TEST_S[m].ravel()); ax.plot(fpr, tpr, color=COLORS[m], lw=2.2 if m == 'MLP' else 1.6, ls='--' if m == 'popularity' else '-', label=f'{LABEL[m]} (AUC {TEST_M[m]["roc_auc_micro"]:.3f})')
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=':'); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.set_title('ROC curves on the test CVEs (micro-average)'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
style(ax, 'both'); ax.legend(frameon=False, loc='lower right', fontsize=8.5, labelcolor=INK); fig.tight_layout()
save_fig(fig, 'fig_lp_roc', 'ROC curves of the five models and the popularity baseline on the test CVEs (micro-average over all CVE-technique pairs)')
fig, ax = plt.subplots(figsize=(6.4, 5.4))
for m in MODELS:
    p_, r_, _ = precision_recall_curve(Ytest, TEST_S[m].ravel()); ax.plot(r_, p_, color=COLORS[m], lw=2.2 if m == 'MLP' else 1.6, ls='--' if m == 'popularity' else '-', label=f'{LABEL[m]} (AP {TEST_M[m]["pr_auc_micro"]:.3f})')
ax.axhline(Ytest.mean(), color=INK2, lw=0.8, ls=':'); ax.text(0.98, Ytest.mean() + 0.01, 'share of true links (a random ranking)', ha='right', fontsize=8, color=INK2)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-recall curves on the test CVEs (micro-average)'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
style(ax, 'both'); ax.legend(frameon=False, loc='upper right', fontsize=8.5, labelcolor=INK); fig.tight_layout()
save_fig(fig, 'fig_lp_pr', 'Precision-recall curves of the five models and the popularity baseline on the test CVEs (micro-average over all CVE-technique pairs)')

# ---------------------------------------------------------------------------------------------------------
section('2. Confusion matrix of the top-3 predictions (a technique is "predicted" when it is among the 3 highest scores of a CVE)')
def conf3(m):
    o = TEST_O[m]; Y, top = o['_Y'], o['_rank'] < 3; n_unreach = float(o['n_all'].sum() - Y.sum())
    TP, FP = int((Y & top).sum()), int((~Y & top).sum()); FN, TN = int((Y & ~top).sum() + n_unreach), int((~Y & ~top).sum())     # links to techniques that no model can score count as missed
    return TP, FP, FN, TN
rows_c = []
for m in MODELS:
    TP, FP, FN, TN = conf3(m); rows_c.append({'Model': LABEL[m], 'TP': f'{TP:,}', 'FP': f'{FP:,}', 'FN': f'{FN:,}', 'TN': f'{TN:,}', 'Accuracy': f'{(TP + TN) / (TP + FP + FN + TN):.3f}',
                 'Precision': f'{TP / (TP + FP):.3f}', 'Recall': f'{TP / (TP + FN):.3f}', 'F1': f'{2 * TP / (2 * TP + FP + FN):.3f}'})
CM = pd.DataFrame(rows_c); print(CM.to_string(index=False)); save_table(CM, 'table_lp_confusion', 'Confusion counts of the top-3 predictions on the test CVEs (all CVE-technique pairs)')
_acc = [float(a) for a in CM['Accuracy']]; print(f'note: accuracy is {min(_acc):.3f} to {max(_acc):.3f} because almost all pairs are true negatives, so accuracy hides the differences; precision, recall and the ranking metrics are the main metrics.')
TP, FP, FN, TN = conf3(SEL); M2 = np.array([[TP, FN], [FP, TN]]); R2 = M2 / M2.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.6, 4.8)); ax.imshow(R2, cmap='Blues', vmin=0, vmax=1)
for i in range(2):
    for j in range(2): ax.text(j, i, f'{M2[i, j]:,}\n({100 * R2[i, j]:.1f}% of the row)', ha='center', va='center', fontsize=10.5, color='white' if R2[i, j] > 0.6 else INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(['predicted\n(in top 3)', 'not predicted']); ax.set_yticks([0, 1]); ax.set_yticklabels(['true link', 'no link']); ax.grid(False)
ax.set_title(f'Confusion matrix, {LABEL[SEL]} (top 3), test CVEs'); fig.tight_layout(); save_fig(fig, 'fig_lp_confusion', f'Confusion matrix of the top-3 predictions of the selected model ({SEL}) on the test CVEs')
EXTRA['confusion_selected'] = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

# ---------------------------------------------------------------------------------------------------------
section(f'3. Per-technique results of the selected model ({SEL}), the 10 most frequent techniques of the test CVEs')
o = TEST_O[SEL]; Yb, top3 = o['_Y'], o['_rank'] < 3; sup = Yb.sum(0); order = np.argsort(-sup)[:10]; rows_t = []
for j in order:
    tp = int((Yb[:, j] & top3[:, j]).sum()); fl = int(top3[:, j].sum()); nm = TNAME.get(L_TRVA[j], '')
    rows_t.append({'Technique': f'{L_TRVA[j]} {nm}'[:46], 'Test CVEs with it': int(sup[j]), 'Flagged in top 3': fl, 'Correctly flagged': tp, 'Precision': f'{tp / fl:.2f}' if fl else '-', 'Recall': f'{tp / sup[j]:.2f}'})
PT = pd.DataFrame(rows_t); print(PT.to_string(index=False))
save_table_wrapped(PT, 'table_lp_per_technique', 'Per-technique results of the selected model (top-3 rule) for the ten most frequent techniques of the test CVEs', {'Technique': 40, 'Test CVEs with it': 9, 'Flagged in top 3': 9, 'Correctly flagged': 10, 'Precision': 9, 'Recall': 9})

# ---------------------------------------------------------------------------------------------------------
section('4. Three example test CVEs (chosen by a fixed rule, not by hand)')
C_ = _load(PROC / 'cves_enriched.parquet'); DESC = dict(zip(C_.cve_id, C_.description.fillna('')))
print('rule: test CVEs in alphabetical order; the FIRST one of each kind is shown. full = every true technique is in the top 3; partial = at least one, but not every, true technique is in the top 3; miss = no true technique in the top 10.')
rk, Y_all = TEST_O[SEL]['_rank'], TEST_O[SEL]['_Y']; kinds = {'full': None, 'partial': None, 'miss': None}
for i in np.argsort(ids[TE]):
    ntrue = int(Y_all[i].sum()); inb3 = int((Y_all[i] & (rk[i] < 3)).sum()); inb10 = int((Y_all[i] & (rk[i] < 10)).sum())
    kind = 'full' if ntrue and inb3 == ntrue else 'partial' if inb3 > 0 else 'miss' if inb10 == 0 and ntrue else None
    if kind and kinds[kind] is None: kinds[kind] = i
rows_e = []
for kind, i in kinds.items():
    if i is None: continue
    cid = ids[TE[i]]; true_t = sorted(true_sets[cid]); order_i = np.argsort(rk[i])[:3]; pred = [L_TRVA[j] for j in order_i]
    rows_e.append({'Case': kind, 'CVE': cid, 'Description (start)': DESC.get(cid, '')[:150] + '...', 'True techniques (expert mapping)': '; '.join(f'{t} {TNAME.get(t, "")}' for t in true_t)[:200],
                   'Top 3 predicted': '; '.join(f'{t} {TNAME.get(t, "")}{" [true]" if t in true_t else ""}' for t in pred)})
EX = pd.DataFrame(rows_e); print(EX.to_string(index=False))
save_table_wrapped(EX, 'table_lp_examples', 'Three example test CVEs: expert techniques and the three techniques predicted by the selected model', {'Case': 7, 'CVE': 14, 'Description (start)': 34, 'True techniques (expert mapping)': 34, 'Top 3 predicted': 34})
EXTRA['example_cves'] = {k: (ids[TE[i]] if i is not None else None) for k, i in kinds.items()}

# ---------------------------------------------------------------------------------------------------------
section('5. Near-duplicates: results by the similarity of the nearest training CVE (description embeddings, test CVEs)')
TX = XG['TEXT']; nn_trva = (TX[TE] @ TX[TRVA].T).max(1); nn_tr = (TX[TE] @ TX[TR].T).max(1)
print(f'nearest training+validation CVE: median cosine {np.median(nn_trva):.2f} (training only: {np.median(nn_tr):.2f}, as in Block 14)')
bins = [('below 0.80', nn_trva < 0.80), ('0.80 to 0.90', (nn_trva >= 0.80) & (nn_trva < 0.90)), ('0.90 or higher', nn_trva >= 0.90)]
rows_s, SIM = [], {}
for name, msk in bins:
    r_ = {'Similarity of the nearest training CVE': name, 'Test CVEs': int(msk.sum())}
    for m in MODELS: r_[SHORT[m]] = f'{TEST_O[m]["ndcg@10"][msk].mean():.3f}' if msk.sum() else '-'
    rows_s.append(r_); SIM[name] = {'n': int(msk.sum()), **{m: (float(TEST_O[m]['ndcg@10'][msk].mean()) if msk.sum() else None) for m in MODELS}, 'hits@3_MLP': (float(TEST_O['MLP']['hits@3'][msk].mean()) if msk.sum() else None)}
SM = pd.DataFrame(rows_s); print('NDCG@10 by similarity group:'); print(SM.to_string(index=False)); save_table(SM, 'table_lp_similarity', 'NDCG@10 of every model on the test CVEs, grouped by the similarity of the nearest training CVE')
for name, msk in bins: print(f'  {name:15s}: {int(msk.sum()):3d} CVEs | Hits@3 of the selected model {TEST_O[SEL]["hits@3"][msk].mean():.2f}' if msk.sum() else f'  {name}: no CVE')
small = [n for n, m_ in bins if 0 < m_.sum() < 20]
if small: print(f'  careful: the group(s) {small} have fewer than 20 CVEs, so their values are very uncertain')
fig, ax = plt.subplots(figsize=(9, 4.8)); w = 0.13
for k, m in enumerate(MODELS):
    vals = [TEST_O[m]['ndcg@10'][msk].mean() if msk.sum() else 0 for _, msk in bins]; ax.bar(np.arange(3) + (k - 2.5) * w, vals, w, color=COLORS[m], label=LABEL[m])
ax.set_xticks(range(3)); ax.set_xticklabels([f'{n}\n({int(m_.sum())} CVEs)' for n, m_ in bins]); ax.set_xlabel('Cosine similarity of the nearest training CVE (description embedding)'); ax.set_ylabel('NDCG@10 on the test CVEs')
ax.set_title('Test performance by similarity to the training data'); style(ax); ax.set_ylim(0, 1.0); h_, l_ = ax.get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=8.5, ncol=3, labelcolor=INK, loc='lower center'); fig.tight_layout(rect=(0, 0.1, 1, 1))
save_fig(fig, 'fig_lp_similarity', 'NDCG@10 on the test CVEs, grouped by the similarity of the nearest training CVE')
EXTRA['similarity'] = {'median_nn_trva': float(np.median(nn_trva)), 'median_nn_tr': float(np.median(nn_tr)), 'groups': SIM}

# ---------------------------------------------------------------------------------------------------------
section('6. Feature-group ablation (VALIDATION CVEs only; kNN and MLP with their tuned settings; MLP: mean and spread over 3 seeds)')
SEEDS_ = (0, 1, 2); WIDTH = {g: XG[g].shape[1] for g in G}
def eval_val(model, keep, seed=None):
    cfg = BEST[model]; gw = {g: ((cfg.get('wt', 1.0) if g == 'TEXT' else 1.0) if g in keep else 0.0) for g in G}
    P = prep_for(TR, gw, 'ablation'); Xtr, Xva = P.transform(TR), P.transform(VA); Ytr = Ymat(TR, L_TR)
    old = globals()['SEED']
    try:
        if seed is not None: globals()['SEED'] = seed
        S, _ = FIT[model](cfg, Xtr, Ytr, Xva, epochs=BEST_EP['MLP'] if model == 'MLP' else None)
    finally: globals()['SEED'] = old
    return float(per_cve(S, VA, L_TR)['ndcg@10'].mean())
sets = [('all five groups', list(G))] + [(f'without {g}', [h for h in G if h != g]) for g in G] + [(f'only {g}', [g]) for g in G]
rows_a, AB = [], {}
for name, keep in sets:
    kv = eval_val('kNN', keep); mv = [eval_val('MLP', keep, s) for s in SEEDS_]
    AB[name] = {'columns': int(sum(WIDTH[g] for g in keep)), 'kNN': kv, 'MLP_mean': float(np.mean(mv)), 'MLP_sd': float(np.std(mv)), 'MLP_runs': mv}
    rows_a.append({'Feature set': name, 'Columns': AB[name]['columns'], 'kNN': f'{kv:.3f}', 'MLP (mean of 3 seeds)': f'{np.mean(mv):.3f}', 'MLP spread (sd)': f'{np.std(mv):.3f}'})
    print(f'  {name:18s} columns {AB[name]["columns"]:4d} | kNN {kv:.3f} | MLP {np.mean(mv):.3f} (sd {np.std(mv):.3f})')
pop_val = float(per_cve(popularity_scores(Ymat(TR, L_TR), len(VA)), VA, L_TR)['ndcg@10'].mean()); print(f'  reference: popularity baseline on the validation CVEs {pop_val:.3f}')
rows_a.append({'Feature set': 'popularity baseline (no features)', 'Columns': 0, 'kNN': f'{pop_val:.3f}', 'MLP (mean of 3 seeds)': f'{pop_val:.3f}', 'MLP spread (sd)': '-'})
AB_DF = pd.DataFrame(rows_a); save_table(AB_DF, 'table_lp_feature_ablation', 'Validation NDCG@10 with feature groups removed or used alone (kNN and MLP with their tuned settings)')
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
for ax, (title, pick) in zip(axs, (('One group removed', [s for s in sets if s[0].startswith('without') or s[0].startswith('all')]), ('One group used alone', [s for s in sets if s[0].startswith('only')]))):
    names = [s[0] for s in pick]; yy = np.arange(len(names))
    ax.barh(yy - 0.18, [AB[n]['kNN'] for n in names], 0.34, color=COLORS['kNN'], label='kNN'); ax.barh(yy + 0.18, [AB[n]['MLP_mean'] for n in names], 0.34, xerr=[AB[n]['MLP_sd'] for n in names], color=COLORS['MLP'], label='MLP (3 seeds)', error_kw={'lw': 1})
    ax.axvline(pop_val, color=INK2, ls=':', lw=1.3, label='popularity baseline'); ax.set_yticks(yy); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlabel('Validation NDCG@10'); ax.set_title(title); ax.set_xlim(0, 1); style(ax, 'x')
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=9.5, labelcolor=INK, loc='lower center', ncol=3); fig.tight_layout(rect=(0, 0.07, 1, 1))
save_fig(fig, 'fig_lp_feature_ablation', 'Validation NDCG@10 of kNN and MLP when one feature group is removed (left) or one group is used alone (right); the dotted line is the popularity baseline')
EXTRA['feature_ablation'] = {'sets': AB, 'popularity_val': pop_val}

# ---------------------------------------------------------------------------------------------------------
section('7. Learning curve: validation NDCG@10 against the number of training CVEs (3 random subsets per size; MLP gets the same number of gradient steps as the final model)')
FR = (0.10, 0.25, 0.50, 0.75, 1.00); rec = []
for fr in FR:
    for s in SEEDS_:
        sub = TR if fr >= 1 else np.sort(np.random.default_rng(100 + s).choice(TR, size=max(30, int(round(fr * len(TR)))), replace=False)); L = label_space(sub); Ytr = Ymat(sub, L)
        pr_k = prep_for(sub, {'TEXT': BEST['kNN'].get('wt', 1.0)}, f'lc_k_{fr}_{s}'); pr_m = prep_for(sub, {'TEXT': 1.0}, f'lc_m_{fr}_{s}')
        Sk, _ = FIT['kNN'](BEST['kNN'], pr_k.transform(sub), Ytr, pr_k.transform(VA)); ep = int(min(400, round(BEST_EP['MLP'] * len(TR) / len(sub)))); old = globals()['SEED']
        try:
            globals()['SEED'] = s; Sm, _ = FIT['MLP'](BEST['MLP'], pr_m.transform(sub), Ytr, pr_m.transform(VA), epochs=ep)
        finally: globals()['SEED'] = old
        rec.append({'fraction': fr, 'n': len(sub), 'techniques': len(L), 'popularity': float(per_cve(popularity_scores(Ytr, len(VA)), VA, L)['ndcg@10'].mean()),
                    'kNN': float(per_cve(Sk, VA, L)['ndcg@10'].mean()), 'MLP': float(per_cve(Sm, VA, L)['ndcg@10'].mean())})
LC = pd.DataFrame(rec); LCm = LC.groupby('fraction').agg(n=('n', 'first'), techniques=('techniques', 'mean'), **{f'{m}_mean': (m, 'mean') for m in ('popularity', 'kNN', 'MLP')}, **{f'{m}_sd': (m, 'std') for m in ('popularity', 'kNN', 'MLP')}).reset_index()
tab_lc = pd.DataFrame({'Share of the training CVEs': [f'{int(100 * f)}%' for f in LCm.fraction], 'Training CVEs': LCm.n.astype(int), 'Techniques seen': LCm.techniques.round(0).astype(int),
                       'Popularity': [f'{v:.3f}' for v in LCm.popularity_mean], 'kNN': [f'{a:.3f} ± {0 if np.isnan(b) else b:.3f}' for a, b in zip(LCm.kNN_mean, LCm.kNN_sd)], 'MLP': [f'{a:.3f} ± {0 if np.isnan(b) else b:.3f}' for a, b in zip(LCm.MLP_mean, LCm.MLP_sd)]})
print(tab_lc.to_string(index=False)); save_table(tab_lc, 'table_lp_learning_curve', 'Validation NDCG@10 of the popularity baseline, kNN and MLP with 10 to 100 percent of the training CVEs (mean and standard deviation of 3 runs)')
fig, ax = plt.subplots(figsize=(7.2, 4.8))
for m in ('popularity', 'kNN', 'MLP'):
    mu, sd = LCm[f'{m}_mean'].values, LCm[f'{m}_sd'].fillna(0).values; ax.plot(LCm.n, mu, color=COLORS[m], marker=MARK[m], ms=7 if m == 'MLP' else 5, lw=2, ls='--' if m == 'popularity' else '-', label=LABEL[m] if m != 'kNN' else 'k-nearest neighbours')
    ax.fill_between(LCm.n, mu - sd, mu + sd, color=COLORS[m], alpha=0.15, linewidth=0)
ax.set_xlabel('Number of training CVEs'); ax.set_ylabel('Validation NDCG@10'); ax.set_title('Learning curve (validation CVEs)'); style(ax, 'both'); ax.legend(frameon=False, labelcolor=INK, loc='lower right'); fig.tight_layout()
save_fig(fig, 'fig_lp_learning_curve', 'Validation NDCG@10 against the number of training CVEs (mean and one standard deviation of three runs)')
EXTRA['learning_curve'] = LC.to_dict('records')

# ---------------------------------------------------------------------------------------------------------
section('8. Redraw two items from Block 15 (a clipped axis label, and the hidden-layer format in the tuning table)')
ks = np.arange(1, 21); fig, ax = plt.subplots(figsize=(8, 5))
for m in MODELS:
    ax.plot(ks, [float(np.mean(TEST_O[m]['_first'] <= k)) for k in ks], color=COLORS[m], marker=MARK[m], ms=5 if m != 'MLP' else 8, lw=2 if m != 'popularity' else 1.5, ls='--' if m == 'popularity' else '-', label=LABEL[m])
ax.set_xlabel('k (number of top-ranked techniques)'); ax.set_ylabel('Hits@k'); ax.set_ylim(0, 1.02); ax.set_xticks([1, 3, 5, 10, 15, 20]); ax.set_title('Share of test CVEs with a true technique in the top k')
ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%']); style(ax); ax.legend(frameon=False, loc='lower right', labelcolor=INK, fontsize=9.5); fig.tight_layout()
save_fig(fig, 'fig_lp_hits_at_k', 'Hits@k of the five models and the popularity baseline on the test CVEs')
TT2 = TT.copy(); TT2['Best setting'] = TT2['Best setting'].str.replace('hidden (512, 256)', 'hidden 512 + 256', regex=False).str.replace('hidden (256,)', 'hidden 256', regex=False)
save_table_wrapped(TT2, 'table_lp_tuning', 'Hyperparameter search space, best setting and validation score of every model', {'Model': 6, 'Search method': 14, 'Search space': 34, 'Val. NDCG@10 (default)': 11, 'Best setting': 32, 'Val. NDCG@10 (tuned)': 11})

_save(EXTRA, DIRS['results'] / 'lp_extra_results.json')
print(f'\nsaved: results/lp_extra_results.json | figures: fig_lp_roc, fig_lp_pr, fig_lp_confusion, fig_lp_similarity, fig_lp_feature_ablation, fig_lp_learning_curve, fig_lp_hits_at_k (redrawn)')
print('tables : table_lp_confusion, table_lp_per_technique, table_lp_examples, table_lp_similarity, table_lp_feature_ablation, table_lp_learning_curve, table_lp_tuning (redrawn)')
print(f'total time {time.time() - T0_:.0f} s'); print('\nBLOCK 15b DONE')
