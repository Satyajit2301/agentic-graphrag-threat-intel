# Agentic GraphRAG for Threat Analysis and Patch Prioritization

**Agentic Graph Retrieval-Augmented Generation over a Cybersecurity Knowledge Graph with CVE-to-ATT&CK Link Prediction for Multi-Hop Threat Analysis and Patch Prioritization**

Course project, *Machine Learning Projects with Python (CSE 4192)*, B.Tech. CSE (AI & ML), Centre for Artificial Intelligence & Machine Learning, Faculty of Engineering & Technology (ITER), Siksha 'O' Anusandhan (Deemed to be University), Bhubaneswar.

| Team member | Regd. No. |
|---|---|
| Satyajit Sethy | 2341019390 |
| E Sailaja | 2341019290 |
| Trushna Tripathy | 2341020046 |

Project guide: Dr. Gyana Ranjan Patra, Assistant Professor.

Live application: <https://agentic-graphrag-threat-intel.streamlit.app> · Report: [`report/main.tex`](report/main.tex)

---

## What this project does

Security teams have to decide which of thousands of published vulnerabilities (CVEs) matter for their own software and how an attacker could use them. The answer needs facts from several public sources and several relations: *product → CVE → ATT&CK technique → tactic, mitigation or attacker group*, plus exploitation signals (KEV, EPSS).

We built

1. **A knowledge graph** of 18,661 nodes (8 node types) and 68,250 edges (9 relations) from NVD, MITRE ATT&CK v20.1, CISA KEV, EPSS, CWE and expert CVE-to-ATT&CK mappings (10,086 CVEs).
2. **A benchmark of 108 multi-hop questions** (9 types, 36 development and 72 test questions). The gold answers are computed from the graph and confirmed by a second, independent traversal.
3. **Three question-answering systems that use the same open language model** (`gpt-oss-20b`, temperature 0) and differ only in the evidence they are given:
   - *plain RAG*: BM25 + `bge-small-en-v1.5` embeddings fused by reciprocal rank fusion, top 20 chunks;
   - *GraphRAG*: the graph neighbourhood of the entities of the question as facts (about 3,600 tokens), one model call;
   - *Agentic GraphRAG*: the model calls six graph tools in a loop (at most 8 calls) and refers to intermediate results as numbered sets.
4. **CVE-to-ATT&CK link prediction** for the 88 % of CVEs that have no expert mapping: logistic regression, kNN, SVM, random forest and an MLP, tuned on validation data and evaluated once on a cold-start test split, against a popularity baseline.
5. **This Streamlit application**: ask a question, see the answer, the tool trace and the evidence graph, and look up the predicted techniques of a CVE.

## Results (all numbers come from our own runs; details in the report)

Question answering on the 72 test questions (average score; answers run once with frozen settings):

| System | Average score (95 % bootstrap interval) | Answers exactly right |
|---|---|---|
| Plain RAG | 17.3 % (10.2 – 25.2) | 7 / 72 |
| GraphRAG | 83.1 % (74.0 – 91.0) | 56 / 72 |
| Agentic GraphRAG | 92.1 % (85.6 – 97.2) | 65 / 72 |

Both graph-based systems are clearly better than plain RAG. The overall difference between GraphRAG and the agent (9.0 points, interval −1.4 to +19.2) is not statistically clear; the agent is clearly better on three-hop questions (92 % against 54 %).

Link prediction on the 181 test CVEs (NDCG@10): popularity baseline 0.446; the five models 0.591 – 0.604. The models cannot be told apart from each other, and near-duplicate CVEs probably make the scores optimistic for CVEs of a new family (see the *Limitations* chapter of the report).

## Repository structure

```
agentic-graphrag-threat-intel/
├── app.py                      Streamlit application (three tabs and a sidebar)
├── core/                       code of the application
│   ├── graph.py                loads the graph tables, entity linking, GraphRAG context
│   ├── tools.py                the six graph tools and the numbered result sets (S1, S2, ...)
│   ├── agent.py                the tool-calling loop (numbered sets S1, S2, ...)
│   ├── llm.py                  client for an OpenAI-style API (Groq or a local Ollama server)
│   ├── evidence.py             evidence graph (DOT) built from the hops of an answer
│   ├── predict.py              ATT&CK technique lookup for a CVE
│   ├── replay.py               replay of recorded evaluation runs (no API key needed)
│   └── score.py                scorer of the benchmark (F1, top-3 overlap, exact count)
├── data/                       created by Block 16 of the notebook (see below)
│   ├── kg_nodes.parquet        knowledge graph: nodes
│   ├── kg_edges.parquet        knowledge graph: edges
│   ├── lp_predictions.parquet  top-20 techniques of the selected MLP for every CVE
│   ├── lp_cve_info.parquet     split (train / val / test) of every CVE
│   ├── lp_test_results.json    test metrics of the link-prediction models
│   ├── benchmark.json          the 108 benchmark questions with gold answers
│   ├── example_questions.json  two test questions per question type (for the drop-down list)
│   └── recorded_runs.json      recorded tool traces of the evaluation run
├── tests/                      automated tests (pytest)
├── notebooks/                  the Google Colab notebook (Blocks 1 to 16) that produced everything
├── report/                     main.tex, figures/, tables/, logo.png
├── requirements.txt            packages of the application
├── requirements-dev.txt        packages for the tests
├── .streamlit/
│   ├── config.toml             theme
│   └── secrets.toml.example    template of the secrets file (the real secrets.toml is NOT committed)
└── .github/workflows/tests.yml runs pytest on every push
```

## Run it on your own computer

```bash
git clone https://github.com/Satyajit2301/agentic-graphrag-threat-intel.git
cd agentic-graphrag-threat-intel

python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

The application opens at <http://localhost:8501>. **Without any key** the tab *ATT&CK link predictor* and the *recorded run* mode of the tab *Ask the analyst* work completely. For live questions you need a free API key from <https://console.groq.com>:

1. copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`;
2. replace the text after `GROQ_API_KEY =` with your key.

To use a local Ollama server instead, set `LLM_BASE_URL = "http://localhost:11434/v1"` and `GROQ_MODEL = "gpt-oss:20b"` in the same file.

## Where the `data/` folder comes from

The notebook writes all results to Google Drive (`MyDrive/GraphRAG_ThreatIntel/`). **Block 16** of the notebook collects what the application needs and makes `MyDrive/GraphRAG_ThreatIntel/app_package/app_data.zip`. Download this zip from Drive and unzip it in the repository folder, so that the folder `data/` is next to `app.py`. No trained model has to be stored: the link-prediction model is kept as its precomputed predictions (`lp_predictions.parquet`) for all 10,086 CVEs. Every file is far below the 100 MB limit of GitHub (Block 16 checks this and prints the sizes).

## Deploy on Streamlit Community Cloud

1. Create the GitHub repository (see below) and push everything, including `data/`.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and choose **Create app → Deploy a public app from GitHub**.
3. Repository: `Satyajit2301/agentic-graphrag-threat-intel`, branch `main`, main file path `app.py`.
4. Under **Advanced settings → Secrets** paste

   ```toml
   GROQ_API_KEY = "gsk_your_own_key"
   ```

   (and, optionally, `GROQ_MODEL`). Choose Python 3.12 in the same dialog.
5. Press **Deploy**. The first build takes a few minutes. The address is shown at the top.

Never put the key in a file of the repository. The free Groq plan has rate limits and a daily token limit; the application waits for short limits, reports an exhausted limit in plain words, and limits each session to 25 questions.

## Create the GitHub repository (first time)

On <https://github.com/new>: repository name `agentic-graphrag-threat-intel`, *Public*, no README (this one exists). Then, in the project folder:

```bash
git init
git add .
git commit -m "Initial commit: Agentic GraphRAG threat-intelligence analyst"
git branch -M main
git remote add origin https://github.com/Satyajit2301/agentic-graphrag-threat-intel.git
git push -u origin main
```

If you prefer a graphical tool, *GitHub Desktop* does the same with *Add local repository → Publish repository*. Dragging files into the web page of GitHub also works, but the browser accepts only files up to 25 MB and 100 files per upload, so use git or GitHub Desktop for `data/`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

- `tests/test_smoke.py` checks the parts one by one (graph loading, entity linking, the tools, the agent with a scripted language model, the step limit, replay, scorer, predictor) and the whole interface with the Streamlit test runner (start, predictor, recorded answer, empty question, missing key, exhausted quota, unknown CVE). It uses a small synthetic graph with the same schema as the real one, so it needs neither `data/` nor a key.
- `tests/test_real_data.py` runs the ideal tool sequence of every benchmark question on the real graph and checks that all 108 gold answers are reproduced and that every GraphRAG context fits its budget. It is skipped automatically when `data/` is missing.

## Limitations (see the report for the full list)

- The 10,086 CVEs are a selection (all KEV CVEs, all CVEs with an expert mapping and the CVEs of 24 products), not a random sample of the NVD.
- The product stacks in the benchmark questions are synthetic, and the gold answers follow the expert CVE-to-ATT&CK mapping, which is incomplete; precision of the technique suggestions is therefore a lower bound.
- The live application calls the hosted `openai/gpt-oss-20b` (Groq). The evaluation used the local `gpt-oss:20b` build (Ollama); the hosted model was not run on the test questions, so its behaviour can differ slightly.
- The technique predictor is a lookup of precomputed scores; it does not compute features for new CVEs.

## Data sources

[NVD](https://nvd.nist.gov) (NIST), [MITRE ATT&CK](https://attack.mitre.org) and [CWE](https://cwe.mitre.org), [CISA Known Exploited Vulnerabilities](https://www.cisa.gov/known-exploited-vulnerabilities-catalog), [EPSS](https://www.first.org/epss/) (FIRST) and the CVE-to-ATT&CK mappings of the [Center for Threat-Informed Defense](https://ctid.mitre.org/projects/mapping-attck-to-cve-for-impact/). The data are a snapshot of November 2026; please follow the terms of use of each source if you reuse them.

## Contributions

| Member | Main responsibility |
|---|---|
| Satyajit Sethy | data collection and cleaning, knowledge graph, benchmark and its independent gold-answer check |
| E Sailaja | plain RAG, GraphRAG and Agentic GraphRAG, tool design, final evaluation and ablation, literature survey |
| Trushna Tripathy | link-prediction features, models and evaluation, Streamlit application, tests and deployment |

All three members took part in planning, running the notebook, testing and checking the numbers of the report.

## Use of an AI assistant

An AI assistant (Claude, by Anthropic) was used to help write code and to draft text. We ran the code ourselves and compared every number in the report with the output of our own runs.
