# AI Lab

Coursework repository for the **AI Lab** class — one directory per lab step, each
containing its own code, report and notes.

| | |
|---|---|
| **Student** | Amirhossein Hassani |
| **Email** | [amirhosein83hasani@gmail.com](mailto:amirhosein83hasani@gmail.com) |
| **Class** | AI Lab |
| **Instructor** | Jokar |

## Repository layout

```
ai-lab/
├── README.md                     # this file — overview, setup, index of steps
├── requirements.txt              # Python dependencies for all steps
├── step-1/
│   ├── README.md                 # report: problem definition (PEAS) + environment properties
│   ├── HOW.md                    # how the work was produced (tools / AI assistance)
│   ├── agent.py                  # the agent: three classifiers compared on handwritten digits
│   ├── check_env.ipynb           # environment check notebook (Python / NumPy / Matplotlib versions)
│   └── notes/
│       └── evaluation-and-risks.md   # metrics, risks & mitigations, reporting protocol
└── step-2/                       # (next step — to be added)
```

Conventions:

- directories are lowercase and hyphenated (`step-1`, `step-2`, …) — no spaces in paths,
  so commands work without quoting in every shell;
- each step keeps its own `README.md` (the report) and `HOW.md` (how it was built);
- supporting notes live in that step's `notes/` directory;
- file names describe their contents (`evaluation-and-risks.md`, not `risks.md`).

## Steps

| Step | Topic | Code | Report |
|---|---|---|---|
| 1 | Handwritten digit recognition (0–9): PEAS, environment, baseline vs. classifiers | [`step-1/agent.py`](step-1/agent.py) | [`step-1/README.md`](step-1/README.md) |

Step 1 compares three "agents" on the built-in 8×8 handwritten-digits dataset
(`sklearn.datasets.load_digits`, 1797 samples):

| Model | What it does | Mean test accuracy (100 runs) |
|---|---|---|
| `random` | uniform random guess (baseline) | ~9.9 % |
| `knn` | k-nearest neighbours, k = 3 | ~98.6 % |
| `logreg` | standardization + logistic regression | ~96.8 % |

Numbers are produced by `agent.py` itself (30 % held-out test split, stratified, seeds 0–99),
not hard-coded.

## Quick start

Requires Python 3.10+ (the environment-check notebook was run on Python 3.13 / Colab).

```bash
git clone https://github.com/deathtoconding/ai-lab.git
cd ai-lab

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python step-1/agent.py
```

`agent.py` prints the per-model accuracies and summary table, then draws a bar chart
with error bars and two animations. It detects Jupyter/Colab and renders the animations
as inline HTML there; in a plain script it opens Matplotlib windows.

## Documentation

- Reports and notes are written in **Persian**, with a short English summary at the top of
  each document; this README and the code comments are in **English**.
- Evaluation metrics, risks and the reporting protocol: [`step-1/notes/evaluation-and-risks.md`](step-1/notes/evaluation-and-risks.md).
- How each step was produced (including AI assistance): the `HOW.md` file inside that step.
