import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Detect whether the code is running in a Jupyter/Colab notebook or a .py script
try:
    from IPython import get_ipython
    from IPython.display import HTML, display
    shell = get_ipython()
    IN_NOTEBOOK = shell is not None and shell.__class__.__name__ in (
        "ZMQInteractiveShell",
        "Shell",
    )
except (ImportError, NameError):
    IN_NOTEBOOK = False

# ---------------------------------------------------------------- settings
STEPS = 100                             # digits shown in one animated run
RUNS = 100                              # number of experiments per model
TEST_SIZE = 0.3                        # share of data used for testing
MODELS = ["random", "knn", "logreg"]   # "agents" of this task

digits = load_digits()
X, y = digits.data, digits.target      # X: (1797, 64) pixel values in 0..16


def build_model(name, seed):
    """Return an untrained classifier corresponding to the chosen model name."""
    if name == "random":
        return DummyClassifier(strategy="uniform", random_state=seed)
    elif name == "knn":
        return KNeighborsClassifier(n_neighbors=3)
    elif name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
    raise ValueError(f"Unknown model: {name}")


def simulate(name, seed):
    """One experiment: random split -> train -> test accuracy (0..100%)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    clf = build_model(name, seed).fit(X_tr, y_tr)
    return float(100 * (clf.predict(X_te) == y_te).mean())


def record(name, seed=1):
    """Frame-by-frame log: (8x8 image, true label, prediction, running score)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    clf = build_model(name, seed).fit(X_tr, y_tr)

    n_steps = min(STEPS, len(X_te))
    preds = clf.predict(X_te[:n_steps])

    score = 0
    frames = [(np.zeros((8, 8)), None, "Start", score)]
    for t in range(n_steps):
        score += int(preds[t] == y_te[t])          # +1 per correct prediction
        frames.append(
            (X_te[t].reshape(8, 8), int(y_te[t]), int(preds[t]), score)
        )
    return frames


def show_animation(fig, anim):
    """Render animation as HTML in notebooks or as a window in standard Python."""
    if IN_NOTEBOOK:
        plt.close(fig)
        display(HTML(anim.to_jshtml()))
    else:
        plt.show()


# --- 1) Single test run per model ---
for name in MODELS:
    print(f"{name:7s} accuracy: {simulate(name, seed=1):.1f} %")

# --- 2) Run multiple experiments per model ---
rows = []
for name in MODELS:
    for seed in range(RUNS):
        rows.append({"model": name, "accuracy": simulate(name, seed)})

df = pd.DataFrame(rows)
print(df.head())

summary = (
    df.groupby("model")["accuracy"]
    .agg(["mean", "std"])
    .reindex(MODELS)          # keep order: random -> knn -> logreg
    .round(1)
)
print(summary)

# --- 3) Static bar chart ---
summary["mean"].plot(
    kind="bar",
    yerr=summary["std"],
    capsize=5,
    color=["#ec4899", "#4f46e5", "#f59e0b"],
)
plt.title(f"Average test accuracy over {RUNS} runs (8x8 handwritten digits)")
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.ylim(0, 105)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 4) Animation 1: models classifying digits one by one ---
runs = [record(name, seed=1) for name in MODELS]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), dpi=80)
fig.subplots_adjust(wspace=0.45, top=0.72)


def draw(t):
    for ax, name, frames in zip(axes, MODELS, runs):
        img, true, pred, score = frames[t]
        ax.clear()
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=16)
        ax.set_xticks([])
        ax.set_yticks([])

        if pred == "Start":
            color, verdict, true_txt, pred_txt = "black", "ready", "—", "—"
            score_txt = "0"
        else:
            correct = pred == true
            color = "#16a34a" if correct else "#dc2626"   # green / red
            verdict = "correct" if correct else "wrong"
            true_txt, pred_txt = str(true), str(pred)
            score_txt = f"{score}/{t}"

        for side in ax.spines.values():
            side.set_edgecolor(color)
            side.set_linewidth(3)

        ax.set_title(
            f"{name} | step {t}\n"
            f"true: {true_txt}  pred: {pred_txt} ({verdict})\n"
            f"score: {score_txt}",
            fontsize=9,
        )


anim = FuncAnimation(fig, draw, frames=len(runs[0]), interval=250, repeat=True)
show_animation(fig, anim)

# --- 5) Animation 2: growing bar chart of average accuracy ---
fig2, ax2 = plt.subplots(figsize=(6, 3.5), dpi=80)
final = summary["mean"]


def grow(k):
    ax2.clear()
    ax2.set_ylim(0, 110)
    ax2.set_title(f"Average test accuracy over {RUNS} runs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Model")
    values = final.values * (k + 1) / 20
    bars = ax2.bar(final.index, values, color=["#ec4899", "#4f46e5", "#f59e0b"])
    for bar, value in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )
    ax2.set_xticks(range(len(final.index)))
    ax2.set_xticklabels(final.index, rotation=0)


grow(0)
fig2.tight_layout()
anim2 = FuncAnimation(fig2, grow, frames=20, interval=120, repeat=False)
show_animation(fig2, anim2)
