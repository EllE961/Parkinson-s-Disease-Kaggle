"""Evaluate the Parkinson's classifier on recordings and on unseen speakers, and write the README figures.

Run from the repo root:  python scripts/make_figures.py   (about a minute)
Prints every number quoted in the README and writes docs/assets/*.png.

The dataset has several voice recordings per person. A random split by recording puts the same person in
train and test, so this script also evaluates with folds grouped by subject (the S## part of `name`).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
CLASS_COLORS = {0: GREEN, 1: EMBER}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote docs/assets/{name}")


def subject_folds(subjects, status, seed, k=5):
    """k folds with whole subjects held out, healthy and Parkinson's subjects dealt out separately."""
    rng = np.random.default_rng(seed)
    per_subject = status.groupby(subjects).first()
    fold_of = {}
    for cls in (0, 1):
        ids = rng.permutation(per_subject[per_subject == cls].index.values)
        offset = rng.integers(k)
        for i, sid in enumerate(ids):
            fold_of[sid] = (i + offset) % k
    f = subjects.map(fold_of).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def protocol_figure(names, row, grp):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    y = np.arange(len(names))[::-1]
    h = 0.38
    for ax, metric, title in zip(axes, ["test_accuracy", "test_balanced_accuracy"],
                                 ["Accuracy", "Balanced accuracy (mean of the two recalls)"]):
        for res, off, color, label in [(row, h / 2, TEMPER, "random split by recording"),
                                       (grp, -h / 2, EMBER, "new speakers (folds grouped by subject)")]:
            m = np.array([r[metric].mean() for r in res])
            s = np.array([r[metric].std() for r in res])
            ax.barh(y + off, m, h, xerr=s, color=color, label=label,
                    error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
            for yi, mi, si in zip(y + off, m, s):
                ax.text(mi + si + 0.012, yi, f"{mi:.3f}", va="center", color=INK, fontsize=8.5)
        ax.set_xlim(0, 1.08)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_title(title, pad=34)
    axes[0].set_yticks(y, names)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2)
    fig.text(0.012, -0.015, "Mean ± std over 50 folds: 5-fold CV repeated 10 times for each protocol, 195 recordings "
             "from 32 subjects.", color=INK3, fontsize=8.5, ha="left", va="top")
    fig.tight_layout(w_pad=3)
    save(fig, "protocols.png")


def eda_figure(df, subjects):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [0.75, 1.5]})
    ax = axes[0]
    rec = df["status"].value_counts().sort_index()
    sub = df.groupby(subjects)["status"].first().value_counts().sort_index()
    x = np.arange(2)
    w = 0.38
    for off, vals, color, label in [(-w / 2, rec, INK3, "recordings"), (w / 2, sub, PURPLE, "subjects")]:
        bars = ax.bar(x + off, vals.values, w, color=color, label=label)
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", va="bottom", color=INK2, fontsize=8.5)
    ax.set_xticks(x, ["healthy", "Parkinson's"])
    ax.set_ylim(0, rec.max() * 1.15)
    ax.legend(loc="upper left")
    ax.set_title("195 recordings, 32 people")

    ax = axes[1]
    fx, fy = "MDVP:Fo(Hz)", "HNR"
    for sid, g in df.groupby(subjects):
        cx, cy = g[fx].mean(), g[fy].mean()
        color = CLASS_COLORS[g["status"].iloc[0]]
        for _, r in g.iterrows():
            ax.plot([cx, r[fx]], [cy, r[fy]], color=color, alpha=0.35, lw=0.8)
    for cls, label in [(0, "healthy"), (1, "Parkinson's")]:
        m = df["status"] == cls
        ax.scatter(df.loc[m, fx], df.loc[m, fy], s=18, color=CLASS_COLORS[cls], edgecolor=BG, linewidth=0.4,
                   label=label, zorder=3)
    ax.set_xlabel("MDVP:Fo, average vocal fundamental frequency (Hz)")
    ax.set_ylabel("HNR, harmonics-to-noise ratio (dB)")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    ax.set_title("Recordings cluster by person")
    fig.text(0.39, -0.015, "Lines join each recording to the mean of that person's recordings.", color=INK3,
             fontsize=8.5, ha="left", va="top")
    fig.tight_layout(w_pad=3)
    save(fig, "eda.png")


def draw_cm(ax, cm, title):
    ax.imshow(cm, cmap=HEAT, vmin=0, vmax=cm.max())
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n{v / total:.1%}", ha="center", va="center", fontsize=11,
                    color=BG if v > cm.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["healthy", "Parkinson's"])
    ax.set_yticks([0, 1], ["healthy", "Parkinson's"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)


def confusion_figure(cm_split, cm_group):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    draw_cm(axes[0], cm_split, "Script's 80/20 split (39 recordings)")
    draw_cm(axes[1], cm_group, "New speakers, out of fold (195)")
    fig.tight_layout(w_pad=4)
    save(fig, "confusion.png")


def main():
    df = pd.read_csv(ROOT / "data" / "parkinsons.csv")
    X = df.drop(["name", "status"], axis=1)
    y = df["status"]
    subjects = df["name"].str.extract(r"_(S\d+)_")[0]
    per_subject = df.groupby(subjects)["status"].first()
    print(f"recordings: {len(df)} ({(y == 1).sum()} Parkinson's, {(y == 0).sum()} healthy); subjects: "
          f"{subjects.nunique()} ({(per_subject == 1).sum()} Parkinson's, {(per_subject == 0).sum()} healthy); "
          f"recordings per subject: {df.groupby(subjects).size().value_counts().to_dict()}")

    share = {c: X[c].groupby(subjects).transform("mean").var() / X[c].var() for c in X}
    print(f"share of feature variance that lies between people: median {np.median(list(share.values())):.3f}, "
          f"MDVP:Fo {share['MDVP:Fo(Hz)']:.3f}, HNR {share['HNR']:.3f}, lowest {min(share, key=share.get)} "
          f"{min(share.values()):.3f}")

    # the script's own split and model
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, subjects, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.transform(X_train), y_train)
    pred = rf.predict(scaler.transform(X_test))
    cm_split = confusion_matrix(y_test, pred)
    print(f"script split: accuracy {accuracy_score(y_test, pred):.4f} ({(pred == y_test).sum()}/{len(y_test)}), "
          f"confusion {cm_split.tolist()}")
    print(f"test recordings whose speaker also has recordings in train: {s_test.isin(set(s_train)).sum()} of "
          f"{len(s_test)} (test covers {s_test.nunique()} subjects)")

    models = {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
        "k-NN (k = 5)": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Random forest (this repo)": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100,
                                                                                            random_state=42)),
        "Gradient boosting": GradientBoostingClassifier(random_state=0),
    }
    row_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    grp_cv = [fold for seed in range(10) for fold in subject_folds(subjects, y, seed)]
    scoring = ["accuracy", "balanced_accuracy"]
    row, grp = [], []
    for name, m in models.items():
        a = cross_validate(m, X, y, cv=row_cv, scoring=scoring, n_jobs=4)
        b = cross_validate(m, X, y, cv=grp_cv, scoring=scoring, n_jobs=4)
        row.append(a)
        grp.append(b)
        print(f"{name:26s} by recording: acc {a['test_accuracy'].mean():.3f} +/- {a['test_accuracy'].std():.3f}, "
              f"bal {a['test_balanced_accuracy'].mean():.3f} | by subject: acc {b['test_accuracy'].mean():.3f} "
              f"+/- {b['test_accuracy'].std():.3f}, bal {b['test_balanced_accuracy'].mean():.3f}")

    # one grouped pass for the out-of-fold confusion matrix and a per-person majority vote
    oof = cross_val_predict(models["Random forest (this repo)"], X, y, cv=subject_folds(subjects, y, 0))
    cm_group = confusion_matrix(y, oof)
    vote = pd.Series(oof).groupby(subjects.values).mean().round().astype(int)
    truth = per_subject.loc[vote.index]
    print(f"grouped out-of-fold: accuracy {accuracy_score(y, oof):.4f}, confusion {cm_group.tolist()}, healthy recall "
          f"{cm_group[0, 0] / cm_group[0].sum():.3f}; per-person majority vote correct for {(vote == truth).sum()} of "
          f"{len(vote)} ({((vote == truth) & (truth == 0)).sum()} of {(truth == 0).sum()} healthy)")

    protocol_figure(list(models), row, grp)
    eda_figure(df, subjects)
    confusion_figure(cm_split, cm_group)


if __name__ == "__main__":
    main()
