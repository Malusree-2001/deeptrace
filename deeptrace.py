#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# DeepTrace - Visual Signature Graphs for Detecting AI-Generated Images
# Author: <your name>

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False
    print("[WARN] scikit-image not available; skipping GLCM features.", file=sys.stderr)


# --------------------- Feature Extraction ---------------------

def read_image(path, size=256):
    """Return grayscale float image resized to square 'size'."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read: {path}")
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def edge_density(gray):
    g8 = (gray * 255).astype(np.uint8)
    edges = cv2.Canny(g8, 100, 200)
    return float((edges > 0).mean())


def laplacian_var(gray):
    g8 = (gray * 255).astype(np.uint8)
    lap = cv2.Laplacian(g8, cv2.CV_64F)
    return float(lap.var())


def noise_residual_stats(gray, ksize=3):
    g8 = (gray * 255).astype(np.uint8)
    med = cv2.medianBlur(g8, ksize)
    resid = (g8.astype(np.float32) - med.astype(np.float32)) / 255.0
    std = float(resid.std() + 1e-12)
    m = resid.mean()
    s2 = resid.var() + 1e-12
    s4 = np.mean((resid - m) ** 4)
    kurtosis_excess = float(s4 / (s2 ** 2) - 3.0)
    return std, kurtosis_excess


def fft_highfreq_ratio(gray, low_frac=0.2):
    f = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    ly, lx = int(h * low_frac / 2), int(w * low_frac / 2)
    cy, cx = h // 2, w // 2
    low = mag[cy - ly:cy + ly, cx - lx:cx + lx].sum()
    tot = mag.sum() + 1e-12
    return float((tot - low) / tot)


def blockiness_score(gray, block=8):
    g = (gray * 255).astype(np.float32)
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    h, w = g.shape
    x = np.arange(w - 1)
    y = np.arange(h - 1)
    mask_v = ((x + 1) % block == 0).astype(np.float32)
    mask_h = ((y + 1) % block == 0).astype(np.float32)
    on_v = (dh * mask_v[None, :]).sum()
    on_h = (dv * mask_h[:, None]).sum()
    total = dh.sum() + dv.sum() + 1e-12
    return float((on_v + on_h) / total)


def glcm_features(gray):
    if not SKIMAGE_OK:
        return np.nan, np.nan
    g8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(g8, distances=(1,), angles=(0,), levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    return float(contrast), float(homogeneity)


def extract_vector(gray):
    std, kurt = noise_residual_stats(gray)
    con, hom = glcm_features(gray)
    return {
        "edge_density": edge_density(gray),
        "laplacian_var": laplacian_var(gray),
        "resid_std": std,
        "resid_kurtosis": kurt,
        "fft_highfreq_ratio": fft_highfreq_ratio(gray),
        "blockiness": blockiness_score(gray),
        "glcm_contrast": con,
        "glcm_homogeneity": hom,
    }


# --------------------- Graph & Analytics ---------------------

def matrix_X(df):
    cols = [c for c in df.columns if c not in ("path", "name", "label")]
    return df[cols].astype(float).values, cols


def knn_graph(X, k=5):
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0.0)
    G = nx.Graph()
    n = S.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        nbrs = np.argsort(S[i])[::-1][:k]
        for j in nbrs:
            if i == j:
                continue
            w = float(S[i, j])
            if w > 0 and not G.has_edge(i, j):
                G.add_edge(i, j, weight=w)
    return G, S


def cluster_and_anomaly(X, dbscan_eps=0.8, dbscan_min_samples=5, random_state=42):
    Xs = StandardScaler().fit_transform(X)

    # Clustering
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(Xs)
    cl = db.labels_

    # Anomaly scoring (higher raw score = less anomalous)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state).fit(Xs)
    raw = iso.score_samples(Xs)
    ranks = raw.argsort().argsort().astype(float)          # rank to [0..N-1]
    anom_score = ranks / (len(ranks) - 1 + 1e-9)           # normalize to [0..1]; higher = more anomalous
    return cl, anom_score


# --------------------- Visualization ---------------------

def plot_feature_hists(df, out_dir):
    cols = [c for c in df.columns if c.startswith(("edge_", "laplacian", "resid_", "fft_", "blockiness", "glcm_"))]
    if not cols:
        return
    rows = (len(cols) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna().values, bins=20)
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "feature_histograms.png", dpi=180)
    plt.close(fig)


def plot_scatter(df, out_dir, x="fft_highfreq_ratio", y="blockiness"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df[x], df[y], c=df.get("cluster", None), cmap="tab10", alpha=0.85)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    if "cluster" in df.columns:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("cluster")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"scatter_{x}_vs_{y}.png", dpi=180)
    plt.close(fig)


def plot_graph(G, df, out_dir):
    if G.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(G, seed=42, k=0.6)
    clusters = df["cluster"].values if "cluster" in df.columns else np.zeros(len(df))
    sizes = 200 + 600 * df.get("anomaly_score", pd.Series(np.zeros(len(df)))).values
    cmap = plt.colormaps["tab10"]  # modern API (no deprecation)
    fig, ax = plt.subplots(figsize=(8, 6))
    if G.number_of_edges() > 0:
        w = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=[2 * ww for ww in w], ax=ax)
    node_colors = [cmap(int((c if c >= 0 else 9) % 10)) for c in clusters]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, ax=ax)
    ax.set_title("Similarity Graph (color=cluster, size=anomaly)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "similarity_graph.png", dpi=200)
    plt.close(fig)


def plot_degree_hist(G, out_dir):
    deg = [d for _, d in G.degree()]
    if not deg:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(deg, bins=range(0, max(deg) + 2), align="left")
    ax.set_xlabel("degree")
    ax.set_ylabel("count")
    ax.set_title("Graph Degree Distribution")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "degree_distribution.png", dpi=180)
    plt.close(fig)


# --------------------- Main Pipeline ---------------------

def collect_images(data_dir, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    p = Path(data_dir)
    return sorted([q for q in p.rglob("*") if q.suffix.lower() in exts])


def infer_label_from_path(p: Path):
    parent = p.parent.name.lower()
    if parent in ("real", "ai", "fake", "synthetic"):
        return "ai" if parent in ("fake", "synthetic") else parent
    return None


def run(data_dir, out_dir, size=256, k=5, anom_thresh=0.8, dbscan_eps=0.8, dbscan_min_samples=5):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = collect_images(data_dir)
    if not paths:
        print(f"[ERR] No images in {data_dir}", file=sys.stderr)
        return 1

    # Extract features
    rows = []
    for p in paths:
        try:
            g = read_image(p, size=size)
            vec = extract_vector(g)
        except Exception as e:
            print("[WARN]", p, e)
            continue
        row = {"path": str(p), "name": p.name, "label": infer_label_from_path(p)}
        row.update(vec)
        rows.append(row)

    if len(rows) < 2:
        print("[ERR] Need at least 2 valid images.", file=sys.stderr)
        return 2

    df = pd.DataFrame(rows)
    X, _ = matrix_X(df)

    # Analytics
    clusters, a_score = cluster_and_anomaly(X, dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples)
    df["cluster"] = clusters
    df["anomaly_score"] = a_score
    df["anomaly_flag"] = (df["anomaly_score"] > anom_thresh).astype(int)

    # Graph
    G, S = knn_graph(X, k=k)
    np.save(Path(out_dir) / "similarity_matrix.npy", S)

    # Save table
    df.to_csv(Path(out_dir) / "features.csv", index=False)

    # Plots
    plot_feature_hists(df, out_dir)
    plot_scatter(df, out_dir)
    plot_graph(G, df, out_dir)
    plot_degree_hist(G, out_dir)

    # ---- Console + text report (flagged images) ----
    anom_df = df[df["anomaly_flag"] == 1][["name", "label", "anomaly_score"]].sort_values(
        "anomaly_score", ascending=False
    )

    print("\n[ANOMALIES]")
    if anom_df.empty:
        print(f"None flagged at threshold = {anom_thresh}")
    else:
        for _, r in anom_df.iterrows():
            print(f"- {r['name']}  label={r['label']}  score={r['anomaly_score']:.3f}")

    # Quick eval if labels exist
    lines = []
    lines.append(f"Threshold: {anom_thresh}")
    lines.append(f"DBSCAN: eps={dbscan_eps}  min_samples={dbscan_min_samples}")
    lines.append(f"Total images: {len(df)} | Flagged anomalies: {len(anom_df)}")

    dfl = df.dropna(subset=["label"]).copy()
    if not dfl.empty:
        dfl["is_ai"] = (dfl["label"].str.lower() == "ai").astype(int)
        tp = int(((df["anomaly_flag"] == 1) & (dfl["is_ai"] == 1)).sum())
        tn = int(((df["anomaly_flag"] == 0) & (dfl["is_ai"] == 0)).sum())
        fp = int(((df["anomaly_flag"] == 1) & (dfl["is_ai"] == 0)).sum())
        fn = int(((df["anomaly_flag"] == 0) & (dfl["is_ai"] == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)

        print(f"[EVAL] TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"[EVAL] Precision={prec:.2f}  Recall={rec:.2f}  Accuracy={acc:.2f}")

        lines.append(f"Confusion Matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
        lines.append(f"Precision={prec:.3f}  Recall={rec:.3f}  Accuracy={acc:.3f}")

    # Save text report
    report_path = Path(out_dir) / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DeepTrace Report\n")
        f.write("\n".join(lines) + "\n\n")
        f.write("Flagged Anomalies (sorted by score):\n")
        if anom_df.empty:
            f.write("None\n")
        else:
            for _, r in anom_df.iterrows():
                f.write(f"{r['name']}\tlabel={r['label']}\tscore={r['anomaly_score']:.3f}\n")

    # Console summary
    print(f"\n[SUMMARY] images={len(df)}, clusters={len(set(clusters))}, anomalies={int(df['anomaly_flag'].sum())}")
    print(f"[OK] Wrote report: {report_path}")
    print(f"[OUT] Results in: {out_dir}")
    return 0


def main():
    p = argparse.ArgumentParser(description="DeepTrace - Visual Signature Graphs for Detecting AI Images")
    p.add_argument("--data_dir", required=True, help="folder with images")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--size", type=int, default=256, help="resize to size x size before analysis")
    p.add_argument("--knn", type=int, default=5, help="k for k-NN similarity graph")
    p.add_argument("--anom_thresh", type=float, default=0.8, help="Anomaly threshold in [0..1]; higher = stricter")
    p.add_argument("--dbscan_eps", type=float, default=0.8, help="DBSCAN eps (cluster tightness)")
    p.add_argument("--dbscan_min_samples", type=int, default=5, help="DBSCAN min_samples")
    args = p.parse_args()

    sys.exit(run(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        size=args.size,
        k=args.knn,
        anom_thresh=args.anom_thresh,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    ))


if __name__ == "__main__":
    main()
