"""Generate measured results, assignment plots, and a self-contained PDF report."""

import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import EngFormatter

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
DOCS = ROOT / "docs"
COLORS = ["#2563eb", "#e76f51", "#059669", "#9333ea", "#c58a00", "#475569"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "#fafbfe",
        "axes.facecolor": "#fafbfe",
        "savefig.facecolor": "#fafbfe",
    }
)


def load_runs():
    runs = []
    for directory in sorted((ROOT / "exp").glob("*")):
        if (
            not (directory / "flags.json").exists()
            or not (directory / "log.csv").exists()
        ):
            continue
        flags = json.loads((directory / "flags.json").read_text())
        with (directory / "log.csv").open() as handle:
            rows = [
                {k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)
            ]
        if not rows or int(rows[-1]["Iteration"]) != flags["n_iter"] - 1:
            continue
        returns = [row["Eval_AverageReturn"] for row in rows]
        threshold = {
            "CartPole-v0": 200,
            "HalfCheetah-v4": 300,
            "LunarLander-v2": 150,
            "InvertedPendulum-v4": 1000,
        }[flags["env_name"]]
        crossings = [
            row["Train_EnvstepsSoFar"]
            for row in rows
            if row["Eval_AverageReturn"] >= threshold
        ]
        runs.append(
            {
                "name": flags["exp_name"],
                "environment": flags["env_name"],
                "seed": flags["seed"],
                "directory": str(directory.relative_to(ROOT)),
                "iterations": len(rows),
                "training_steps": int(rows[-1]["Train_EnvstepsSoFar"]),
                "peak_eval_return": max(returns),
                "final_eval_return": returns[-1],
                "last10_mean_eval_return": float(np.mean(returns[-10:])),
                "threshold": threshold,
                "first_threshold_steps": int(crossings[0]) if crossings else None,
                "flags": flags,
                "rows": rows,
            }
        )
    return runs


def plot_group(
    runs, filename, title, metric="Eval_AverageReturn", threshold=None, labels=None
):
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for i, run in enumerate(runs):
        rows = run["rows"]
        x = [row["Train_EnvstepsSoFar"] for row in rows]
        y = np.array([row[metric] for row in rows])
        label = labels.get(run["name"], run["name"]) if labels else run["name"]
        color = COLORS[i % len(COLORS)]
        ax.plot(x, y, color=color, linewidth=0.7, alpha=0.20)
        # Causal moving average; raw observations remain visible underneath.
        smooth = [np.mean(y[max(0, t - 4) : t + 1]) for t in range(len(y))]
        ax.plot(x, smooth, label=label, color=color, linewidth=1.8)
    if threshold is not None:
        ax.axhline(
            threshold,
            color="#64748b",
            linestyle="--",
            linewidth=1,
            label=f"Target: {threshold:,}",
        )
    ax.set(
        title=title,
        xlabel="Training environment steps (evaluation excluded)",
        ylabel="Training-batch baseline MSE"
        if metric == "Baseline Loss"
        else "Average episodic return",
    )
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, loc="best")
    fig.text(
        0.51,
        -0.025,
        "Seed 1 · faint: raw observations · bold: trailing 5-iteration mean",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.savefig(MEDIA / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return MEDIA / filename


def results_markdown(runs):
    lines = [
        "# Experimental results",
        "",
        "Generated from completed run CSVs by `uv run python scripts/analyze.py`.",
        "",
        (
            "All primary ablations use **seed 1**. Peak return is the best single evaluation batch; "
            "final and trailing-ten means are also reported to expose instability. Evaluation samples "
            "are stochastic and limited; these are not confidence intervals or multi-seed guarantees."
        ),
        "",
        "| Experiment | Peak eval | Final eval | Last 10 mean | Training steps | First target steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        crossing = (
            f"{run['first_threshold_steps']:,}" if run["first_threshold_steps"] else "—"
        )
        lines.append(
            f"| [{run['name']}](../{run['directory']}/log.csv) | {run['peak_eval_return']:.1f} | "
            f"{run['final_eval_return']:.1f} | {run['last10_mean_eval_return']:.1f} | "
            f"{run['training_steps']:,} | {crossing} |"
        )
    lines += [
        "",
        "Targets: CartPole 200; HalfCheetah 300; LunarLander 150; InvertedPendulum 1,000.",
        "",
        "## 1. CartPole: returns, normalization and batch size",
        "",
    ]
    lookup = {run["name"]: run for run in runs if run["seed"] == 1}
    for prefix in ("cartpole", "cartpole_lb"):
        if all(
            name in lookup
            for name in [prefix, prefix + "_rtg", prefix + "_na", prefix + "_rtg_na"]
        ):
            a, b, c, d = [
                lookup[name]
                for name in [
                    prefix,
                    prefix + "_rtg",
                    prefix + "_na",
                    prefix + "_rtg_na",
                ]
            ]
            label = (
                "Small batch (1,000)" if prefix == "cartpole" else "Large batch (4,000)"
            )
            lines += [
                (
                    f"**{label}.** Last-ten mean evaluation returns: trajectory {a['last10_mean_eval_return']:.1f}, "
                    f"reward-to-go {b['last10_mean_eval_return']:.1f}, normalized trajectory "
                    f"{c['last10_mean_eval_return']:.1f}, normalized reward-to-go {d['last10_mean_eval_return']:.1f}. "
                    f"Without normalization, {'reward-to-go' if b['last10_mean_eval_return'] > a['last10_mean_eval_return'] else 'trajectory returns'} "
                    "performed better by this endpoint metric."
                ),
                "",
            ]
    lines += [
        (
            "Reward-to-go is generally preferred because an action cannot influence past rewards. "
            "Removing those past terms preserves the expected score-function gradient while usually "
            "reducing its variance. Relative discounting weights future rewards from each decision."
        ),
        "",
        (
            "Normalization rescales the update and subtracts a batch mean; it is a useful heuristic "
            "but introduces finite-batch bias. Compare normalized and unnormalized rows above rather "
            "than assuming it always helps. Batch size trades gradient noise against interaction cost: "
            "100 iterations at 4,000 steps use roughly four times the data of 1,000-step batches. "
            "The charts compare actual training steps, not iteration counts."
        ),
        "",
        "![Small-batch CartPole](../media/cartpole_small.png)",
        "",
        "![Large-batch CartPole](../media/cartpole_large.png)",
        "",
        "## 2. HalfCheetah: learned baseline",
        "",
    ]
    for name in ("cheetah", "cheetah_baseline", "cheetah_baseline_bgs1"):
        if name in lookup:
            run = lookup[name]
            sentence = f"**{name}:** peak {run['peak_eval_return']:.1f}; final {run['final_eval_return']:.1f}."
            if "Baseline Loss" in run["rows"][0]:
                sentence += (
                    f" Baseline MSE (mean over optimizer steps on each training batch) changed from "
                    f"{run['rows'][0]['Baseline Loss']:.1f} to {run['rows'][-1]['Baseline Loss']:.1f}."
                )
            lines += [sentence, ""]
    lines += [
        (
            "The one-gradient-step ablation keeps the baseline learning rate at 0.01 and "
            "reduces value updates from five to one. This slows fitting to the current batch; "
            "policy outcomes above show whether that change helped this seed. Value targets "
            "change with the policy, so training MSE need not decrease monotonically, and lower "
            "training MSE alone does not prove better control. The baseline uses raw targets."
        ),
        "",
        "![HalfCheetah return](../media/cheetah_returns.png)",
        "",
        "![Baseline training loss](../media/cheetah_baseline_loss.png)",
        "",
        "## 3. LunarLander: generalized advantage estimation",
        "",
    ]
    lunar = [r for r in runs if r["name"].startswith("lunar_lander")]
    if lunar:
        best = max(lunar, key=lambda r: r["peak_eval_return"])
        stable = max(lunar, key=lambda r: r["last10_mean_eval_return"])
        lines += [
            (
                f"Best peak: **{best['name']}**, {best['peak_eval_return']:.1f}. Best trailing-ten "
                f"mean: **{stable['name']}**, {stable['last10_mean_eval_return']:.1f}."
            ),
            "",
        ]
    lines += [
        (
            "λ=0 uses one-step TD residuals, reducing variance while depending heavily on value "
            "accuracy. λ=1 telescopes to Monte Carlo reward-to-go minus the baseline under the "
            "zero-tail boundary convention, reducing bootstrap bias while increasing variance. "
            "Intermediate λ mixes horizons. The observed ranking above and curves below reflect "
            "this seed, not a universal monotonic relationship. All five runs retain the PDF’s "
            "fixed batch size, learning rate, architecture, discount, and iteration budget."
        ),
        "",
        (
            "The official starter uses standard LunarLander-v2 without an additional action-noise "
            "wrapper; this protocol is preserved. See [PROVENANCE.md](PROVENANCE.md)."
        ),
        "",
        "![GAE comparison](../media/lunar_gae.png)",
        "",
        "## 4. InvertedPendulum: sample efficiency",
        "",
    ]
    pendulum = [r for r in runs if r["name"].startswith("pendulum")]
    successful = [r for r in pendulum if r["first_threshold_steps"]]
    if successful:
        best = min(successful, key=lambda r: r["first_threshold_steps"])
        lines += [
            (
                f"Earliest observed 1,000-return evaluation: **{best['name']}**, at "
                f"**{best['first_threshold_steps']:,} training steps**. "
                f"The below-100K target {'was met' if best['first_threshold_steps'] < 100000 else 'was not met'}."
            ),
            "",
            "```json",
            json.dumps(
                {
                    k: best["flags"][k]
                    for k in (
                        "batch_size",
                        "discount",
                        "learning_rate",
                        "n_layers",
                        "layer_size",
                        "use_reward_to_go",
                        "normalize_advantages",
                        "use_baseline",
                        "baseline_gradient_steps",
                        "gae_lambda",
                    )
                },
                indent=2,
            ),
            "```",
            "",
        ]
    else:
        lines += ["No completed run has reached the 1,000-return target yet.", ""]
    lines += [
        (
            "Tuning reduces batches from 5,000 to 500 or 1,000 transitions to allow more "
            "frequent policy updates per interaction budget, and combines reward-to-go, "
            "normalization, a value baseline, γ=0.99 and GAE. The candidates also vary learning "
            "rate and λ. These coupled changes are a configuration search, not a controlled "
            "causal attribution of gains to any one hyperparameter. All attempts are retained."
        ),
        "",
        "![Pendulum sample efficiency](../media/pendulum_efficiency.png)",
        "",
        "## Reproduction and limitations",
        "",
        (
            "Exact command lines are in [commands.sh](commands.sh); per-run flags and source "
            "revision are beside each CSV. The eight CartPole, three HalfCheetah, five LunarLander, "
            "and four InvertedPendulum configurations are generated by `scripts/experiments.py`."
        ),
        "",
        (
            "Single-seed comparisons and peak selection can be optimistic. A larger study should "
            "repeat every setting over independent seeds and evaluate saved policies over many "
            "held-out episodes. Training steps exclude evaluation interactions as required by the "
            "assignment. Time-limit tails are zero, not bootstrapped. Videos show individual rollouts "
            "of final checkpoints and do not establish average performance."
        ),
        "",
    ]
    return "\n".join(lines)


def make_pdf(markdown, plots):
    with PdfPages(DOCS / "report.pdf") as pdf:
        # Keep the written analysis readable and include the complete result table separately.
        text = markdown.split("## 1.")[1:]
        overview = [
            "Policy Gradient Lab — Experiment Report",
            "",
            "Berkeley CS285 Spring 2026 · Assignment 2",
            "",
            "REINFORCE, reward-to-go, learned value baselines, GAE and sample efficiency.",
            "",
            "All metrics are computed from saved run CSVs. Primary comparisons use seed 1.",
            "See docs/RESULTS.md for clickable raw logs, exact flags, and full discussion.",
        ]

        def text_pages(lines):
            wrapped = []
            for line in lines:
                wrapped.extend(
                    textwrap.wrap(line, width=102, replace_whitespace=False) or [""]
                )
            for start in range(0, len(wrapped), 58):
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(
                    0.07,
                    0.95,
                    "\n".join(wrapped[start : start + 58]),
                    va="top",
                    fontsize=9,
                    linespacing=1.4,
                    fontfamily="DejaVu Sans",
                )
                pdf.savefig(fig)
                plt.close(fig)

        text_pages(overview)
        runs = load_runs()
        text_pages(
            [
                "Measured evaluation returns (seed 1)",
                "",
                "Experiment | peak | final | last 10 mean | first target steps",
                "",
            ]
            + [
                f"{r['name']} | {r['peak_eval_return']:.1f} | {r['final_eval_return']:.1f} | "
                f"{r['last10_mean_eval_return']:.1f} | {r['first_threshold_steps']}"
                for r in runs
            ]
        )
        analysis = "## 1." + "## 1.".join(text) if text else markdown
        clean_lines = [
            line.replace("**", "").replace("`", "")
            for line in analysis.splitlines()
            if not line.startswith("![")
        ]
        text_pages(clean_lines)
        for plot in plots:
            if plot:
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.imshow(plt.imread(plot))
                ax.axis("off")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        # Include exact CLI configurations in the report, not just a reference.
        text_pages(
            [
                "Exact experiment commands",
                "",
                *(DOCS / "commands.sh").read_text().splitlines(),
            ]
        )


def main():
    MEDIA.mkdir(exist_ok=True)
    runs = load_runs()
    plots = []
    plots.append(
        plot_group(
            [
                r
                for r in runs
                if r["name"].startswith("cartpole") and "_lb" not in r["name"]
            ],
            "cartpole_small.png",
            "CartPole · 1,000-step training batches",
            metric="Train_AverageReturn",
            threshold=200,
        )
    )
    plots.append(
        plot_group(
            [r for r in runs if r["name"].startswith("cartpole_lb")],
            "cartpole_large.png",
            "CartPole · 4,000-step training batches",
            metric="Train_AverageReturn",
            threshold=200,
        )
    )
    cheetah = [r for r in runs if r["name"].startswith("cheetah")]
    plots.append(
        plot_group(
            cheetah,
            "cheetah_returns.png",
            "HalfCheetah · value baseline comparison",
            threshold=300,
        )
    )
    plots.append(
        plot_group(
            [r for r in cheetah if "Baseline Loss" in r["rows"][0]],
            "cheetah_baseline_loss.png",
            "HalfCheetah · fitting the value baseline",
            metric="Baseline Loss",
        )
    )
    plots.append(
        plot_group(
            [r for r in runs if r["name"].startswith("lunar_lander")],
            "lunar_gae.png",
            "LunarLander · GAE bias–variance tradeoff",
            threshold=150,
        )
    )
    plots.append(
        plot_group(
            [r for r in runs if r["name"].startswith("pendulum")],
            "pendulum_efficiency.png",
            "InvertedPendulum · sample efficiency",
            threshold=1000,
        )
    )
    summary = [
        {k: v for k, v in run.items() if k not in ("rows", "flags")} for run in runs
    ]
    (DOCS / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = results_markdown(runs)
    (DOCS / "RESULTS.md").write_text(markdown)
    make_pdf(markdown, plots)
    print(
        f"Analyzed {len(runs)} completed runs; generated {sum(p is not None for p in plots)} plots and docs/report.pdf."
    )


if __name__ == "__main__":
    main()
