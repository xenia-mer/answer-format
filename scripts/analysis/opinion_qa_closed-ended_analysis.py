import pandas as pd
import numpy as np
import ast
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def safe_parse(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def build_option_maps(orig_df):
    maps = {}

    for _, row in orig_df.iterrows():
        key = row["key"]

        options = safe_parse(row["options"])
        ordinals = safe_parse(row["option_ordinal"])

        option_to_index = {}
        option_to_ordinal = {}

        ordinal_idx = 0

        for opt in options:
            option_to_index[opt] = len(option_to_index)

            if opt != "Refused" and ordinal_idx < len(ordinals):
                option_to_ordinal[opt] = ordinals[ordinal_idx]
                ordinal_idx += 1

        maps[key] = {
            "options": options,
            "option_to_index": option_to_index,
            "option_to_ordinal": option_to_ordinal
        }

    return maps


def compute_model_distribution(df, key, option_map):
    subset = df[df["key"] == key]

    K = len(option_map["options"])
    counts = np.zeros(K)

    for ans in subset["chosen_answer"]:
        if ans in option_map["option_to_index"]:
            idx = option_map["option_to_index"][ans]
            counts[idx] += 1

    probs = counts / counts.sum() if counts.sum() > 0 else counts

    refused_idx = option_map["option_to_index"].get("Refused", None)
    p_refused = probs[refused_idx] if refused_idx is not None else 0.0

    return probs, p_refused


def compute_human_distribution(human_df, key, option_map):
    subset = human_df[human_df["key"] == key]

    probs = np.zeros(len(option_map["options"]))

    for _, row in subset.iterrows():
        opt = row["option"]
        val = row["weighted_share"]

        if opt in option_map["option_to_index"]:
            idx = option_map["option_to_index"][opt]
            probs[idx] = val

    if probs.sum() > 0:
        probs = probs / probs.sum()

    refused_idx = option_map["option_to_index"].get("Refused", None)
    p_refused = probs[refused_idx] if refused_idx is not None else 0.0

    return probs, p_refused


def compute_wasserstein(p, q, option_map):
    opts = [
        opt for opt in option_map["options"]
        if opt != "Refused" and opt in option_map["option_to_ordinal"]
    ]

    if len(opts) == 0:
        return 0.0

    idxs = [option_map["option_to_index"][o] for o in opts]

    p_w = p[idxs]
    q_w = q[idxs]

    if p_w.sum() == 0 or q_w.sum() == 0:
        return 0.0

    ord_vals = [option_map["option_to_ordinal"][o] for o in opts]

    w = wasserstein_distance(ord_vals, ord_vals, p_w, q_w)

    rng = max(ord_vals) - min(ord_vals)
    return w / rng if rng > 0 else 0.0


def compute_jsd(p, q):
    eps = 1e-12
    p = p + eps
    q = q + eps

    p = p / p.sum()
    q = q / q.sum()

    return jensenshannon(p, q, base=2)


def evaluate_model(model_df, human_df, orig_df, alpha=0.5, groupby=None):
    maps = build_option_maps(orig_df)
    results = []

    def run_eval(df_subset, group_name, key, option_map):
        p, _ = compute_human_distribution(human_df, key, option_map)
        q, _ = compute_model_distribution(df_subset, key, option_map)

        w = compute_wasserstein(p, q, option_map)
        js = compute_jsd(p, q)

        combined = alpha * w + (1 - alpha) * js

        return {
            "group": group_name,
            "key": key,
            "wasserstein": w,
            "jsd": js,
            "combined": combined
        }

    if groupby is None:
        for key, option_map in maps.items():
            if key not in model_df["key"].values:
                continue
            results.append(run_eval(model_df, "all", key, option_map))
    else:
        for g in model_df[groupby].dropna().unique():
            df_g = model_df[model_df[groupby] == g]

            for key, option_map in maps.items():
                if key not in df_g["key"].values:
                    continue
                results.append(run_eval(df_g, g, key, option_map))

    df = pd.DataFrame(results)

    summary = df.groupby("group")[[
        "wasserstein",
        "jsd",
        "combined"
    ]].mean()

    return df, summary


def bootstrap_metrics(model_df, human_df, orig_df, B=1000, alpha=0.5):
    bootstrap_results = []

    for b in range(B):
        # sample rows with replacement from the entire dataframe
        boot_df = model_df.sample(n=len(model_df), replace=True)

        _, summary = evaluate_model(
            boot_df,
            human_df,
            orig_df,
            alpha=alpha
        )

        if not summary.empty and "all" in summary.index:
            bootstrap_results.append(summary.loc["all"])

    return pd.DataFrame(bootstrap_results)


# descriptive uncertainty quantification
def compute_ci(bootstrap_df):
    results = {}

    for col in bootstrap_df.columns:
        values = bootstrap_df[col].values

        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)

        results[col] = {
            "mean": values.mean(),
            "ci_95": (ci_low, ci_high)
        }

    return results


def format_ci(name, result):
    ci_low, ci_high = result["ci_95"]

    # check for degenerate CIs
    if abs(ci_high - ci_low) < 1e-6:
        warning = " [DEGENERATE]"
        return f"{name}: mean={result['mean']:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]{warning}"
    else:
        return f"{name}: mean={result['mean']:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]"


orig = pd.read_csv("data/opinion_qa/opinion_qa_gender_ordinal.csv")
human = pd.read_csv("data/opinion_qa/opinion_qa_weighted_distributions.csv")

gemma_closed = pd.concat([
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_gemma12b_closed-ended_3_options.csv"),
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_gemma12b_closed-ended_4_options.csv"),
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_gemma12b_closed-ended_5_options.csv"),
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_gemma12b_closed-ended_6_options.csv"),
], ignore_index=True)

llama_closed = pd.concat([
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_llama8b_closed-ended_3_options.csv"),
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_llama8b_closed-ended_4_options.csv"),
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_llama8b_closed-ended_5_options.csv"),
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_llama8b_closed-ended_6_options.csv"),
], ignore_index=True)

mistral_closed = pd.concat([
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_mistral7b_closed-ended_3_options.csv"),
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_mistral7b_closed-ended_4_options.csv"),
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_mistral7b_closed-ended_5_options.csv"),
    pd.read_csv("data/response_collection/opinion_qa_closed-ended/opinion_qa_mistral7b_closed-ended_6_options.csv"),
], ignore_index=True)

models = {
    "Gemma": gemma_closed,
    "Llama": llama_closed,
    "Mistral": mistral_closed
}

print("==Overall==")

for name, model_df in models.items():
    print(f"\n{name}")

    df_res, summary = evaluate_model(model_df, human, orig)
    print("  Overall:")
    print(summary)

    df_cond, summary_cond = evaluate_model(
        model_df, human, orig,
        groupby="answer_order"
    )

    print("\n  By answer_order:")
    print(summary_cond)

print("==CI (effect size estimation)==")

print("\n==Overall==")

for name, df_model in models.items():
    print(f"\n{name}:")
    
    boot = bootstrap_metrics(df_model, human, orig, B=1000)
    if not boot.empty:
        ci_results = compute_ci(boot)
        
        for metric, res in ci_results.items():
            print("  ", format_ci(metric, res))
    else:
        print("  No bootstrap results generated")

print("\n==Per answer_order==")

for name, df_model in models.items():
    print(f"\n{name}:")
    
    for cond in sorted(df_model["answer_order"].dropna().unique()):
        df_cond = df_model[df_model["answer_order"] == cond]
        
        boot = bootstrap_metrics(df_cond, human, orig, B=1000)
        if not boot.empty:
            ci_results = compute_ci(boot)
            
            print(f"  Condition: {cond}")
            for metric, res in ci_results.items():
                print("     ", format_ci(metric, res))
        else:
            print(f"  Condition {cond}: No bootstrap results generated")