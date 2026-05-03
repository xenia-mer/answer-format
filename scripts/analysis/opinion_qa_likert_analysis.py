import pandas as pd
import numpy as np
import ast
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def load_likert(path, scale_condition):
    df = pd.read_csv(path)
    df["scale_condition"] = scale_condition
    return df


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

            # no ordinal to Refused
            if opt != "Refused" and ordinal_idx < len(ordinals):
                option_to_ordinal[opt] = ordinals[ordinal_idx]
                ordinal_idx += 1

        maps[key] = {
            "options": options,
            "option_to_index": option_to_index,
            "option_to_ordinal": option_to_ordinal
        }

    return maps

# to [0,1] accounting for polarity, ordinal position (distribution-based, not directional)
def normalise_likert(rating, scale, polarity):
    x = (rating - 1) / (scale - 1)

    if polarity == "neg":
        x = 1 - x

    return x


def compute_model_distribution(df, key, option_map):
    subset = df[df["key"] == key]

    K = len(option_map["options"])
    scores = np.zeros(K)

    for _, row in subset.iterrows():
        opt = row["option"]
        
        if opt not in option_map["option_to_index"]:
            continue
            
        idx = option_map["option_to_index"][opt]
        
        # parse scale and polarity from scale_condition
        try:
            scale = int(row["scale_condition"].split("_")[0])
            polarity = row["scale_condition"].split("_")[1]
        except (ValueError, IndexError, AttributeError) as e:
            continue
        
        # skip NaN ratings
        rating = row["likert_rating"]
        if pd.isna(rating):
            continue
            
        scores[idx] += normalise_likert(rating, scale, polarity)

    if scores.sum() > 0:
        probs = scores / scores.sum()
    else:
        # return uniform distribution if no data
        probs = np.ones(K) / K

    return probs


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
    else:
        # return uniform distribution if no data
        probs = np.ones(len(option_map["options"])) / len(option_map["options"])

    return probs


def compute_wasserstein(p, q, option_map):
    opts = [
        opt for opt in option_map["options"]
        if opt != "Refused" and opt in option_map["option_to_ordinal"]
    ]

    if len(opts) == 0:
        return 0.0

    idxs = [option_map["option_to_index"][o] for o in opts]

    p_w = np.array(p[idxs], dtype=float)
    q_w = np.array(q[idxs], dtype=float)
    
    # replace any NaN or inf values with 0
    p_w = np.nan_to_num(p_w, nan=0.0, posinf=0.0, neginf=0.0)
    q_w = np.nan_to_num(q_w, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ensure non-negative
    p_w = np.maximum(p_w, 0)
    q_w = np.maximum(q_w, 0)
    
    # normalize
    p_sum = p_w.sum()
    q_sum = q_w.sum()
    
    if p_sum > 1e-10:
        p_w = p_w / p_sum
    else:
        return 0.0
        
    if q_sum > 1e-10:
        q_w = q_w / q_sum
    else:
        return 0.0

    ord_vals = [option_map["option_to_ordinal"][o] for o in opts]

    try:
        w = wasserstein_distance(ord_vals, ord_vals, p_w, q_w)
    except ValueError as e:
        print(f"  Wasserstein calculation failed: {e}")
        print(f"  p_w: {p_w}, sum={p_w.sum()}")
        print(f"  q_w: {q_w}, sum={q_w.sum()}")
        return 0.0

    # normalise by range of ordinal values
    rng = max(ord_vals) - min(ord_vals)
    return w / rng if rng > 0 else 0.0


def compute_jsd(p, q):
    eps = 1e-12
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    # replace NaN/inf with 0
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    
    p = p + eps
    q = q + eps

    p = p / p.sum()
    q = q / q.sum()

    return jensenshannon(p, q, base=2)


def evaluate_model(model_df, human_df, orig_df, alpha=0.5, groupby=None):
    maps = build_option_maps(orig_df)
    results = []

    def run_eval(df_subset, group_name, key, option_map):
        try:
            p = compute_human_distribution(human_df, key, option_map)
            q = compute_model_distribution(df_subset, key, option_map)

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
        except Exception as e:
            print(f"Error processing key '{key}' for group '{group_name}': {str(e)}")
            return None

    if groupby is None:
        for key, option_map in maps.items():
            if key not in model_df["key"].values:
                continue
            result = run_eval(model_df, "all", key, option_map)
            if result is not None:
                results.append(result)
    else:
        for g in model_df[groupby].dropna().unique():
            df_g = model_df[model_df[groupby] == g]

            for key, option_map in maps.items():
                if key not in df_g["key"].values:
                    continue
                result = run_eval(df_g, g, key, option_map)
                if result is not None:
                    results.append(result)

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

gemma_likert = pd.concat([
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_2_pos.csv", "2_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_2_neg.csv", "2_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_4_pos.csv", "4_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_4_neg.csv", "4_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_5_pos.csv", "5_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_5_neg.csv", "5_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_9_pos.csv", "9_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_9_neg.csv", "9_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_10_pos.csv", "10_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_gemma12b_likert_10_neg.csv", "10_neg")
], ignore_index=True)

llama_likert = pd.concat([
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_2_pos.csv", "2_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_2_neg.csv", "2_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_4_pos.csv", "4_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_4_neg.csv", "4_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_5_pos.csv", "5_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_5_neg.csv", "5_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_9_pos.csv", "9_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_9_neg.csv", "9_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_10_pos.csv", "10_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_llama8b_likert_10_neg.csv", "10_neg")
], ignore_index=True)

mistral_likert = pd.concat([
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_2_pos.csv", "2_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_2_neg.csv", "2_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_4_pos.csv", "4_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_4_neg.csv", "4_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_5_pos.csv", "5_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_5_neg.csv", "5_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_9_pos.csv", "9_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_9_neg.csv", "9_neg"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_10_pos.csv", "10_pos"),
    load_likert("data/response_collection/opinion_qa_likert/opinion_qa_mistral7b_likert_10_neg.csv", "10_neg")
], ignore_index=True)


models = {
    "Gemma": gemma_likert,
    "Llama": llama_likert,
    "Mistral": mistral_likert
}

print("==Overall==")

for name, model_df in models.items():
    print(f"\n{name}")

    df_res, summary = evaluate_model(model_df, human, orig)
    print("  Overall:")
    print(summary)

    df_cond, summary_cond = evaluate_model(
        model_df, human, orig,
        groupby="scale_condition"
    )

    print("\n  By scale condition:")
    print(summary_cond)

print("==CI==")

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

print("\n==Per scale_condition==")

for name, df_model in models.items():
    print(f"\n{name}:")
    
    for cond in sorted(df_model["scale_condition"].dropna().unique()):
        df_cond = df_model[df_model["scale_condition"] == cond]
        
        boot = bootstrap_metrics(df_cond, human, orig, B=1000)
        if not boot.empty:
            ci_results = compute_ci(boot)
            
            print(f"  Condition: {cond}")
            for metric, res in ci_results.items():
                print("     ", format_ci(metric, res))
        else:
            print(f"  Condition {cond}: No bootstrap results generated")