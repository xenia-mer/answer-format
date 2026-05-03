import pandas as pd

df = pd.read_csv("data/bbq/bbq_gender.csv", delimiter=';')

df["question_id"] = (
    df["question_index"].astype(str) + "_" +
    df["question_polarity"] + "_" +
    df["context_condition"]
)

def map_gender(value):
    value = str(value).strip().lower()

    if value in {"f", "woman", "girl"}:
        return "F"
    if value in {"m", "man", "boy"}:
        return "M"
    if value == "unknown":
        return "U"

answer_columns = [col for col in df.columns if col.startswith("answer_info/ans") and col.endswith("/1")]

for col in answer_columns:
    new_col = col.replace("/1", "")
    df[new_col] = df[col].apply(map_gender)

def get_ground_truth(row):
    label_index = int(row["label"])
    return row[f"answer_info/ans{label_index}"]

df["ground_truth"] = df.apply(get_ground_truth, axis=1)

new_df = pd.DataFrame()

new_df["example_id"] = df["example_id"]
new_df["question_id"] = df["question_id"]
new_df["full_question"] = df["context"].str.strip() + " " + df["question"].str.strip()

new_df["ans0"] = df["ans0/0"]
new_df["answer_info/ans0"] = df["answer_info/ans0"]

new_df["ans1"] = df["ans1/0"]
new_df["answer_info/ans1"] = df["answer_info/ans1"]

new_df["ans2"] = df["ans2/0"]
new_df["answer_info/ans2"] = df["answer_info/ans2"]

new_df["label"] = df["label"]
new_df["ground_truth"] = df["ground_truth"]

orders = {
    "MFU": ["M", "F", "U"],
    "FMU": ["F", "M", "U"],
    "MUF": ["M", "U", "F"],
    "FUM": ["F", "U", "M"],
    "UMF": ["U", "M", "F"],
    "UFM": ["U", "F", "M"],
}

def reorder_answers(row, order):
    mapping = {
        row["answer_info/ans0"]: row["ans0"],
        row["answer_info/ans1"]: row["ans1"],
        row["answer_info/ans2"]: row["ans2"],
    }
    return [mapping[o] for o in order]

for name, order in orders.items():
    new_df[f"answers_{name}"] = new_df.apply(
        lambda row: reorder_answers(row, order),
        axis=1
    )

new_df.to_csv("data/bbq/bbq_gender_closed-ended.csv", index=False, sep=';')