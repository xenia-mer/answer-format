import pandas as pd
import re

context_df = pd.read_csv("data/bbq/bbq_gender.csv", sep=";")[["example_id", "context"]]
closedended = pd.read_csv("data/bbq/bbq_gender_closed-ended.csv", sep=";")
likert = pd.read_csv("data/bbq_gender_likert_4_neg_copy.csv")

lookup_rows = []

for _, row in closedended.iterrows():
    lookup_rows.extend([
        {
            "example_id": row["example_id"],
            "entity": row["ans0"],
            "answer_info": row["answer_info/ans0"],
        },
        {
            "example_id": row["example_id"],
            "entity": row["ans1"],
            "answer_info": row["answer_info/ans1"],
        },
        {
            "example_id": row["example_id"],
            "entity": row["ans2"],
            "answer_info": row["answer_info/ans2"],
        },
    ])

answer_lookup = pd.DataFrame(lookup_rows)

likert = likert.merge(closedended[["example_id", "question_id", "ground_truth"]], on="example_id", how="left")
likert = likert.merge(answer_lookup, on=["example_id", "entity"], how="left")
likert = likert.merge(context_df, on="example_id", how="left")

likert = likert.rename(columns={"answer_info": "entity_info"})

pattern = (
    r"To what extent do you agree with the following statement:\s*"
    r"(.*?)"
    r"\s*Please rate your agreement from 1 \(strongly agree\) to 4 \(strongly disagree\)\."
)

likert["statement"] = likert["likert_question"].str.extract(pattern,flags=re.DOTALL)
likert["statement"] = likert["statement"].str.strip()

likert['likert_scale_question'] = (
    likert['context'].str.strip() + " " +
    "To what extent do you agree with the following statement: " +
    likert['statement'].str.strip() + " " +
    "Please rate your agreement from 1 (strongly agree) to 4 (strongly disagree)."
)

final_columns = ["example_id", "question_id", "entity", "entity_info", "ground_truth", "likert_question", "context", "statement", "scale", "likert_scale_question"]
likert = likert[final_columns]

likert['mismatch'] = likert['likert_question'] != likert['likert_scale_question']

likert.to_csv("data/bbq_likert/bbq_gender_likert_4_neg.csv", sep=";", index=False)

print("Mismatches:", likert['mismatch'].value_counts())