import os
import csv
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

NEW_PROMPT = """You are converting survey questions into Likert-scale questions.

Use the following structure:
- a question synthesized from original context
- the phrase: "To what extent do you agree with the following statement:"
- a declarative statement corresponding only to the current answer option
- the phrase: "Please rate your agreement from 1 (strongly agree) to 4 (strongly disagree)."

Rules:
- Do not explain your reasoning.
- Do not add notes or commentary.
- Do not invent new information.
- Produce exactly one Likert-scale question.
- The sentence before the Likert statement must be a question.
- The Likert statement must express the full proposition implied by choosing the given answer option.

Example:

Question:
Please indicate whether you think the following is a reason why black people in our country may have a harder time getting ahead than white people. Less access to high-paying jobs

Answer option:
Major reason

Output:
Would you say having less access to high-paying jobs is a reason why black people may have a harder time getting ahead than white people? To what extent do you agree with the following statement: Black people having less access to high-paying jobs is a major reason why they may have a harder time getting ahead than white people. Please rate your agreement from 1 (strongly agree) to 4 (strongly disagree).

---

After generating, perform a validation pass.
The output must be grammatically correct and faithful to the original information.
If the output is not faithful to the original information, regenerate.
If the output is not grammatically correct, fix the grammar.
Output only the final validated Likert-scale question. Do not explain your reasoning and do not add notes or commentary.

Now perform the task below.
"""

def build_prompt(question, option):
    return (
        NEW_PROMPT
        + f"""
Question:
{question}

Answer option:
{option}

Output:
"""
    )


def rewrite_to_likert(model, tokenizer, question, option):
    prompt = build_prompt(question, option)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Output:" in decoded:
        decoded = decoded.split("Output:")[-1].strip()

    return decoded


def main():
    input_file = "data/opinion_qa/opinion_qa_gender.csv"
    output_file = "data/opinion_qa/opinion_qa_gender_likert_4_neg_subset.csv"

    rows = []
    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            row = {k.strip(): v for k, v in row.items() if k is not None}
            if not row["key"].startswith("WHYNOT"): # only failed Qs stay
                continue
            rows.append(row)

    tokenizer, model = load_model()

    output_rows = []

    for row in tqdm(rows, desc="Processing questions"):
        key = row["key"]
        question = row["question"]

        options = ast.literal_eval(row["options"])

        for option in options:
            if option == "Refused":
                continue

            likert_question = rewrite_to_likert(
                model,
                tokenizer,
                question,
                option
            )

            output_rows.append({
                "key": key,
                "option": option,
                "likert_question": likert_question,
                "scale": "1=Strongly agree, 2=Agree, 3=Disagree, 4=Strongly disagree"
            })

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "option",
                "likert_question",
                "scale"
            ]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("Done.")


if __name__ == "__main__":
    main()