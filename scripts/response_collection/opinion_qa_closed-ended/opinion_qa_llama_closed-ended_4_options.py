import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import csv
import torch
import ast
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cuda")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

N_SAMPLES = 10
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 16


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model.to(DEVICE)
    model.eval()

    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def build_mc_messages(question, answers):
    letters = ["A", "B", "C", "D"]

    options = "\n".join(f"{l}. {a}" for l, a in zip(letters, answers))

    system_message = """You are answering a multiple-choice question.

Instructions:
- Choose exactly ONE option.
- Answer using ONLY the letter (A, B, C, or D).
- Do not explain your reasoning."""

    user_message = f"""Question:
{question}

Options:
{options}

Answer:"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return messages


def get_model_answer(model, tokenizer, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        return_dict=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"\b([ABCD])\b", completion)
    choice = match.group(1) if match else None

    return choice, full_output


def main():
    input_file = "data/opinion_qa/opinion_qa_closed-ended/opinion_qa_gender_4_options_named_permutations.csv"
    output_file = "data/response_collection/opinion_qa_closed-ended/opinion_qa_llama8b_closed-ended_4_options.csv"

    tokenizer, model = load_model()

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    answer_orders = [
        "options_1234",
        "options_2341",
        "options_3412",
        "options_4123",
        "options_4321",
        "options_4231"
    ]

    output_rows = []

    for row in tqdm(rows, desc="OpinionQA Llama8B Closed-ended 4-option questions"):
        id = row["id"]
        key = row["key"]
        question = row["question"]

        for order_key in answer_orders:
            answers = ast.literal_eval(row[order_key])

            prompt = build_mc_messages(question, answers)

            for sample_idx in range(N_SAMPLES):
                choice, full_output = get_model_answer(
                    model, tokenizer, prompt
                )

                chosen_answer = None
                if choice:
                    chosen_answer = answers[["A", "B", "C", "D"].index(choice)]

                output_rows.append({
                    "id": id,
                    "key": key,
                    "answer_order": order_key,
                    "sample_id": sample_idx,
                    "answers": answers,
                    "chosen_letter": choice,
                    "chosen_answer": chosen_answer,
                    "full_output": full_output,
                    "model": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P
                })

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=output_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("Done.")


if __name__ == "__main__":
    main()