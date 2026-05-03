import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import csv
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

N_SAMPLES = 10
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 512


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    model.eval()

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer, model, generation_config


def build_open_messages(question, explain_reasoning):
    reasoning_instruction = (
        "Explain your reasoning."
        if explain_reasoning
        else "Do not explain your reasoning."
    )

    system_message = (
        f"""You are answering a question.

Instructions:
- Answer the question directly.
- {reasoning_instruction}
"""
    )

    user_message = f"""Question:
{question}
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return messages


def get_open_response(model, tokenizer, generation_config, messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    new_tokens = outputs[0][input_ids.shape[-1]:]

    completion = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    full_output = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return completion, full_output


def main():
    input_file = "data/opinion_qa/opinion_qa_gender_open.csv"
    output_file = "data/response_collection/opinion_qa_open/opinion_qa_llama8b_open_responses.csv"

    tokenizer, model, generation_config = load_model()

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    output_rows = []

    for row in tqdm(rows, desc="OpinionQA Llama8B Open-ended questions"):
        id = row["id"]
        key = row["key"]
        question = row["question"]

        for explain_reasoning in [False, True]:
            reasoning_condition = (
                "no_reasoning"
                if not explain_reasoning
                else "with_reasoning"
            )

            messages = build_open_messages(
                question,
                explain_reasoning
            )

            for sample_idx in range(N_SAMPLES):
                answer, full_output = get_open_response(
                    model,
                    tokenizer,
                    generation_config,
                    messages
                )

                output_rows.append({
                    "id": id,
                    "key": key,
                    "reasoning_condition": reasoning_condition,
                    "sample_id": sample_idx,
                    "question": question,
                    "model_answer": answer,
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