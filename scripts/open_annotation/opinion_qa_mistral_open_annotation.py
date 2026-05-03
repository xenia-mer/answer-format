import os
import csv
import ast
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DEVICE = torch.device("cuda")

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

MAX_NEW_TOKENS = 32 # for longer answers


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config
    )

    model.eval()

    generation_config = GenerationConfig(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer, model, generation_config


def build_annotation_messages(question, options, model_answer):
    system_message = """You are an annotation system.

Your task is to map a model's answer to one of the provided multiple-choice options.

Rules:
- Select the option that best matches the model's answer.
- If the answer expresses uncertainty or says it cannot be determined, select the appropriate uncertainty option.
- Only output the exact text of one of the options.
- Do not explain your reasoning.
- Do not output anything other than the selected option.
"""

    options_block = "\n".join(
        [f"{i+1}. {opt}" for i, opt in enumerate(options)]
    )

    user_message = f"""Question:
{question}

Options:
{options_block}

Model answer:
{model_answer}

Which option best matches the model answer?
Output the exact option text.
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return messages


def get_annotation(model, tokenizer, generation_config, messages):
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

    return completion


def map_text_to_option(annotation, options):
    annotation = annotation.strip() # normalise

    annotation = re.sub(r"^\d+[\.\)]\s*", "", annotation) # remove enumeration

    for opt in options: # exact
        if annotation == opt.strip():
            return opt

    for opt in options: # containment
        if opt.strip().lower() in annotation.lower():
            return opt

    return None


def main():
    input_file = "data/response_collection/opinion_qa_open/opinion_qa_mistral7b_open_responses.csv"
    original_mc_file = "data/opinion_qa/opinion_qa_gender.csv"
    output_file = "data/open_annotation/opinion_qa_mistral7b_open_responses_annotation_unprocessed.csv"

    tokenizer, model, generation_config = load_model()

    with open(original_mc_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        mc_rows = list(reader)

    mc_lookup = {
        row["key"]: {
            "question": row["question"],
            "options": ast.literal_eval(row["options"])
        }
        for row in mc_rows
    }

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        open_rows = list(reader)

    output_rows = []

    for row in tqdm(open_rows, desc="OpinionQA Mistral7B Open questions Qwen annotation"):
        key = row["key"]
        model_answer = row["model_answer"]

        mc_entry = mc_lookup[key]
        question = mc_entry["question"]
        options = mc_entry["options"]

        messages = build_annotation_messages(
            question,
            options,
            model_answer
        )

        annotation = get_annotation(
            model,
            tokenizer,
            generation_config,
            messages
        )

        annotated_option = map_text_to_option(annotation, options)

        output_row = row.copy()
        output_row["annotated_label"] = annotated_option
        output_row["raw_annotation_output"] = annotation # for manual fuzzy mapping later

        if annotated_option is None:
            output_row["annotation_status"] = "unmapped"
        else:
            output_row["annotation_status"] = "mapped"

        output_rows.append(output_row)

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