import pandas as pd
import os
import re

def extract_wave(key):
    match = re.search(r'_W(\d+)$', key)
    if match:
        return match.group(1)


def compute_weighted_distribution(df, response_col, weight_col):
    df_valid = df[df[response_col].notna()].copy()   # drop NA (question not asked)
    
    weighted_counts = df_valid.groupby(response_col)[weight_col].sum()

    weighted_share = weighted_counts / weighted_counts.sum()
    
    return weighted_share.reset_index().rename(columns={
        response_col: "option",
        weight_col: "weighted_share"
    })


def process_questions(question_table, base_path):
    results = []
    
    for _, row in question_table.iterrows():
        key = row['key']
        wave = extract_wave(key)
        
        file_path = os.path.join(base_path, wave, "responses.csv")
        
        df = pd.read_csv(file_path, low_memory=False)
        
        weight_col = f"WEIGHT_W{wave}"
        
        dist = compute_weighted_distribution(df, key, weight_col)
        dist['key'] = key
        dist['wave'] = wave
        
        results.append(dist)

    return pd.concat(results, ignore_index=True)

opinionqa = pd.read_csv("data/opinion_qa/opinion_qa_gender.csv", sep=';')
keys = opinionqa['key']

base_path = "santukar_opinionqa_qs"

result = process_questions(opinionqa, base_path)

result = result.drop(columns=['wave'])

result = result[['key', 'option', 'weighted_share']]

result.to_csv("data/opinion_qa/opinion_qa_weighted_distributions.csv", index=False)