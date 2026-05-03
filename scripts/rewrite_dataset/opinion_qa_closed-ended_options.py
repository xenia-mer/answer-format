import pandas as pd
import ast

df = pd.read_csv('data/opinion_qa/opinion_qa_gender.csv', sep=';')

df['options'] = df['options'].apply(ast.literal_eval)
df['num_options'] = df['options'].apply(len)
unique_option_counts = df['num_options'].unique()

for n in unique_option_counts:
    subset = df[df['num_options'] == n].copy()
    subset.to_csv(f'data/opinion_qa/opinion_qa_gender_{n}_options.csv', index=False, sep=';')