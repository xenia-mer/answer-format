import pandas as pd
import ast
from itertools import permutations
from math import factorial

for n in range(3, 7):
    df = pd.read_csv(f'data/opinion_qa/opinion_qa_closed-ended/opinion_qa_gender_{n}_options.csv', sep=';')
    df['options'] = df['options'].apply(ast.literal_eval)
    num_perms = factorial(n)
    permuted_columns = {}

    for idx, row in df.iterrows():
        opts = row['options']
        perms = list(permutations(opts))
        perms = [list(p) for p in perms]
        permuted_columns[idx] = perms

    for i in range(num_perms):
        df[f'options{i}'] = df.index.map(lambda idx: permuted_columns[idx][i])

    df = df.drop(columns=['options', 'num_options'])
    
    df.to_csv(f'data/opinion_qa/opinion_qa_closed-ended/opinion_qa_gender_{n}_options_permutations.csv', index=False, sep=';')
    print(f"Saved dataset with all {num_perms} permutations for {n} options")