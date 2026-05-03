import pandas as pd

df = pd.read_csv("data/open_annotation/opinion_qa_mistral7b_open_responses_annotation.csv")

print(df[df['annotation_status'] == 'unmapped'])

df_copy = df.copy()
df_copy.loc[df['annotation_status'] == 'unmapped', 'annotated_label'] = 'Men and women are basically similar'
print(df_copy[df_copy['annotation_status'] == 'unmapped'])

df_copy.drop(columns=['raw_annotation_output', 'annotation_status']).to_csv(
    'data/open_annotation/opinion_qa_mistral7b_open_responses_annotation_edit.csv', 
    index=False
    )