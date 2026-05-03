import pandas as pd

df = pd.read_csv("data/opinion_qa/opinion_qa_likert/opinion_qa_gender_likert_4_neg.csv", sep=';')
dfid = pd.read_csv("data/opinion_qa/opinion_qa_gender.csv", sep=';')

id_mapping = dict(zip(dfid['key'], dfid['id']))
df['id'] = df['key'].map(id_mapping)
cols = ['id'] + [col for col in df.columns if col != 'id']
df = df[cols]

df.to_csv("data/opinion_qa/opinion_qa_likert/opinion_qa_gender_likert_4_neg.csv", sep=';', index=False)
print(df.head())