import pandas as pd
import os

opinionqa = pd.read_csv("data/opinion_qa/opinion_qa_gender.csv", sep=';')

base_path = "santukar_opinionqa_qs"

dfs = []

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    
    if os.path.isdir(folder_path):
        file_path = os.path.join(folder_path, "info.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=["key", "option_ordinal"])
            dfs.append(df)

df_info = pd.concat(dfs, ignore_index=True)

opinionqa = opinionqa.merge(df_info, on="key", how="left")

opinionqa.to_csv("data/opinion_qa/opinion_qa_gender_ordinal.csv", index=False)