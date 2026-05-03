import pandas as pd


df_responses = pd.read_csv("data/response_collection/bbq_likert/bbq_gemma12b_likert_5_neg.csv", sep=',')
df_bias_info = pd.read_csv("data/bbq/bbq_gender.csv", sep=';')


df_responses[['question_index', 'question_polarity', 'context_condition']] = df_responses['question_id'].str.split('_', expand=True)
cols_to_keep = [
    'example_id', 'question_id', 'question_polarity', 'context_condition', 'entity', 'entity_info', 'sample_id', 'ground_truth', 'likert_rating'
    ]
df = df_responses[cols_to_keep]


df_disambig = df[df['context_condition'] == 'disambig']
df_ambig = df[df['context_condition'] == 'ambig']


df_disambig = df_disambig.drop(columns=['context_condition'])
df_ambig = df_ambig.drop(columns=['context_condition'])


stereo_map = dict(zip(df_bias_info['example_id'], df_bias_info['additional_metadata/stereotyped_groups/0']))


def add_columns(subset):
    subset = subset.copy()
    subset['stereotyped_group'] = subset['example_id'].map(stereo_map)
    return subset


df_disambig = add_columns(df_disambig)
df_ambig = add_columns(df_ambig)


df_disambig.to_csv("data/bbq_analysis/bbq_likert/bbq_gemma12b_likert_5_neg_responses_labelled_disambig.csv", index=False)
df_ambig.to_csv("data/bbq_analysis/bbq_likert/bbq_gemma12b_likert_5_neg_responses_labelled_ambig.csv", index=False)