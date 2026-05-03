import pandas as pd


df_responses = pd.read_csv("data/response_collection/bbq_closed-ended/bbq_gemma12b_closed-ended_responses.csv", sep=',')
df_label_info = pd.read_csv("data/bbq/bbq_gender_closed-ended.csv", sep=';')
df_bias_info = pd.read_csv("data/bbq/bbq_gender.csv", sep=';')


df_responses[['question_index', 'question_polarity', 'context_condition']] = df_responses['question_id'].str.split('_', expand=True)
cols_to_keep = ['example_id', 'question_id', 'question_polarity', 'context_condition', 'answer_order', 'sample_id', 'answers', 'ground_truth', 'chosen_answer']
df = df_responses[cols_to_keep]


df_disambig = df[df['context_condition'] == 'disambig']
df_ambig = df[df['context_condition'] == 'ambig']


df_disambig = df_disambig.drop(columns=['context_condition'])
df_ambig = df_ambig.drop(columns=['context_condition'])


def build_label_map(df_label_info):
    label_map = {}
    for _, row in df_label_info.iterrows():
        example_id = row['example_id']
        mapping = {
            row['ans0']: row['answer_info/ans0'],
            row['ans1']: row['answer_info/ans1'],
            row['ans2']: row['answer_info/ans2'],
        }
        label_map[example_id] = mapping
    return label_map


label_map = build_label_map(df_label_info)

stereo_map = dict(zip(df_bias_info['example_id'], df_bias_info['additional_metadata/stereotyped_groups/0']))


def add_columns(subset):
    subset = subset.copy()
    subset['chosen_answer_label'] = subset.apply(lambda x: label_map.get(x['example_id'], {}).get(x['chosen_answer']), axis=1)
    subset['stereotyped_group'] = subset['example_id'].map(stereo_map)
    return subset


df_disambig = add_columns(df_disambig)
df_ambig = add_columns(df_ambig)


df_disambig.to_csv("data/bbq_analysis/bbq_closed-ended/bbq_gemma12b_closed-ended_responses_labelled_disambig.csv", index=False)
df_ambig.to_csv("data/bbq_analysis/bbq_closed-ended/bbq_gemma12b_closed-ended_responses_labelled_ambig.csv", index=False)