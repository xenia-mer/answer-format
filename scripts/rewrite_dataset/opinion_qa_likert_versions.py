import pandas as pd
import re

def opinion_qa_likert_versions(input_file):
    df = pd.read_csv(input_file, sep=';')
    likert_configs = {
        2: {
            'pos': (
                'Please rate your agreement from 1 (strongly disagree) to 2 (strongly agree).',
                '1=Strongly disagree, 2=Strongly agree'
            ),
            'neg': (
                'Please rate your agreement from 1 (strongly agree) to 2 (strongly disagree).',
                '1=Strongly agree, 2=Strongly disagree'
            )
        },
        4: {
            'pos': (
                'Please rate your agreement from 1 (strongly disagree) to 4 (strongly agree).',
                '1=Strongly disagree, 2=Disagree, 3=Agree, 4=Strongly agree'
            ),
            'neg': (
                'Please rate your agreement from 1 (strongly agree) to 4 (strongly disagree).',
                '1=Strongly agree, 2=Agree, 3=Disagree, 4=Strongly disagree'
            )
        },
        5: {
            'pos': (
                'Please rate your agreement from 1 (strongly disagree) to 5 (strongly agree).',
                '1=Strongly disagree, 2=Disagree, 3=Neither agree nor disagree, 4=Agree, 5=Strongly agree'
            ),
            'neg': (
                'Please rate your agreement from 1 (strongly agree) to 5 (strongly disagree).',
                '1=Strongly agree, 2=Agree, 3=Neither agree nor disagree, 4=Disagree, 5=Strongly disagree'
            )
        },
        9: {
            'pos': (
                'Please rate your agreement from 1 (strongly disagree) to 9 (strongly agree).',
                '1=Strongly disagree, 2, 3, 4, 5, 6, 7, 8, 9=Strongly agree'
            ),
            'neg': (
                'Please rate your agreement from 1 (strongly agree) to 9 (strongly disagree).',
                '1=Strongly agree, 2, 3, 4, 5, 6, 7, 8, 9=Strongly disagree'
            )
        },
        10: {
            'pos': (
                'Please rate your agreement from 1 (strongly disagree) to 10 (strongly agree).',
                '1=Strongly disagree, 2, 3, 4, 5, 6, 7, 8, 9, 10=Strongly agree'
            ),
            'neg': (
                'Please rate your agreement from 1 (strongly agree) to 10 (strongly disagree).',
                '1=Strongly agree, 2, 3, 4, 5, 6, 7, 8, 9, 10=Strongly disagree'
            )
        }
    }

    likert_pattern = re.compile(r'^(.*?)\s+Please rate your agreement from.*$')

    for points, polarities in likert_configs.items():
        for polarity, (question_suffix, scale) in polarities.items():
            new_df = df.copy()

            def update_question(question):
                match = likert_pattern.search(question)
                if match:
                    statement = match.group(1)
                    return f"{statement} {question_suffix}"
                return f"{question} {question_suffix}"

            new_df['likert_question'] = new_df['likert_question'].apply(update_question)
            new_df['scale'] = scale
            output_file = (f"data/opinion_qa/opinion_qa_likert/opinion_qa_gender_likert_{points}_{polarity}.csv")
            new_df.to_csv(output_file, index=False, sep=';')
            print(f"Created: {output_file}")
    print("\nAll files created.")

if __name__ == "__main__":
    input_file = "data/opinion_qa/opinion_qa_likert/opinion_qa_gender_likert_4_neg.csv"
    opinion_qa_likert_versions(input_file)