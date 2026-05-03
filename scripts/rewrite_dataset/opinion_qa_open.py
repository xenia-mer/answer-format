import pandas as pd
import re

def opinion_open_extract(file_path):
    df = pd.read_csv(file_path, sep=';')
    questions_dict = {}
    
    for index, row in df.iterrows():
        key = row['key']
        full_question = row['likert_question']
        pattern = r'(.+?)(?=To what extent do you agree with the following statement|$)'
        match = re.search(pattern, full_question)
        
        if match:
            question_part = match.group(1).strip()
            question_part = re.sub(r'[:.]\s*$', '', question_part)
            
            if key not in questions_dict:
                questions_dict[key] = []
            
            if question_part not in questions_dict[key]:
                questions_dict[key].append(question_part)
    
    return questions_dict

def print_qs(questions_dict):
    for key, questions in questions_dict.items():
        print(f"\nKey: {key}")
        if len(questions) == 1:
            print(f"  Question: {questions[0]}")
        else:
            print(f"  Found {len(questions)} versions:")
            for i, question in enumerate(questions, 1):
                print(f"    {i}. {question}")

if __name__ == "__main__":
    file_path = "data/opinion_qa/opinion_qa_likert/opinion_qa_gender_likert_4_neg.csv"
    questions_dict = opinion_open_extract(file_path)
    print_qs(questions_dict)
    
    output_data = []
    for key, questions in questions_dict.items():
        for question in questions:
            output_data.append({'key': key, 'question': question})
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv('data/opinion_qa/opinion_qa_open.csv', index=False, sep=';')