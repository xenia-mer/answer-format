# Effects of Answer Format Variation on Gender Bias in Large Language Models

As Large Language Models (LLMs) are extensively deployed in sensitive applications, ranging from private chatbots to policy regulations, the issue of bias in their responses, particularly gender bias, becomes increasingly important. Existing studies commonly rely on single-format evaluations, despite established evidence from the social sciences showing that the format of a question can substantially influence how humans respond.

This project investigates how answer format variation affects the measurement and manifestation of gender bias in LLMs. The analysis employs parallel versions of questions derived from two datasets, an NLP benchmark (BBQ, [Parrish et al., 2022](https://aclanthology.org/2022.findings-acl.165/)) and a set derived from U.S. public opinion surveys (OpinionQA, [Santukar et al., 2023](https://github.com/tatsu-lab/opinions_qa/tree/main)), with systematically varied response formats across multiple models. This approach addresses an important gap: whilst real-world LLM applications involve diverse interaction formats, our understanding of bias has been largely developed through single-format evaluations.

```
├── ./
│   ├── data/
│   │   ├── bbq/
│   │   │   ├── bbq_likert/
│   │   ├── bbq_analysis/
│   │   │   ├── bbq_closed-ended/
│   │   │   ├── bbq_likert/
│   │   │   ├── bbq_open/
│   │   ├── open_annotation/
│   │   ├── opinion_qa/
│   │   │   ├── opinion_qa_closed-ended/
│   │   │   ├── opinion_qa_likert/
│   │   ├── response_collection/
│   │   │   ├── bbq_closed-ended/
│   │   │   ├── bbq_likert/
│   │   │   ├── bbq_open/
│   │   │   ├── opinion_qa_closed-ended/
│   │   │   ├── opinion_qa_likert/
│   │   │   ├── opinion_qa_open/
│   ├── envs/
│   ├── scripts/
│   │   ├── analysis/
│   │   │   ├── bbq_likert_labelling/
│   │   ├── open_annotation/
│   │   ├── response_collection/
│   │   │   ├── bbq_closed-ended/
│   │   │   ├── bbq_likert/
│   │   │   │   ├── bbq_mistral/
│   │   │   │   ├── bbq_llama/
│   │   │   │   ├── bbq_gemma/
│   │   │   ├── bbq_open/
│   │   │   ├── opinion_qa_closed-ended/
│   │   │   ├── opinion_qa_likert/
│   │   │   │   ├── opinion_qa_gemma/
│   │   │   │   ├── opinion_qa_llama/
│   │   │   │   ├── opinion_qa_mistral/
│   │   │   ├── opinion_qa_open/
│   │   ├── rewrite_dataset/
```
