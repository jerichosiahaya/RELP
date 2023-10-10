# ✨ RELP (RAG Enhancement for LLM-based Prediction)

## Project Description and Motivation
### The RELP approach

Language Model is a powerful tool, particularly for text classification. However, they may not always produce optimal results when lacking sufficient contextual information.

### Artificial Context-Dependent Intelligence
One of the limitations of Language Models are they may not perform well when it doesn't have enough context. This means if we don't give it enough information or background, it might give us incorrect or not-so-good answers. Imagine trying to understand a story without knowing the beginning or the middle—it's tough! Context helps the AI understand and give us better, more accurate results. So, giving the AI enough information or context is really important for getting the best outcomes.

To address this challenge, this project proposes alternative approaches utilizing RAG enhancement. Both methods mutually enhance each other. RAG provides contextual similarity that can function as a knowledge base using few-shot learning, while LLM offers contextual predictions based on the acquired knowledge.

## Model Validation
The model uses two contextual text embeddings: IndoBERT and OpenAI's ADA-002, with 3-shot learning for contextual prediction using GPT-3.5-Turbo version. Validation was conducted on 50 testing data points, resulting in:

|  | f1-score | accuracy|
|--|--|--|
| IndoBERT (3-shot) | 0.859 | 0.860
| ADA-002 (3-shot) | 0.816 | 0.820

