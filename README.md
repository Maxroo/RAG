# Retrieval-Augmented Generation (RAG) with Complex Claims

## Overview
This project focuses on **Retrieval-Augmented Generation (RAG)** with a focus on handling **complex claims**. The goal was to improve how large language models (LLMs) process and respond to intricate multi-layered information by using external data to support model responses.

This project was part of a research initiative at the **University of Victoria**, in collaboration with Professor Alex Thomo, Yichun Zhao and sara asghari

## Objective
The project aimed to enhance the performance of language models in the context of multi-layered claims by utilizing retrieval techniques. We built an automation program using **LlamaIndex** (formerly known as GPT Index) to retrieve relevant data from vectorized texts and improve the model's understanding and generation capabilities.

## Technologies Used
- **Python**: The primary programming language.
- **LlamaIndex**: For managing embeddings and performing efficient retrieval of relevant data.
- **Large Language Models (LLMs)**: Utilized to generate responses based on retrieved context.
- **Embeddings**: Text vectorization techniques for efficient search and matching of context.
- **ChromaDB**: A vector database used for efficient storage, retrieval, and management of embeddings. It facilitates fast search and retrieval of context relevant to the claims.
- **Hugging Face Transformers**: A library for working with pre-trained large language models (LLMs) such as GPT, BERT, and others, used to generate responses based on the retrieved context.

## Approach
1. **Data Collection**: We gathered a dataset that contained complex claims and multi-layered information.
2. **Embedding**: Text data was processed and vectorized using embedding techniques to allow for efficient retrieval.
3. **Retrieval**: The automation program was set up to search for relevant context using **LlamaIndex** and other external libraries.
4. **Generation**: We fed the retrieved data into a language model, which generated responses based on the context provided.
5. **Reranking**: A reranker was applied to improve the relevance of the retrieved context to the claims.
6. **Evaluation**: The system was evaluated on its ability to generate coherent, accurate, and contextually relevant responses.
