import os
from dotenv import load_dotenv, find_dotenv
import openai
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.llms.openai import OpenAI
from FlagEmbedding import FlagReranker

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb.utils.embedding_functions as embedding_functions
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import pandas as pd

EMD_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def setup_chromadb(db_name="enwiki", emd_model_name=EMD_MODEL_NAME):
    chroma_client = chromadb.PersistentClient(path='/home/thomo/yichun/RAG/chromadb')
    emb_model  = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emd_model_name)
    chroma_collection = chroma_client.get_or_create_collection(name=db_name,
                                                                     metadata={"hnsw:space": "cosine"}, # l2 is the default
                                                                     embedding_function=emb_model)
    return chroma_client, chroma_collection

def load_chromadb(chroma_collection, titles, texts, ids):
    # chroma_collection = setup_chromadb(db_name)
    title_list = [{"title": title} for title in titles]
    chroma_collection.upsert( # instead of add
            documents = texts,
            metadatas = title_list, # pages.title.apply(lambda title: {"title": title}).tolist(),
            ids = ids# pages.index.map(str).tolist()
        )
    
def build_contexts_with_chromadb(chroma_collection, emd_model_name=EMD_MODEL_NAME, window_size = 3 ,llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1)):
    emd_model_llama = HuggingFaceEmbedding(model_name=emd_model_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # service_context = ServiceContext.from_defaults(embed_model=emd_model_llama)

    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=emd_model_llama,
        node_parser=node_parser,
    )
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,
    )

    return index

def query_with_chromadb(index, question, top_x=6):
    query_engine = index.as_query_engine()
    results = query_engine.query(question, top_k=top_x)



def get_openai_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")

openai.api_key = get_openai_api_key()
openai_client = openai.OpenAI(api_key=get_openai_api_key())

def construct_prompt(question, documents):
    prompt_start = (
    "Answer the question based on the context below.\n\n"+
    "Context:\n"
    )

    prompt_end = (
        f"\n\nQuestion: {question}\nAnswer:"
    )

    prompt = (
        prompt_start + "\n\n---\n\n".join(documents) + 
        prompt_end
    )
    return prompt

def openai_query(question, documents):
    prompt = construct_prompt(question, documents)
    res = openai_client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0,
        max_tokens=636,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    # print(f"Prompt: {prompt}") # for debugging
    # print(f"Response: {res}") # for debugging
    return res

def build_sentence_window_index(
    documents, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1), 
    embed_model="BAAI/bge-small-en-v1.5", save_dir="./index/sentence_index_default", insert=False
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )
        if(insert):
            for doc in documents:
                sentence_index.insert(doc, service_context=sentence_context)            
            sentence_index.storage_context.persist(persist_dir=save_dir)

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np
from FlagEmbedding import FlagReranker
import re 
from split_string import split_string_with_limit
import tiktoken

def custom_sent_tokenize(text, max_token_length=256):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    token_length = 0
    result = []
    current_sentence = ""
    for sentence in sentences:
        if token_length + len(sentence.split()) <= max_token_length:
            current_sentence += sentence
            token_length += len(sentence.split())
        else:
            result.append(current_sentence.strip())
            current_sentence = sentence
            token_length = len(sentence.split())
    if current_sentence:
        result.append(current_sentence.strip())
    return result

def retrieve_context_from_texts(texts, question, top_x = 6, chunk_length=256):
    # Tokenize question and texts into sentences
    question_sentences = sent_tokenize(question)
    # text_sentences = [custom_sent_tokenize(text, ) for text in texts]
    text_sentences = [chunked_tokens(text,"cl100k_base", 256) for text in texts]
    
    # Flatten list of text sentences
    flat_text_sentences = [sentence for sublist in text_sentences for sentence in sublist]
    # Compute TF-IDF vectors for question and text sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(question_sentences + flat_text_sentences)

    # Compute cosine similarity between question and text sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # Sort sentences by similarity to question
    num_question_sentences = len(question_sentences)
    sorted_indices = np.argsort(similarity_matrix[:num_question_sentences, num_question_sentences:])[0][::-1]

    # Retrieve top-ranked sentences
    if len(sorted_indices) < top_x:
        top_x = len(sorted_indices)
    relevant_context = [flat_text_sentences[i] for i in sorted_indices[:top_x]] 
    return relevant_context
    

def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    texts = split_string_with_limit(text, chunk_length, encoding)
    return texts
