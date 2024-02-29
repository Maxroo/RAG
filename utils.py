import os
from dotenv import load_dotenv, find_dotenv
import openai
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.llms.openai import OpenAI

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
        model="gpt-3.5-turbo",
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
    embed_model="local:BAAI/bge-small-en-v1.5", save_dir="./index/sentence_index_default", insert=False
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
