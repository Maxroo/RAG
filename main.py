from pathlib import Path
import pickle
# import os
import json
import sys
import time
import requests as rq
from sklearn.metrics import classification_report
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import utils

CONFIG = None
LLM = None
mode = ''

def get_file_set(filename):
    try:
        with open(filename, 'rb') as handle:
            file = pickle.load(handle)
        return file
    except IOError as e:
        print(f"Couldn't read to file indexed_files.pickle")
        return None
    
def save_file_set(file, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def construct_request(question, size = 2):
    request = "http://localhost:9200/enwiki/_search?pretty"
    headers = {
    "Content-Type": "application/json"
    }
    # Request payload
    payload = {
        "query": {
            "simple_query_string" : {
                "query": "Gregg Rolie and Rob Tyner, are not a keyboardist.",
                "fields": ["title", "text"],
                "default_operator": "or"
            }
        },
        "size": size
    }
    payload["query"]["simple_query_string"]["query"] = question
    return request, headers, payload

def compare_response(result, expected):
    if 'true' in result.lower() and 'supports' in expected.lower():
        return True
    if 'false' in result.lower() and 'refutes' in expected.lower():
        return True
    return False

def get_response_no_index(question,response):
    json_response = response.json()
    texts = []
    for hit in json_response['hits']['hits']:
        source = hit['_source']
        texts.append(source.get('text', 'N/A'))
    res = utils.openai_query(question + " . Is the statement true or false?", texts)
    answer = res.choices[0].text
    token_usage = res.usage.total_tokens
    return answer, token_usage

def get_texts_from_response(response):
    json_response = response.json()
    texts = []
    for hit in json_response['hits']['hits']:
        source = hit['_source']
        texts.append(source.get('text', 'N/A'))
    return texts

def semintic_search(question, texts):
    # print(f"Question: {arg_question}")
    relevant_context = utils.retrieve_context_from_texts(texts, question)
    return relevant_context

def send_to_openai(question, texts):
    res = utils.openai_query(question + " . Is the statement true or false?", texts)
    answer = res.choices[0].text
    token_usage = res.usage.total_tokens
    return answer, token_usage

def send_to_together(question, texts):
    res = utils.togetherai_query(question + " . Is the statement true or false?", texts, llm = LLM)
    return res

def main():
    global CONFIG
    global LLM
    global mode
    
    with open('config.json', encoding='utf-8') as f:
        CONFIG = json.load(f)

    if CONFIG is None:
        print("Couldn't read config.json")
        exit()
        
    if CONFIG['global']['is_openAI']:
        LLM = OpenAI(model=CONFIG['global']['openAI'], temperature=0.1)
    else:
        LLM = TogetherLLM(model=CONFIG['global']['together'], temperature=0.1, api_key=CONFIG['api_keys']['TOGETHER_API_KEY'])

    if LLM is None:
        print("Couldn't initialize LLM")
        exit()
    
    token_used = 0
    start = time.time()
    question_count = 0
    correct = 0
    skiped = 0
    read_index_set_time = 0
    index_time = 0
    inference_time = 0
    if len(sys.argv) > 2:
        mode = sys.argv[1]
    print(f"Mode: {mode}")
    
    if mode == '-m':
        arg_question = sys.argv[2]
        if(arg_question == "test"):
            arg_question = "Gregg Rolie and Rob Tyner, are not a keyboardist."
        print(f"Question: {arg_question}")
        question = arg_question
        timer = time.time()
        request, headers, payload = construct_request(question)
        response = rq.get(request, headers=headers, json=payload, timeout=5)
        texts = get_texts_from_response(response)
        context = semintic_search(question, texts)
        semintic_search_time = time.time()-timer
        answer, token_usage = send_to_openai(question, context)
        print(f"token usage: {token_usage}")
        print (f"question took {time.time()-timer} seconds")
        print(f"Questions: {question} | Answer: {answer}\n")

    elif mode == '-s':
        print("Test Semintic search")
        arg_question = sys.argv[2]
        if(arg_question == "test"):
            arg_question = "Gregg Rolie and Rob Tyner, are not a keyboardist."
        print(f"Question: {arg_question}")
        question = arg_question
        request, headers, payload = construct_request(question, 3)
        response = rq.get(request, headers=headers, json=payload, timeout=5)
        timer = time.time()
        texts = get_texts_from_response(response)
        relevant_context = semintic_search(question, texts)
        print(f"Semintic search took {time.time()-timer} seconds")
        print(f"Relevant context:")
        for text in relevant_context:
            print(f"Text: {text}")

    elif mode == '-e':
        print("Test Elastic search")
        arg_question = sys.argv[2]
        if arg_question == "test":
            arg_question = "Gregg Rolie and Rob Tyner, are not a keyboardist."
        print(f"Question: {arg_question}")
        question = arg_question
        request, headers, payload = construct_request(question, 3)
        response = rq.get(request, headers=headers, json=payload, timeout=5)
        json_response = response.json()
        for hit in json_response['hits']['hits']:
            source = hit['_source']
            print("title:", source.get('title', 'N/A'))

    elif mode == '-v':
        print("Test Elastic search with vectorDB and llama_index")
        arg_question = sys.argv[2]
        if(arg_question == "test"):
            arg_question = "Gregg Rolie and Rob Tyner, are not a keyboardist."
        print(f"Question: {arg_question}")
        question = arg_question
        request, headers, payload = construct_request(question, 3)
        response = rq.get(request, headers=headers, json=payload, timeout=5)
        json_response = response.json()
        titles = []
        texts = []
        ids = []
        for hit in json_response['hits']['hits']:
            source = hit['_source']
            title = source.get('title', 'N/A')
            text = source.get('text', 'N/A')
            id = str(source.get('page_id', 'N/A'))
            titles.append(title)
            texts.append(text)
            ids.append(id)
        chroma_client, chroma_collection = utils.setup_chromadb("enwiki")
        utils.load_chromadb(chroma_collection, titles, texts, ids)
        index = utils.build_contexts_with_chromadb(chroma_collection)
        timer = time.time()
        engine = utils.get_sentence_window_query_engine(index)
        print(f"Getting engine took {time.time()-timer} seconds")
        timer = time.time()
        query_answer = engine.query(question + "Is the statement true or false?")
        print (f"Querying took {time.time()-timer} seconds")
        answer = query_answer.response
        print(f"Questions: {question} | Answer: {answer} | took: {time.time() - start}\n")

    #init log and result files to record
    #loop through the dev2hops.json file and get the question
    elif mode == '-f':
        with open("log.txt", "w"):
            pass
        file_path = sys.argv[2]
        with open(file_path, "r") as file:
            top_x = 16
            chunk_length = 256
            elastic_search_file_size = 8
            # maximum token openAI 3.5 can handle is 4096
            y_true = []
            y_pred = []
            file = json.load(file)
            for statement in file:
                question_timer = time.time()
                question = statement['claim']
                expected = statement['label']
                if 'supports' in expected.lower():
                    y_true.append(1)
                else:
                    y_true.append(0)
                request, headers, payload = construct_request(question, size = elastic_search_file_size)
                response = rq.get(request, headers=headers, json=payload)
                timer = time.time()
                # answer, token_usage = get_response_no_index(question ,response)
                texts = get_texts_from_response(response)
                context = semintic_search(question, texts)
                semintic_search_time = time.time()-timer
                timer = time.time()
                answer = None
                if CONFIG['global']['is_openAI']:
                    answer, token_usage = send_to_openai(question, context)
                else:
                    answer = send_to_together(question, context)
                    token_usage = 0 # not keep in track of token usage for togetherAI
                token_used += token_usage
                openai_time = time.time()-timer
                question_count += 1
                if compare_response(answer, expected):
                    correct += 1

                if 'true' in answer.lower():
                    y_pred.append(1)
                else :
                    y_pred.append(0)

                with open("log.txt", "a") as log:
                    log.write(f"Question: {question} | xpected: {expected} | Answer: {answer} | Token_usage: {token_usage} | Took: {time.time() - question_timer} |")
                    log.write(f"Semintic search took {semintic_search_time} seconds, OpenAI took {openai_time} seconds\n")
            with open("result.txt", "a") as result:
                result.write(f"\nCheap RAG file: {file_path} | top_x: {top_x} | chunk_length: {chunk_length} | elastic_search_file_size: {elastic_search_file_size}")
                result.write("\n------------------------------------------------------------------------------------------------------------------\n")
                result.write(f"model: {LLM.model} | Total question: {question_count} | corrects: {correct} | Accuracy: {correct/question_count * 100}% | took {time.time() - start}s | Total Token used: {token_used}\n")
                result.write(f"Classification report: \n{classification_report(y_true, y_pred)}")

    elif mode == '-vc':
        elastic_search_file_size = 8
        chunk_size = 512
        chunk_overlap = 70
        similarity_top_k = 6
        rerank_top_n = 3
        emd_model_llama = HuggingFaceEmbedding(model_name=utils.EMD_MODEL_NAME)
        chroma_client, chroma_collection = utils.setup_chromadb("enwiki-chunks")
        index = utils.get_vector_store_index(chroma_collection, emd_model_llama, llm = LLM)
        engine = utils.get_query_engine(index, similarity_top_k = similarity_top_k, rerank_top_n = rerank_top_n)
        file_set = get_file_set("file_set_vc.pickle")    
        if file_set == None:
            file_set = set()
        # file_set = set()
        with open("log-vc.txt", "w"):
            pass
        file_path = sys.argv[2]
        with open(file_path, "r") as file:
            # maximum token openAI 3.5 can handle is 4096
            y_true = []
            y_pred = []
            file = json.load(file)
            for statement in file:
                question_timer = time.time()
                question = statement['claim']
                expected = statement['label']

                if 'supports' in expected.lower():
                    y_true.append(1)
                else:
                    y_true.append(0)  

                request, headers, payload = construct_request(question, size = elastic_search_file_size)
                response = rq.get(request, headers=headers, json=payload)
                json_response = response.json()
                titles = []
                texts = []
                ids = []
                for hit in json_response['hits']['hits']:
                    source = hit['_source']
                    title = source.get('title', 'N/A')
                    text = source.get('text', 'N/A')
                    id = str(source.get('page_id', 'N/A'))
                    if id in file_set:
                        continue
                    file_set.add(id)
                    titles.append(title)
                    texts.append(text)
                    ids.append(id)
                 
                timer = time.time()
                if len(texts) != 0:
                    index_chunk = utils.parse_chunks_chromadb_return_index(texts, chroma_collection, emd_model_llama, chunk_size = chunk_size, chunk_overlap = chunk_overlap, llm = LLM)
                index_time = time.time()-timer
                
                timer = time.time()
                query_answer = engine.query(question + "Is the statement true or false?")
                query_time = time.time()-timer
                
                answer = query_answer.response
                if compare_response(answer, expected):
                    correct += 1 
                question_count += 1
                
                if 'true' in answer.lower():
                    y_pred.append(1)
                else :
                    y_pred.append(0)
                
                with open("log-vc.txt", "a") as log:
                    log.write(f"Question: {question} | Expected: {expected} | Answer: {answer} | Took: {time.time() - question_timer} |")
                    log.write(f"query_time took {query_time} seconds, index_time took {index_time} seconds\n")    
            
            with open("result-vc.txt", "a") as result:
                result.write(f"\n mode: chromaDB |  file: {file_path} | chunk_size: {chunk_size} | chunk_overlap: {chunk_overlap} | similarity top k: {similarity_top_k} | rerank_top_n : {rerank_top_n}  | elastic_search_file_size: {elastic_search_file_size}")
                result.write("\n------------------------------------------------------------------------------------------------------------------\n")
                result.write(f"model: {LLM.model} | Total question: {question_count} | corrects: {correct} | Accuracy: {correct/question_count * 100}% | took {time.time() - start}s\n")
                result.write(f"Classification report: \n{classification_report(y_true, y_pred, target_names=['refutes', 'supports'])}")
        save_file_set(file_set, "file_set_vc.pickle")

    elif mode == '-vn':
        elastic_search_file_size = 8
        sentence_window_size = 5
        similarity_top_k = 6
        rerank_top_n = 3
        emd_model_llama = HuggingFaceEmbedding(model_name=utils.EMD_MODEL_NAME)
        chroma_client, chroma_collection = utils.setup_chromadb("enwiki-nodes")
        index = utils.get_vector_store_index(chroma_collection, emd_model_llama, llm = LLM)
        engine = utils.get_sentence_window_query_engine(index, similarity_top_k = similarity_top_k, rerank_top_n = rerank_top_n)
        file_set = get_file_set("file_set_vn.pickle")    
        if file_set == None:
            file_set = set()
        # file_set = set()
        with open("log-vn.txt", "w"):
            pass
        file_path = sys.argv[2]
        with open(file_path, "r") as file:
            # maximum token openAI 3.5 can handle is 4096
            y_true = []
            y_pred = []
            file = json.load(file)
            for statement in file:
                question_timer = time.time()
                question = statement['claim']
                expected = statement['label']
                
                if 'supports' in expected.lower():
                    y_true.append(1)
                else:
                    y_true.append(0)  
                    
                request, headers, payload = construct_request(question, size = elastic_search_file_size)
                response = rq.get(request, headers=headers, json=payload, timeout=5)   
                json_response = response.json()
                titles = []
                texts = []
                ids = []
                for hit in json_response['hits']['hits']:
                    source = hit['_source']
                    title = source.get('title', 'N/A')
                    text = source.get('text', 'N/A')
                    id = str(source.get('page_id', 'N/A'))
                    if id in file_set:
                        continue
                    file_set.add(id)
                    titles.append(title)
                    texts.append(text)
                    ids.append(id)
                timer = time.time()
                if len(texts) != 0:
                    index_nodes = utils.parse_nodes_chromadb_return_index(texts, chroma_collection, emd_model_llama, sentence_window_size, llm = LLM)
                index_time = time.time()-timer
                
                timer = time.time()
                query_answer = engine.query(question + "Is the statement true or false?")
                query_time = time.time()-timer
                
                answer = query_answer.response
                if compare_response(answer, expected):
                    correct += 1 
                question_count += 1
                
                if 'true' in answer.lower():
                    y_pred.append(1)
                else :
                    y_pred.append(0)
                
                with open("log-vn.txt", "a") as log:
                    log.write(f"Question: {question} | Expected: {expected} | Answer: {answer} | Took: {time.time() - question_timer} |")
                    log.write(f"query_time took {query_time} seconds, index_time took {index_time} seconds")    
            
            with open("result-vn.txt", "a") as result:
                result.write(f"\n mode: chromaDB |  file: {file_path} | sentence window size: {sentence_window_size} | similarity top k: {similarity_top_k} | rerank_top_n : {rerank_top_n}  | elastic_search_file_size: {elastic_search_file_size}")
                result.write(f"\n------------------------------------------------------------------------------------------------------------------\n")
                result.write(f"model: {LLM.model} | Total question: {question_count} | corrects: {correct} | Accuracy: {correct/question_count * 100}% | took {time.time() - start}s\n")
                result.write(f"Classification report: \n{classification_report(y_true, y_pred, target_names=['refutes', 'supports'])}")
        save_file_set(file_set, "file_set_vn.pickle")

    elif mode == '-vh':
        elastic_search_file_size = 8
        chunk_size = [2048, 512, 128]
        similarity_top_k = 6
        rerank_top_n = 3
        emd_model_llama = HuggingFaceEmbedding(model_name=utils.EMD_MODEL_NAME)
        # chroma_client, chroma_collection = utils.setup_chromadb("enwiki-hierarchy")
        # index = utils.get_vector_store_index(chroma_collection, emd_model_llama, llm = LLM)
        # engine = utils.get_hierarchy_node_query_engine(index, similarity_top_k = similarity_top_k, rerank_top_n = rerank_top_n)
        # file_set = get_file_set("file_set_vh.pickle")    
        # if file_set == None:
        #     file_set = set()
        # file_set = set()
        with open("log-vh.txt", "w"):
            pass
        file_path = sys.argv[2]
        with open(file_path, "r") as file:
            # maximum token openAI 3.5 can handle is 4096
            y_true = []
            y_pred = []
            file = json.load(file)
            for statement in file:
                question_timer = time.time()
                question = statement['claim']
                expected = statement['label']
                
                if 'supports' in expected.lower():
                    y_true.append(1)
                else:
                    y_true.append(0)  
                    
                request, headers, payload = construct_request(question, size = elastic_search_file_size)
                response = rq.get(request, headers=headers, json=payload, timeout=5)
                json_response = response.json()
                titles = []
                texts = []
                ids = []
                for hit in json_response['hits']['hits']:
                    source = hit['_source']
                    title = source.get('title', 'N/A')
                    text = source.get('text', 'N/A')
                    id = str(source.get('page_id', 'N/A'))
                    # if id in file_set:
                    #     continue
                    # file_set.add(id)
                    titles.append(title)
                    texts.append(text)
                    ids.append(id)
                 
                timer = time.time()
                if len(texts) != 0:
                    index_nodes = utils.parse_hierarchy_nodes_chromadb_return_index(texts, None, emd_model_llama, chunk_size = chunk_size, llm = LLM)
                    engine = utils.get_hierarchy_node_query_engine(index_nodes, similarity_top_k = similarity_top_k, rerank_top_n = rerank_top_n)
                index_time = time.time()-timer
                
                timer = time.time()
                query_answer = engine.query(question + "Is the statement true or false?")
                query_time = time.time()-timer
                
                answer = query_answer.response
                if compare_response(answer, expected):
                    correct += 1 
                question_count += 1
                
                if 'true' in answer.lower():
                    y_pred.append(1)
                else :
                    y_pred.append(0)
                
                with open("log-vh.txt", "a") as log:
                    log.write(f"Question: {question} | Expected: {expected} | Answer: {answer} | Took: {time.time() - question_timer} |")
                    log.write(f"query_time took {query_time} seconds, index_time took {index_time} seconds\n")    
            
            with open("result-vh.txt", "a") as result:
                result.write(f"\n mode: llamaindex, hierarchy node |  file: {file_path} | chunk_size: {chunk_size} | similarity top k: {similarity_top_k} | rerank_top_n : {rerank_top_n}  | elastic_search_file_size: {elastic_search_file_size}")
                result.write(f"\n------------------------------------------------------------------------------------------------------------------\n")
                result.write(f"model: {LLM.model} | Total question: {question_count} | corrects: {correct} | Accuracy: {correct/question_count * 100}% | took {time.time() - start}s\n")
                result.write(f"Classification report: \n{classification_report(y_true, y_pred)}")

    else:
        print("Invalid mode, Usage python3 main.py -q <question> or python3 main.py -f <file_path>")
        return        

if __name__ == "__main__":
    main()