import utils
# from llama_index.response.notebook_utils import display_response    
from pathlib import Path
from llama_index.core import Document
import requests as rq
import json 
import pickle
import os
import sys
import time

is_insert = False
mode = ''

def construct_request(question):
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
        "size": 2
    }
    payload["query"]["simple_query_string"]["query"] = question
    return request, headers, payload


# loop through the json file from response and get each field
def process_response(response):
    timer = time.time()
    json_response = response.json()
    if mode == '-f':
        with open("time.txt", "a") as log:
            log.write(f"elastic search took {time.time()-timer} seconds |", end = '')
    else: 
        print(f"elastic search took {time.time()-timer} seconds")
    index = None
    for hit in json_response['hits']['hits']:
        # Accessing individual fields in each hit
        source = hit['_source']
        # print("Document ID:", source.get('page_id', 'N/A'))
        # print("text:", source.get('text', 'N/A'))
        # print("\n\n\n\n")
        index = index_document(source.get('page_id', 'N/A'), source.get('text', 'N/A'))
    return index
        
def get_indexed_files():
    try:
        with open('indexed_files.pickle', 'rb') as handle:
            indexed_files = pickle.load(handle)
        return indexed_files
    except IOError as e:
        print(f"Couldn't read to file indexed_files.pickle")
        return None
    
def save_indexed_files(indexed_files):
    with open('indexed_files.pickle', 'wb') as handle:
        pickle.dump(indexed_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def check_indexed_files(file_id, indexed_files):
    if file_id in indexed_files:
        return True
    return false
        
def index_document(page_id, text):    
    index_file_set = get_indexed_files()    
    document = Document(text=text)
    Documents = [document] 
    
    if index_file_set == None:
        index_file_set = set()
    if page_id in index_file_set:
        index = utils.build_sentence_window_index(Documents)
    else:
        global is_insert 
        is_insert = True
        index = utils.build_sentence_window_index(Documents, insert = True)
        index_file_set.add(page_id)
        save_indexed_files(index_file_set)
    return index

def compare_response(result, expected):
    if 'true' in result.lower() and 'supports' in expected.lower():
        return True
    if 'false' in result.lower() and 'refutes' in expected.lower():
        return True
    return False

def process_response_no_index(response):
    json_response = response.json()
    texts = []
    for hit in json_response['hits']['hits']:
        source = hit['_source']
        texts.append(source.get('text', 'N/A'))
    res = utils.openai_query(question + " . Is the statement true or false?", texts)
    answer = res.choices[0].text
    token_usage = res.usage[0].total_tokens
    return answer, token_usage

def main():
    
    global is_insert 
    global mode
    
    start = time.time()
    question_count = 0
    correct = 0
    skiped = 0
    read_index_set_time = 0
    index_time = 0
    inference_time = 0
    if len(sys.argv) > 2:
        mode = sys.argv[1]
    ## -------------- test ----------------
    if(mode == '-q'):
        arg_question = sys.argv[2]
        if(arg_question == "test"):
            arg_question = "Gregg Rolie and Rob Tyner, are not a keyboardist."
        print(f"Question: {arg_question}")
        question = arg_question
        index = None
        expected = "supports"
        request, headers, payload = construct_request(question)
        response = rq.get(request, headers=headers, json=payload)   
        timer = time.time()
        index = process_response(response)
        print(f"Indexing took {time.time()-timer} seconds, Insert: {is_insert}" )
        if(index == None):
            #skip this question
            print(f"error: Index is None for question {question}")
            skiped += 1
            return 
        timer = time.time()
        engine = utils.get_sentence_window_query_engine(index)
        print(f"Getting engine took {time.time()-timer} seconds")
        timer = time.time()
        query_answer = engine.query(question + "Is the statement true or false?")
        print (f"Querying took {time.time()-timer} seconds")
        answer = query_answer.response
        if compare_response(answer, expected):
            correct += 1 
        print(f"Questions: {question} | Expected: {expected} | Answer: {answer}\n")
    ## -------------- end test ----------------
            
    #init log and result files to record
    #loop through the dev2hops.json file and get the question
    elif mode == '-f':
        with open("log.txt", "w"):
            pass
        with open("result.txt", "w"):
            pass
        file_path = sys.argv[2]
        with open(file_path, "r") as file:
            for statement in file:
                
                question_timer = time.time()
                question_count += 1
                print(statement["claim"])
                # question = statement['claim']
                # expected = statement['label']
                
                # request, headers, payload = construct_request(question)
                # response = rq.get(request, headers=headers, json=payload)   
                
                # timer = time.time()
                # answer, token_usage = process_response_no_index(response)
                # if compare_response(answer, expected):
                #     correct += 1        
                # with open("log.txt", "a") as log:
                #     log.write(f"Question: {question} | Expected: {expected} | Answer: {answer} | Took: {time.time() - timer}\n")
                # with open("result.txt", "w") as result:
                #     result.write(f"total question: {question_count} | corrects: {correct} | accuarcy {correct/question * 100}%\n took {time.time() - start}")
                #     result.write(f"\nToken_usage: {token_usage}\n")
    elif mode == '-m':
        arg_question = sys.argv[2]
        if(arg_question == "test"):
            arg_question = "Gregg Rolie and Rob Tyner, are not a keyboardist."
        print(f"Question: {arg_question}")
        question = arg_question
        index = None
        request, headers, payload = construct_request(question)
        response = rq.get(request, headers=headers, json=payload)   
        timer = time.time()
        json_response = response.json()
        texts = []
        for hit in json_response['hits']['hits']:
            source = hit['_source']
            texts.append(source.get('text', 'N/A'))
        timer = time.time()
        answer = utils.openai_query(question + " . Is the statement true or false?", texts)
        print (f"Querying took {time.time()-timer} seconds")
        print(f"Questions: {question} | Answer: {answer}\n")
        
    
    else:
        print("Invalid mode, Usage python3 main.py -q <question> or python3 main.py -f <file_path>")
        return        

if __name__ == "__main__":
    main()