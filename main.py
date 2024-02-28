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
    json_response = response.json()
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

def main():
    start = time.time()
    question_count = 0
    correct = 0
    skiped = 0
    read_index_set_time = 0
    index_time = 0
    inference_time = 0
    open('time.txt', 'w').close()
    open('log.txt', 'w').close()
    # if len(sys.argv) > 1:
    #     option = sys.argv[1]
    #     if(option == "new"):
    #         open('log.txt', 'w').close()
    ## -------------- test ----------------
    
    # if(arg_question == "test"):
    #     arg_question = "Gregg Rolie and Rob Tyner, are not a keyboardist."
    
    # print(f"Question: {arg_question}")
    
    # question = arg_question
    # index = None
    # expected = "supports"
    # request, headers, payload = construct_request(question)
    # response = rq.get(request, headers=headers, json=payload)   
    # index = process_response(response)
    # if(index == None):
    #     #skip this question
    #     log.write(f"error: Index is None for question {question}")
    #     skiped += 1
    # engine = utils.get_sentence_window_query_engine(index)
    # query_answer = engine.query(question + "Is the statement true or false?")
    # answer = query_answer.response
    # if compare_response(answer, expected):
    #     correct += 1 
    # log.write(f"Questions: {question} | Expected: {expected} | Answer: {answer}\n")
    ## -------------- end test ----------------
            
    #init log and result files to record
    #loop through the dev2hops.json file and get the question
    dev2hops = open("dev2hops.json", "r")
    dev2hops_json = json.load(dev2hops)
    for statement in dev2hops_json:
        is_insert = False
        question_timer = time.time()
        question_count += 1
        index = None
        question = statement['claim']
        expected = statement['label']
        request, headers, payload = construct_request(question)
        response = rq.get(request, headers=headers, json=payload)   
        
        timer = time.time()
        index = process_response(response)
        with open("time.txt", "a") as log:
            log.write(f"Indexing took {time.time()-timer} seconds, Insert = {is_insert}\n")
        if(index == None):
            #skip this question
            log = open("log.txt", "a")
            log.write(f"Index is None for question {question}")
            log.close()
            skiped += 1
            continue
        timer = time.time()
        engine = utils.get_sentence_window_query_engine(index)
        with open("time.txt", "a") as log:
            log.write(f"Getting engine took {time.time()-timer} seconds\n")
        
        timer = time.time()
        query_answer = engine.query(question + "Is the statement true or false?")
        with open("timer.txt", "a") as log:
            log.write(f"Querying took {time.time()-timer} seconds\n")
            log.write(f"Total time for question {question} is {time.time()-question_timer} seconds\n")
            log.write(f"-----------------------------------------------------------")
        
        answer = query_answer.response
        if compare_response(answer, expected):
            correct += 1

        log = open("log.txt", "a") 
        log.write(f"Questions: {question} | Expected: {expected} | Answer: {answer} | Took: {time.time()- question_timer} seconds \n")
        log.close()
        result = open("result.txt", "w")
        result.write(f"Total Questions: {question_count}\nSkiped: {skiped}\nCorrect: {correct}\nAccuracy: {correct/question_count * 100}%\n")
        result.write('Took', time.time()-start, 'seconds.')
        result.close()

if __name__ == "__main__":
    main()