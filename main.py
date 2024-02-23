import utils
# from llama_index.response.notebook_utils import display_response    
from pathlib import Path
from llama_index.core import Document
import requests as rq
import json 
import pickle
import os

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
        print("Document ID:", source.get('page_id', 'N/A'))
        # print("text:", source.get('text', 'N/A'))
        # # print(hit)
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
        return index
    else:
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
    question_count = 0
    correct = 0
    skiped = 0
    test_quest = "Gregg Rolie and Rob Tyner, are not a keyboardist."
    # init log and result files to record
    log = open("log.txt", "w")
    result = open("result.txt", "w")
    # loop through the dev2hops.json file and get the question
    dev2hops = open("dev2hops.json", "r")
    dev2hops_json = json.load(dev2hops)
    print(len(dev2hops_json))
    question = test_quest
    index = None
    expected = "supports"
    request, headers, payload = construct_request(question)
    response = rq.get(request, headers=headers, json=payload)   
    index = process_response(response)
    if(index == None):
        #skip this question
        log.write(f"Index is None for question {question}")
        skiped += 1
    engine = utils.get_sentence_window_query_engine(index)
    query_answer = engine.query(question + "For this statement give me a true or false answer. and why?")
    answer = query_answer.response
    if compare_response(answer, expected):
        correct += 1 
    log.write(f"Questions: {question}\nExpected: {expected}\nAnswer: {answer}\n")
    # for statement in dev2hops_json:
    #     index = None
    #     question = statement['claim']
    #     expected = statement['label']
    #     request, headers, payload = construct_request(question)
    #     response = rq.get(request, headers=headers, json=payload)   
    #     index = process_response(response)
    #     if(index == None):
    #         #skip this question
    #         log.write(f"Index is None for question {question}")
    #         skiped += 1
    #         continue
    #     engine = get_sentence_window_query_engine(index)
    #     answer = engine.query(question + "For this statement give me a true or false answer.")
    #     if compare_response(answer, expected):
    #         correct += 1 
    #     #log.write(f"Questions: {question}\nExpected: {expected}\nAnswer: {answer}\n")
    # result.write(f"Total Questions: {question_count}\nSkiped: {skiped}\nCorrect: {correct}\nAccuracy: {correct/question_count * 100}%\n")
    log.close()
    result.close()

if __name__ == "__main__":
    main()