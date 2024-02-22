import utils
# from llama_index.response.notebook_utils import display_response    
from pathlib import Path
from llama_index.core import Document
import requests as rq
import json 
import pickle
import os

def get_indexed_files():
    try:
        with open('indexed_files.pickle', 'rb') as handle:
            indexed_files = pickle.load(handle)
        return indexed_files
    except IOError as e:
        print(f"Couldn't read to file indexed_files.pickle")
        return exit(1)
    
def save_indexed_files(indexed_files):
    with open('indexed_files.pickle', 'wb') as handle:
        pickle.dump(indexed_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def check_indexed_files(file_id, indexed_files):
    if file_id in indexed_files:
        return True
    return false

es = os.getenv("es")
index = os.getenv("index")

def construct_request(question):
    request = f"{es}/{index}/_search?pretty -H 'Content-Type: application/json' -d'"
    object = {
            "query": {
                "query_string" : {
                    "query": "",
                    "fields": ["text"]
                }
            }, 
            "size":2
            }
    object["query"]["query_string"]["query"] = question
    return request, object

request, test = construct_request("Gregg Rolie and Rob Tyner, are not a keyboardist.")
print(test)
response = rq.get(request, test)
print(response)





