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

def construct_request(question):
    request = "http://localhost:9200/enwiki/_search?pretty"
    
    headers = {
    "Content-Type": "application/json"
    }
    # Request payload
    payload = {
        "query": {
            "query_string": {
                "query": "",
                "fields": ["text"]
            }
        },
        "size": 3
    }
    payload["query"]["query_string"]["query"] = question
    return request, headers, payload

request, headers, payload = construct_request("Gregg Rolie and Rob Tyner, are not a keyboardist.")
response = rq.get(request, headers=headers, json=payload)


# loop through the json file from response and get each field
def read_resposne(response):
    json_response = response.json()
    for hit in json_response['hits']['hits']:
        # Accessing individual fields in each hit
        source = hit['_source']
        print("Document ID:", hit['_id'])
        print("opening_text:", source.get('opening_text', 'N/A'))
        # print(hit)
        print("\n\n\n\n")

read_resposne(response)


