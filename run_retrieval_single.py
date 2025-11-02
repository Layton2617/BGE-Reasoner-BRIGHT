import numpy as np
import yaml
import diskannpy
import time
import os
import csv
import struct

import pickle
import pandas as pd
import heapq
import bisect
import pytrec_eval

import math

from beir.datasets.data_loader import GenericDataLoader

import logging
import pathlib, os
import sys

from tqdm import tqdm
import random

import json

from datetime import datetime

import copy
from copy import deepcopy
import multiprocessing
from multiprocessing import Pool
import sys
import re

import argparse

from pathlib import Path

from openai import OpenAI
from google import genai
from google.genai import types

import matplotlib.pyplot as plt

from datasets import load_dataset

import faiss

llm_model_name = "gemini-2.0-flash"

if "gpt" in llm_model_name:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
elif "gemini" in llm_model_name:
    # genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str, default="BEIR", help="which benchmark data format to use")
    parser.add_argument("--dataset_name", type=str, default=None, help="which data set to use")
    parser.add_argument("--corpus", type=str, default=None, help="which corpus to use")
    parser.add_argument("--data_type", type=str, default="fp32", help="which data set to use")
    parser.add_argument("--algo_name", type=str, default="diskann", help="which algorithm to use")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the cheap biencoder name")
    parser.add_argument("--expensive_model_name", type=str, default=None, help="Name of the expensive biencoder name")
    parser.add_argument("--graph_degree", type=int, default=-1, help="what max degree to use when building the index")
    parser.add_argument("--time_tag", type=str, default="", help="time tag")
    return parser.parse_args()

args = parse_arguments()

dataset_name=args.dataset_name

benchmark_name=args.benchmark_name

data_type=args.data_type

corpus_type=args.corpus

graph_degree = args.graph_degree
algo_name=args.algo_name

print(dataset_name)

split="test"

if benchmark_name=="BRIGHT":
    data = load_dataset('xlangai/BRIGHT', 'examples')[dataset_name]
    documents = load_dataset('xlangai/BRIGHT', 'documents')[dataset_name]

    queries={}
    qrels={}
    excluded_ids={}
    for query in data:
        queries[query["id"]]=query["query"]
        excluded_ids[query["id"]]=query["excluded_ids"]
        answer={}
        for entry in query["gold_ids"]:
            answer[entry]=1
        qrels[query["id"]]=answer

    corpus={}
    for doc in documents:
        corpus[doc["id"]]={"title":"","text":doc["content"]}
    
if graph_degree==-1:
    if len(corpus)>1000000:
        graph_degree=32
    elif len(corpus)>100000:
        graph_degree=16
    elif len(corpus)>10000:
        graph_degree=12
    else:
        graph_degree=8

print("corpus len", len(corpus))
print("query len", len(queries))
print("qrel len", len(qrels))

log_path=f"./experiment_log/{benchmark_name}_{args.model_name}_{args.expensive_model_name}_log/"
csv_path=f"./results/{benchmark_name}_{args.model_name}_{args.expensive_model_name}_csv/"
embedding_path="./embedding_data/"

for directory in [log_path,csv_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

model_name=args.model_name

expensive_model_name=args.expensive_model_name

distance_metric = "angular"

metric="expensive"

time_tag=args.time_tag

output_path=os.path.join(log_path,f"{dataset_name}_{model_name}_{expensive_model_name}_{split}_{time_tag}")
output_file=open(output_path,"w")

csv_name=f"{dataset_name}_{model_name}_{expensive_model_name}_{split}_deg{graph_degree}_{time_tag}.csv"

csv_file=open(os.path.join(csv_path,csv_name), 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["algo_name","opt_recall","ndcg_gt","# of bi-encoder evals","# of reranker evals","# of tokens","# of api_calls"])

print(dataset_name)

query=[]
query_id={}
query_name=[]
for qid, value in queries.items():
    query_id[qid]=len(query)
    query_name.append(qid)
    query.append(value)
print("read query complete")

query_embedding_name=os.path.join(embedding_path,os.path.join(f"./{benchmark_name}/{dataset_name}_{split}_{model_name}_query_embeddings.npy"))

print(query_embedding_name)
if os.path.exists(query_embedding_name):
    query_embeddings = np.load(query_embedding_name).astype("float32")
    print("load from file")
else:
    print("no bi-encoder embedding provided")
    assert(False)
print("query embeddings complete", query_embeddings.shape)

expensive_query_embeddings=None

passage=[]
passage_id=[]
passage_name_id={}

def clean_format(x):
    ret=x.replace("[","")
    ret=ret.replace("]","")
    return ret

for pid, value in corpus.items():
    passage_id.append(pid)
    passage_name_id[pid]=len(passage)
    passage.append(clean_format(value["title"]+" "+value["text"]))

passage_embedding_name=os.path.join(embedding_path,os.path.join(f"./{benchmark_name}/{dataset_name}_{model_name}_passage_embeddings.npy"))

if os.path.exists(passage_embedding_name):
    full_passage_embeddings=np.load(passage_embedding_name).astype("float32")
    print("load from file")
    print("passage embeddings complete", full_passage_embeddings.shape)
else:
    print("no bi-encoder embedding provided")
    assert(False)

groundtruth={}
for qid, res in qrels.items():
    groundtruth[qid]=[]
    for pid, value in res.items():
        if value!=0 and pid in passage_name_id:
            groundtruth[qid].append((value,passage_name_id[pid]))
    groundtruth[qid]=[x[1] for x in sorted(groundtruth[qid],reverse=True)]

passage_embeddings=full_passage_embeddings

def excluded_passages(qid):
    global passage_id,passage_name_id,passage,passage_embeddings,full_passage_embeddings

    current_excluded_ids=[]
    for pid in excluded_ids[qid]:
        if pid in corpus:
            current_excluded_ids.append(pid)
    current_excluded_ids=list(set(current_excluded_ids))

    passage_embeddings=np.zeros((full_passage_embeddings.shape[0]-len(current_excluded_ids),full_passage_embeddings.shape[1]),dtype=np.float32)
    passage_id=[]
    passage_name_id={}
    passage=[]

    i=0
    for pid, value in corpus.items():
        if pid not in current_excluded_ids:
            passage_id.append(pid)
            passage_name_id[pid]=len(passage)
            passage_embeddings[len(passage)]=full_passage_embeddings[i]
            passage.append(value["title"]+" "+value["text"])
        i+=1

    print(f"exclude passages complete: total {full_passage_embeddings.shape[0]} now {len(passage)}")

def build_index(qid,passage_embeddings,ann_algo_name):
    k = 10
    if ann_algo_name=="diskann":

        def get_index_query_parameters(config_path):
            configurations = []
            with open(config_path, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)["float"][distance_metric]
            for algo_type in config:
                if algo_type["name"] != "vamana(diskann)":  # Avoids vamana-pq(diskann)
                    continue
                configurations += algo_type["run_groups"].values()
            return configurations

        config_yml_path = "vamana-config.yaml"
        configurations = get_index_query_parameters(config_yml_path)
        configuration=configurations[-1]

        args = configuration["args"][0]
        alpha = args["alpha"]
        complexity=args["l_build"]
        prefix = f"{dataset_name}-{query_id[qid]}-{model_name}-{alpha}-{complexity}-{graph_degree}"

        print(prefix)

        if not os.path.exists(f"./indices/{dataset_name}"):
            os.makedirs(f"./indices/{dataset_name}")
        directory = f"./indices/{dataset_name}"

        if not os.path.exists(directory + "/" + prefix):

            diskannpy.build_memory_index(
                passage_embeddings,
                alpha=alpha,
                complexity=complexity,
                graph_degree=graph_degree,
                distance_metric="cosine",
                index_directory=directory,
                num_threads=0,
                use_pq_build=False,
                use_opq=False,
                index_prefix=prefix,
            )

        return directory + "/" + prefix
    
    elif ann_algo_name=="knn":

        prefix = f"{dataset_name}-{query_id[qid]}-{model_name}-knn-{graph_degree}"

        if not os.path.exists(f"./indices/{dataset_name}"):
            os.makedirs(f"./indices/{dataset_name}")
        directory = f"./indices/{dataset_name}"

        if os.path.exists(directory + "/" + prefix + ".pkl") is False:
            norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
            normalized_passage_embeddings = passage_embeddings / norms

            normalized_query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

            dim = normalized_passage_embeddings.shape[1]  # Dimensionality of your vectors
            M = 32  # Number of neighbors in the graph
            ef_construction = 200  # Construction parameter
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = ef_construction
            index.add(normalized_passage_embeddings)
            
            print("hnsw indexing complete!!")

            index.hnsw.efSearch = 200

            k=graph_degree+1
            D, I_graph = index.search(normalized_passage_embeddings, k)

            index.hnsw.efSearch = 5000

            D , I_query = index.search(normalized_query_embeddings,500)

            with open(os.path.join(directory + "/" + prefix + ".pkl"), "wb") as f:
                pickle.dump(I_graph[:,1:], f)
                pickle.dump(I_query, f)

        return os.path.join(directory + "/" + prefix + ".pkl")
        
expensive_passage_embeddings=None

comp_count={}

def cos_sim(A,B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / norm_A / norm_B
    return similarity

def dist(qid,pid,metric="cos",comp_count_factor=1):
    global comp_count

    if metric=="cos":
        score=cos_sim(query_embeddings[qid],passage_embeddings[pid])
        comp_count["biencoder"]+=comp_count_factor
        return 1-score
    elif metric=="l2":
        l2_dist=np.linalg.norm(query_embeddings[qid]-passage_embeddings[pid],ord=2)
        comp_count["biencoder"]+=comp_count_factor
        return l2_dist

def dist_all(qid,pids,metric="cos",comp_count_factor=1):

    if pids==[]:
        return []
 
    scores=[]
    for pid in pids:
        scores.append(dist(qid,pid,metric=metric,comp_count_factor=comp_count_factor))
    return scores

def dist_rank(qid,pids):

    example_str=""
    for i in range(len(pids)):
        if i!=0:
            example_str+=">"
        example_str+=f"[{i}]"
    
    rank_list=[]
    output_validity=True
        
    if "gemini" in llm_model_name:
        sys_instruct=f"You are an intelligent assistant that can rank answers based on their relevancy to the question. I will provide you with {len(pids)} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query[qid]}."

        messages=[]
        for i,pid in enumerate(pids):
            messages.append(f"[{i+1}] {passage[pid]}")

        messages.append(f"Search Query: {query[qid]}. \nRank the {len(pids)} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be like [1] > [2] ... > [{len(pids)}]. Only response the ranking results, do not say any word or explain.")

        total_tokens=(len(sys_instruct)+sum([len(x) for x in messages]))//4

        output=""
        for try_id in range(1):
            call_success=True
            try:

                response = client.models.generate_content(
                    model=llm_model_name,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        maxOutputTokens=window_size*10,
                        system_instruction=sys_instruct),
                    contents=messages
                )
                output=response.text
            except:
                call_success=False
            if call_success is True and type(output)!=type(None):
                break
            else:
                t=1
                time.sleep(t)

        comp_count["tokens"]+=total_tokens
        comp_count["api_calls"]+=1

    rank_list=[]
    output_validity=True
    if output is not None:
        for i in range(len(pids)):
            if f"[{i+1}]" in output:
                rank_list.append((output.find(f"[{i+1}]"),i))
            else:
                output_validity=False
                rank_list.append((len(output)+dist(qid,pids[i],metric="cos",comp_count_factor=0),i))
    else:
        output_validity=False
    
    if output_validity is False:
        comp_count["invalid"]+=1
        return pids
    else:
        rank_list=sorted(rank_list)
        ret=[]
        for i in range(len(pids)):
            ret.append(pids[rank_list[i][1]])
        return ret

graph=[]
start=0
max_deg=0
num_nodes=0

max_nodes=len(passage)

def read_diskann_index(index_path):
    with open(index_path, 'rb') as index_path:
        global graph,start,max_deg,num_nodes

        graph=[]

        expected_file_size = struct.unpack('Q', index_path.read(struct.calcsize('Q')))[0]
        max_deg = struct.unpack('I', index_path.read(struct.calcsize('I')))[0]
        start = struct.unpack('I', index_path.read(struct.calcsize('I')))[0]
        file_frozen_pts = struct.unpack('Q', index_path.read(struct.calcsize('Q')))[0]
        bytes_read=struct.calcsize('Q')*2+struct.calcsize('I')*2
        i=0
        while bytes_read!=expected_file_size:
            deg=struct.unpack('I', index_path.read(struct.calcsize('I')))[0]
            graph.append(list(struct.unpack(f'{deg}I', index_path.read(deg * 4))))
            bytes_read+=(deg+1)*struct.calcsize('I')
            i+=1

        num_nodes=len(graph)

        print("read completes")

knn_answers_for_query=[]

def read_knn_index(index_path):
    global graph,start,max_deg,num_nodes,knn_answers_for_query
    print("read knn index from numpy file")
    with open(index_path, "rb") as f:
        graph = pickle.load(f)
        knn_answers_for_query = pickle.load(f)

    num_nodes=len(graph)
    max_deg=graph_degree

in_Q=[0]*max_nodes
in_vis=[0]*max_nodes
col=0

def greedy_search(qid,k_neighbors,search_L,start,metric="cos"):
    global graph,in_Q,in_vis,col

    col+=1

    vis=[]

    if isinstance(start,list):
        cur_dist=[]
        for pid in start:
            cur_dist.append(dist(qid,pid,metric=metric))
        Q=[]
        for i in range(len(start)):
            Q.append((cur_dist[i],start[i]))
            in_Q[start[i]]=col
        heapq.heapify(Q)
    else:
        Q=[(dist(qid,start,metric),start)]
        in_Q[start]=col

    p=Q[0][1]
    while p!=-1:
        if in_vis[p]!=col:
            bisect.insort(vis,(Q[0][0],p))
            in_vis[p]=col
        heapq.heappop(Q)

        V=[x for x in graph[p] if in_Q[x]!=col]
        cur_dist=dist_all(qid,V,metric)

        for i in range(len(V)):
            x=V[i]
            if in_Q[x]!=col:
                heapq.heappush(Q,(cur_dist[i],x))
                in_Q[x]=col

        p=-1
        if len(Q)>0:
            if len(vis)<search_L or Q[0][0]<=vis[search_L-1][0]:
                p=Q[0][1]

    for i in range(min(len(Q),k_neighbors)):
        if in_vis[Q[0][1]]!=col:
            bisect.insort(vis,(Q[0][0],Q[0][1]))
        heapq.heappop(Q)

    neighbors=[]
    distances=[]
    for i in range(min(k_neighbors,len(vis))):
        neighbors.append(vis[i][1])
        distances.append(vis[i][0])

    len_Q=len(Q)
    len_V=len(V)

    return neighbors

def insert(qid,Q,x):
    global comp_count,in_Q,col
    
    comp_count["expensive"]+=len(x)
    for pid in x:
        Q.append(pid)
    for i in range(1):
        j=len(Q)
        while(j>0):
            new_pid=False
            for pid in Q[max(0,j-window_size):j]:
                if in_Q[pid]!=col:
                    new_pid=True
            if new_pid is True:
                Q[max(0,j-window_size):j]=dist_rank(qid,Q[max(0,j-window_size):j])
            else:
                break
            if j<=window_size:
                break
            else:
                j-=window_size//2

def greedy_search_cmp_quota(qid,k_neighbors,query_quota,start,metric="llm"):
    global graph,in_Q,in_vis,col

    global comp_count

    if query_quota==100:
        queue_size=20
    elif query_quota==200:
        queue_size=20
    elif query_quota==300:
        queue_size=30
    elif query_quota==400:
        queue_size=40
    elif query_quota==500:
        queue_size=50
    else:
        queue_size=50

    current_dist_count=0

    col+=1

    neighbors_seen=[]

    if isinstance(start,list):
        if len(start)>query_quota:
            start=start[:query_quota]
        Q=[]
        insert(qid,Q,start)
        for x in start:
            in_Q[x]=col

        neighbors_seen=start

        Q=Q[:queue_size]

        current_dist_count+=len(start)
    else:
        Q=[start]
        in_Q[start]=col

        neighbors_seen=[start]

        current_dist_count=1

    id=0
    while id<len(Q):
        p=Q[id]
        in_vis[p]=col
        V=[x for x in graph[p] if in_Q[x]!=col]
        if current_dist_count+len(V)>query_quota:
            V=V[:query_quota-current_dist_count]
        
        insert(qid,Q,V)
        for pid in V:
            in_Q[pid]=col

        neighbors_seen+=V

        Q=Q[:queue_size] 
        current_dist_count+=len(V)  
        id=0
        while id<len(Q) and in_vis[Q[id]]==col:
            id+=1

    neighbors=[]
    for x in Q[:k_neighbors]:
        neighbors.append(x)

    return neighbors,neighbors_seen

def SlideGAR(qid, pids, k_neighbors,query_quota,metric="llm"):

    L=[]
    for i in range(window_size):
        L.append(pids[i])
    R1=[]
    i=window_size
    current_dist_count=0
    iteration=0
    while current_dist_count<=query_quota:
        B=dist_rank(qid,L)

        if iteration==0:
            current_dist_count=len(L)
            comp_count["expensive"]+=len(L)
        else:
            current_dist_count+=len(L)-window_size//2
            comp_count["expensive"]+=len(L)-window_size//2

        L1=B[:window_size//2]

        if current_dist_count>=query_quota:
            break

        Frontier=[]
        for x in B:
            for y in graph[x]:
                if (y not in Frontier) and (y not in B) and (y not in R1):
                    Frontier.append(y)
                    if len(Frontier)>=window_size//2:
                        break
        R1=B[window_size//2:]+R1
        if iteration%2==0:
            L=L1+Frontier[:window_size//2]
        else:
            L=L1
            while i < len(pids):
                if (pids[i] not in R1) and (pids[i] not in L):
                    L.append(pids[i])
                i+=1
                if len(L)>=window_size:
                    break
    
        if current_dist_count+len(L)-len(L1)>query_quota:
            L=L[:len(L1)+query_quota-current_dist_count]
        iteration+=1
    R1=L1+R1

    neighbors_seen=R1

    neighbors=R1[:k_neighbors]
    distances=[i/k_neighbors for i in range(k_neighbors)]
    return neighbors,distances,neighbors_seen

def rerank(qid, pids, k_neighbors,metric="cos"):
    global comp_count,in_Q,col

    col+=1

    if metric=="llm":
        Q=[]
        insert(qid,Q,pids)
        Q=Q[:k_neighbors]
        neighbors=Q[:k_neighbors]
        distances=[i/k_neighbors for i in range(k_neighbors)]
        return neighbors[:k_neighbors], distances[:k_neighbors]
    else:
        print("unsupported metric")

def process_query(qid,retrieval_algo,k_neighbors,query_complexity,query_quota,second_query_complexity,second_query_quota):
    global comp_count,start,cos_threshold

    comp_count={"biencoder":0,"expensive":0,"invalid":0,"processed_query":0,"time":0,"tokens":0,"api_calls":0}

    comp_count["processed_query"]+=1

    assert(query_complexity==0 or query_quota==0)
    assert(second_query_complexity==0 or second_query_quota==0)

    start_time=time.time()

    if algo_name=="RR" or algo_name=="RGS":
        retrieval_neighbors=greedy_search(query_id[qid],k_neighbors=query_complexity,search_L=query_complexity,start=start)
    else:
        retrieval_neighbors=knn_answers_for_query[query_id[qid]]        

    if retrieval_algo=="bi(llm-baseline)":
        second_L=max(k_neighbors,second_query_quota)

        expensive_neighbors,expensive_distances=rerank(query_id[qid],copy.deepcopy(retrieval_neighbors[:second_L]),k_neighbors=k_neighbors,metric="llm")
        
        neighbors_seen=retrieval_neighbors[:second_L]

        neighbors=expensive_neighbors[:k_neighbors]
        distances=expensive_distances[:k_neighbors]
    elif retrieval_algo=="bi(llm-SlideGAR)":

        second_L=max(k_neighbors,second_query_quota)

        expensive_neighbors,expensive_distances,neighbors_seen=SlideGAR(query_id[qid],copy.deepcopy(retrieval_neighbors[:second_L]),k_neighbors=k_neighbors,query_quota=second_query_quota,metric="llm")
        
        neighbors=expensive_neighbors[:k_neighbors]
        distances=expensive_distances[:k_neighbors]
    elif retrieval_algo=="bi(llm-ours)":
        in_Q=[]
        in_vis=[]

        second_L=int(second_query_quota//5)

        expensive_neighbors,neighbors_seen=greedy_search_cmp_quota(query_id[qid],k_neighbors=k_neighbors,query_quota=second_query_quota,start=start if second_L==0 else retrieval_neighbors[:second_L],metric="expensive")

        neighbors=expensive_neighbors[:k_neighbors]
        distances=[i/k_neighbors for i in range(k_neighbors)]
    else:
        neighbors=retrieval_neighbors[:k_neighbors]
        distances=dist_all(query_id[qid],neighbors,metric="cos",comp_count_factor=0)

        expensive_neighbors=neighbors
        neighbors_seen=neighbors

    end_time=time.time()
    comp_count["time"]=end_time-start_time

    count_gt=0
    count_opt=0
    count_final=0
    for p_name in qrels[qid]:
        if qrels[qid][p_name]>0:
            pid=passage_name_id[p_name]
            count_gt+=1
            if pid in neighbors_seen:
                count_opt+=1
            if pid in expensive_neighbors:
                count_final+=1

    count_gt=min(count_gt,10)
    count_opt=min(count_opt,10)
    count_final=min(count_final,10)

    opt_recall=count_opt/count_gt
    final_recall=count_final/count_gt

    assert(opt_recall>=final_recall)

    ret={qid:{}}
    for i in range(len(neighbors)):
        pid=passage_id[neighbors[i]]
        distance=distances[i]
        ret[qid][pid]=1-float(distance)

    ret["opt_recall"]=opt_recall
    ret["final_recall"]=final_recall

    return {"retrieval_algo":retrieval_algo,"second_query_quota":second_query_quota, **ret,**comp_count}

def calculate_ndcg(qrels, answers, cutoffs):
    count=0

    run = {}
    for qid, doc_scores in answers.items():
        run[qid] = {}
        for doc_id, score in doc_scores.items():
            if doc_id!=qid:
                run[qid][doc_id] = score

    metrics_to_evaluate = {f"ndcg_cut.{x}": x for x in cutoffs}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_evaluate)
    results = evaluator.evaluate(run)

    ndcg_averages = []
    for cutoff in cutoffs:
        ndcg_key = f"ndcg_cut_{cutoff}"
        ndcg_values = [res[ndcg_key] for res in results.values() if ndcg_key in res]
        if ndcg_values:
            ndcg_averages.append(sum(ndcg_values) / len(ndcg_values))
        else:
            ndcg_averages.append(0.0)
    
    return ndcg_averages

first_query_complexity=5000

results=[]

qid_count=0

for qid in tqdm(groundtruth):
    excluded_passages(qid)

    if algo_name=="RGS" or algo_name=="RR":
        prefix=build_index(qid,passage_embeddings,"diskann")
    else:
        prefix=build_index(qid,passage_embeddings,"knn")

    query_args=[]
    query_args.append((qid,"bi",10,first_query_complexity,0,0,0))
    if algo_name=="RGS":
        read_diskann_index(prefix)
        window_size=10
        for second_query_quota in [100]:
            query_args.append((qid,"bi(llm-ours)",10,first_query_complexity,0,0,second_query_quota))
    elif algo_name=="RR":
        read_diskann_index(prefix)
        window_size=10
        for second_query_quota in [100,300,500]:
            query_args.append((qid,"bi(llm-baseline)",10,first_query_complexity,0,0,second_query_quota))
    elif algo_name=="SlideGAR":
        read_knn_index(prefix)
        window_size=20
        for second_query_quota in [100,300,500]:
            query_args.append((qid,"bi(llm-SlideGAR)",10,first_query_complexity,0,0,second_query_quota))
    num_processes=10
    with Pool(processes=num_processes) as pool:
        results+=pool.starmap(process_query, query_args)

    qid_count+=1

    if qid_count>10:
        break

plot_algo_name={
    "bi":"Retrieve",
    "bi(llm-ours)":"RetrieveGuidedSearch",
    "bi(llm-baseline)":"Retreve-and-Rerank",
    "bi(llm-SlideGAR)":"SlideGAR"
}

for retrieval_algo in ["bi","bi(llm-ours)","bi(llm-baseline)","bi(llm-SlideGAR)"]:
    for second_query_quota in [0,100,300,500]:
        opt_recall_list=[]
        final_recall_list=[]
        
        predictions_trec={}
        comp_count={"biencoder":0,"expensive":0,"invalid":0,"processed_query":0,"time":0,"tokens":0,"api_calls":0}
        for entry in results:
            if entry["retrieval_algo"]==retrieval_algo and entry["second_query_quota"]==second_query_quota:
                for key,value in entry.items():
                    if key in ["biencoder","expensive","invalid","processed_query","time","tokens","api_calls"]:
                        comp_count[key]+=value
                    elif key=="opt_recall":
                        opt_recall_list.append(value)
                    elif key=="final_recall":
                        final_recall_list.append(value)
                    elif key!="retrieval_algo" and key!="second_query_quota":
                        predictions_trec[key]=value
        
        if len(predictions_trec)==0:
            continue

        avg_opt_recall=sum(opt_recall_list)/len(opt_recall_list)
        avg_final_recall=sum(final_recall_list)/len(final_recall_list)

        comp_count["biencoder"]/=comp_count["processed_query"]
        comp_count["expensive"]/=comp_count["processed_query"]
        comp_count["invalid"]/=comp_count["processed_query"]
        comp_count["tokens"]/=comp_count["processed_query"]
        comp_count["api_calls"]/=comp_count["processed_query"]
        ndcg_score=0

        print("avg invalid per query", comp_count["invalid"])
        print("avg invalid per query", comp_count["invalid"],file=output_file)

        test_predictions_trec=deepcopy(predictions_trec)
        ndcg_score_gt = calculate_ndcg(qrels, test_predictions_trec, [1,5,10])

        print("Queries:", len(groundtruth),file=output_file)
        print("NDCG: ", ndcg_score_gt,file=output_file)
        output_file.flush()
        print("Queries:", len(groundtruth))
        print("NDCG: ", ndcg_score_gt)
        
        csv_writer.writerow([plot_algo_name[retrieval_algo],avg_opt_recall,ndcg_score_gt,comp_count["biencoder"],comp_count["expensive"],comp_count["tokens"],comp_count["api_calls"]])
        csv_file.flush()