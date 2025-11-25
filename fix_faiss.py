import re

with open("run_retrieval.py", "r") as f:
    content = f.read()

# 1. Add faiss_index global variable after graph=[]
old = "graph=[]\nstart=0"
new = """graph=[]
start=0
faiss_index=None  # FAISS index for direct search"""
content = content.replace(old, new)

# 2. Modify knn section to save and use FAISS index
old_knn = """elif ann_algo_name=="knn":

        prefix = f"{dataset_name}-{model_name}-knn-{graph_degree}"

        if os.path.exists(directory + "/" + prefix + ".npy") is False:
            norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
            normalized_passage_embeddings = passage_embeddings / norms

            dim = normalized_passage_embeddings.shape[1]  # Dimensionality of your vectors
            M = 64  # Number of neighbors in the graph
            ef_construction = 200  # Construction parameter
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = ef_construction
            index.add(normalized_passage_embeddings)
            
            ef_search = 1000  # A higher value leads to more accurate but slower searches
            index.hnsw.efSearch = ef_search

            k=graph_degree+1
            D, I = index.search(normalized_passage_embeddings, k)

            np.save(os.path.join(directory + "/" + prefix + ".npy"), I[:,1:])
        index_path=f"./indices/{prefix}.npy"
        graph=np.load(index_path).tolist()
        num_nodes=len(graph)
        max_deg=graph_degree

        in_vis=[0]*num_nodes
        in_Q=[0]*num_nodes
        col=0"""

new_knn = """elif ann_algo_name=="knn":
        global faiss_index
        
        prefix = f"{dataset_name}-{model_name}-knn-{graph_degree}"
        faiss_index_path = os.path.join(directory, prefix + "_faiss.index")

        # Normalize embeddings
        norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
        normalized_passage_embeddings = passage_embeddings / norms

        if os.path.exists(faiss_index_path):
            # Load existing FAISS index
            faiss_index = faiss.read_index(faiss_index_path)
            print(f"Loaded FAISS index from {faiss_index_path}")
        else:
            # Build new FAISS index
            dim = normalized_passage_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            faiss_index.add(normalized_passage_embeddings)
            faiss.write_index(faiss_index, faiss_index_path)
            print(f"Built and saved FAISS index to {faiss_index_path}")

        # Still build kNN graph for compatibility
        if os.path.exists(directory + "/" + prefix + ".npy") is False:
            k=graph_degree+1
            D, I = faiss_index.search(normalized_passage_embeddings, k)
            np.save(os.path.join(directory + "/" + prefix + ".npy"), I[:,1:])
        
        index_path=f"./indices/{prefix}.npy"
        graph=np.load(index_path).tolist()
        num_nodes=len(graph)
        max_deg=graph_degree

        in_vis=[0]*num_nodes
        in_Q=[0]*num_nodes
        col=0"""

content = content.replace(old_knn, new_knn)

# 3. Add faiss_search function after greedy_search
faiss_search_func = """
def faiss_search(qid, k_neighbors):
    \"\"\"Use FAISS to directly search for k nearest neighbors\"\"\"
    global faiss_index, query_embeddings
    query_vec = query_embeddings[qid:qid+1].astype("float32")
    query_vec = query_vec / np.linalg.norm(query_vec)
    D, I = faiss_index.search(query_vec, k_neighbors)
    return list(I[0]), list(1 - D[0])  # Convert IP similarity to distance

"""

# Insert after greedy_search function
content = content.replace("def insert(qid,Q,x):", faiss_search_func + "def insert(qid,Q,x):")

# 4. Modify process_query to use faiss_search for first stage
old_retrieval = """    if (qid,query_complexity,"complexity") in search_results:
        retrieval_neighbors=search_results[(qid,query_complexity,"complexity")]
    else:
        retrieval_neighbors=greedy_search(query_id[qid],k_neighbors=query_complexity,search_L=query_complexity,start=start,metric="cos")"""

new_retrieval = """    if (qid,query_complexity,"complexity") in search_results:
        retrieval_neighbors=search_results[(qid,query_complexity,"complexity")]
    else:
        # Use FAISS direct search for better results
        if faiss_index is not None:
            retrieval_neighbors, _ = faiss_search(query_id[qid], k_neighbors=query_complexity)
        else:
            retrieval_neighbors=greedy_search(query_id[qid],k_neighbors=query_complexity,search_L=query_complexity,start=start,metric="cos")"""

content = content.replace(old_retrieval, new_retrieval)

with open("run_retrieval.py", "w") as f:
    f.write(content)

print("Done! Applied FAISS search fix.")
