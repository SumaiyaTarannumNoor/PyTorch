import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gradio as gr
import pickle
import numpy as np
import math
import random
from collections import defaultdict

# Dataset Download & Loading

import kagglehub

path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
print(f"Dataset downloaded to: {path}")

ratings_files = [f for f in os.listdir(path) if "rating" in f.lower() and f.endswith(".csv")]
movies_files = [f for f in os.listdir(path) if "movie" in f.lower() and f.endswith(".csv")]

if not ratings_files or not movies_files:
    raise FileNotFoundError("Ratings or movies CSV not found in the dataset folder.")

RATINGS_CSV = os.path.join(path, ratings_files[0])
MOVIES_CSV = os.path.join(path, movies_files[0])

ratings_full = pd.read_csv(RATINGS_CSV)
movies_full = pd.read_csv(MOVIES_CSV)

# Use a fraction for faster training/testing
fraction = 0.2
ratings = ratings_full.sample(frac=fraction, random_state=42)
movies = movies_full[movies_full['movieId'].isin(ratings['movieId'].unique())].copy()


# Prepare Mappings

user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}
idx2movie = {i: m for m, i in movie2idx.items()}
movieid2title = movies.set_index("movieId")['title'].to_dict()

n_users = len(user2idx)
n_movies = len(movie2idx)
print(f"Users: {n_users}, Movies: {n_movies}, Ratings: {len(ratings)}")


# Train / Validation / Test Split

train_val, test = train_test_split(ratings, test_size=0.20, random_state=42)
train, val = train_test_split(train_val, test_size=0.125, random_state=42)  # 70/20/10 split
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")


# Dataset & DataLoader

class MovieLensDataset(Dataset):
    def __init__(self, df, user2idx, movie2idx):
        self.users = torch.tensor(df['userId'].map(user2idx).values, dtype=torch.long)
        self.movies = torch.tensor(df['movieId'].map(movie2idx).values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


train_ds = MovieLensDataset(train, user2idx, movie2idx)
val_ds = MovieLensDataset(val, user2idx, movie2idx)
test_ds = MovieLensDataset(test, user2idx, movie2idx)

train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=1024)
test_dl = DataLoader(test_ds, batch_size=1024)


# Matrix Factorization Model

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        u = self.user_emb(users)
        v = self.item_emb(items)
        dot = (u * v).sum(1)
        return dot + self.user_bias(users).squeeze() + self.item_bias(items).squeeze()


# Device Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Model & Mappings

model = MatrixFactorization(n_users, n_movies, n_factors=64).to(device)

# Load checkpoint and mappings
model_path = "mf_model_64.pt"
mapping_path = "mapping_64.pkl"

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with open(mapping_path, "rb") as f:
    mapping = pickle.load(f)
user2idx = mapping['user2idx']
movie2idx = mapping['movie2idx']
idx2movie = mapping['idx2movie']
movieid2title = mapping['movieid2title']


# Prepare evaluation helpers

# Prepare test_truth and seen_indicies
test_truth = defaultdict(list)
for row in test.itertuples(index=False):
    if row.rating >= 4.0:
        if row.userId in user2idx and row.movieId in movie2idx:
            test_truth[row.userId].append(row.movieId)

test_truth_idx = {uid: set(movie2idx[m] for m in mids if m in movie2idx) 
                  for uid, mids in test_truth.items()}

train_grouped = train.groupby('userId')['movieId'].apply(list).to_dict()
seen_indicies = {uid: set(movie2idx[m] for m in mids if m in movie2idx) 
                 for uid, mids in train_grouped.items()}


# Recommendation helpers

def recommend_topN_fast(model, user_idx, seen_indicies_set, N=10, batch_size=2048):
    model.eval()
    all_items = torch.arange(n_movies, device=device)
    scores_chunks = []
    with torch.no_grad():
        for start in range(0, n_movies, batch_size):
            end = min(start + batch_size, n_movies)
            items_batch = all_items[start:end]
            users_batch = torch.full((end - start,), user_idx, dtype=torch.long, device=device)
            scores_batch = model(users_batch, items_batch)
            scores_chunks.append(scores_batch.cpu())
    scores = torch.cat(scores_chunks).numpy()
    if seen_indicies_set:
        scores[list(seen_indicies_set)] = -np.inf
    top_idx = np.argpartition(-scores, N)[:N]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx.tolist()

def precision_recall_ndcg_fast(model, user_ids, test_truth_idx, seen_indicies, K=10):
    precisions, recalls, ndcgs = [], [], []
    for uid in user_ids:
        if uid not in user2idx:
            continue
        relevent = test_truth_idx.get(uid, set())
        if len(relevent) == 0:
            continue
        recs = recommend_topN_fast(model, user2idx[uid], seen_indicies.get(uid, set()), N=K)
        hit_count = sum(1 for r in recs if r in relevent)
        prec = hit_count / K
        rec = hit_count / len(relevent)
        dcg = sum(1 / math.log2(i+2) for i, r in enumerate(recs) if r in relevent)
        idcg = sum(1 / math.log2(i+2) for i in range(min(len(relevent), K)))
        ndcg = dcg / idcg if idcg > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg)
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


# Evaluate Sample

all_test_users = list(test_truth_idx.keys())
sample_users = random.sample(all_test_users, min(2000, len(all_test_users)))
p10, r10, n10 = precision_recall_ndcg_fast(model, sample_users, test_truth_idx, seen_indicies, K=10)
print(f"Precision@10: {p10:.4f}, Recall@10: {r10:.4f}, NDCG@10: {n10:.4f}")


# User recommendation

def recommend_movies(user_id, N=10):
    if user_id not in user2idx:
        print(f"User ID {user_id} not found.")
        return []
    uidx = user2idx[user_id]
    seen_set = seen_indicies.get(user_id, set())
    top_movies_indicies = recommend_topN_fast(model, uidx, seen_set, N=N)
    titles = []
    for midx in top_movies_indicies:
        raw_movie_id = idx2movie.get(midx, None)
        title = movieid2title.get(raw_movie_id, None)
        titles.append(title if title else f"Movie ID {raw_movie_id}")
    return titles

# Gradio Interface

def gradio_recommend(user_id, top_n=5):
    try:
        user_id = int(user_id)
    except ValueError:
        return ["Invalid user ID"]
    return recommend_movies(user_id, N=top_n)

iface = gr.Interface(
    fn=gradio_recommend,
    inputs=["number", gr.Number(value=5, label="Top N")],
    outputs="text",
    live=True
)

iface.launch()
