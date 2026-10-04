"""
ExpHarness experience graph server (FastAPI).

Each node stores a natural-language experience (a skill distilled from a successful
trajectory or a lesson from a failed one), its Contriever embedding, an EMA utility
estimate and a retrieval count. Retrieval is controlled by the copilot's (R, W):

  1. Semantic seeding     cosine top-m seeds (m = 10)
  2. Graph diffusion      personalized PageRank from the seeds, restart prob. alpha = R/100
  3. Utility-aware rank   score = lambda * UCB + (1 - lambda) * cosine,  lambda = W/100

Hyper-parameters follow the paper: K_nn = 5, theta = 0.3, |V|_max = 2000, m = 10,
top-K = 10 (set by the caller), UCB c = 1.0, EMA rate beta = 0.1.

Endpoints
  POST /retrieve        {queries, R_values, W_values, topk} -> experiences + node indices
  POST /update_rewards  per-episode rewards -> EMA utility update of retrieved nodes
  POST /add_experience  insert new experiences (dedup + kNN edges + capacity pruning)
  POST /save            dump the graph to JSON
  GET  /stats           graph size / retrieval statistics
"""

import argparse
import json
import logging
import os
import threading
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ============================================================================
# Contriever Retriever
# ============================================================================

class ContrieverRetriever:
    """Contriever-based text encoder and retriever."""

    def __init__(self, model_name: str = "facebook/contriever", device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"[ContrieverRetriever] Loaded {model_name} on {self.device}")

    def encode_texts(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                last_hidden = last_hidden.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
                embeddings = last_hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def rerank_with_cached_embeddings(
        self, query_embedding: np.ndarray, candidate_indices: List[int],
        candidate_embeddings: np.ndarray, candidate_texts: List[str],
        top_k: int = 3,
    ):
        if not candidate_texts:
            return [], []
        similarities = np.dot(candidate_embeddings, query_embedding)
        top_k = min(top_k, len(candidate_texts))
        top_indices = np.argsort(-similarities)[:top_k]
        reranked_texts = [candidate_texts[idx] for idx in top_indices]
        reranked_original_indices = [candidate_indices[idx] for idx in top_indices]
        return reranked_texts, reranked_original_indices


# ============================================================================
# Experience Graph (in-process)
# ============================================================================

class ExperienceGraph:
    # Hyper-parameters (paper, Appendix "Implementation Details")
    NUM_SEEDS = 10        # m: semantic seeds
    UCB_C = 1.0           # c: UCB exploration coefficient used in ranking
    EMA_BETA = 0.1        # beta: utility EMA rate

    def __init__(self, retriever, k_neighbors=5, sim_threshold=0.3, max_nodes=2000):
        self.retriever = retriever
        self.k_neighbors = k_neighbors
        self.sim_threshold = sim_threshold
        self.max_nodes = max_nodes

        self.raw_texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.edge_list: List[tuple] = []
        self.edge_index: Optional[torch.Tensor] = None
        self.num_nodes: int = 0
        # Bandit statistics per node
        self.retrieval_counts: List[int] = []
        self.avg_rewards: List[float] = []      # EMA of rewards
        self.node_ages: List[int] = []           # steps since node was added
        self.global_step: int = 0                # incremented by update_rewards
        self.total_retrievals: int = 0           # total retrieval events
        self.ema_alpha: float = self.EMA_BETA    # EMA rate beta for utility updates
        self.ucb_c: float = 1.5                  # UCB exploration coefficient
        self.protection_age: int = 5             # new node protection (in steps)
        self.lock = threading.Lock()

    def _rebuild_edge_index(self):
        self._edge_version = getattr(self, '_edge_version', 0) + 1   # invalidates the PPR cache
        if not self.edge_list:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            self.edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous()

    def _ucb_score(self, node_idx: int) -> float:
        """Compute UCB score for a node. Lower score = more likely to be evicted."""
        import math
        age = self.node_ages[node_idx]
        count = self.retrieval_counts[node_idx]
        avg_r = self.avg_rewards[node_idx]

        # New node: protect
        if age < self.protection_age:
            return float('inf')

        # Never retrieved: borrow neighbor score, decay with age
        if count == 0:
            neighbor_rewards = []
            for s, d in self.edge_list:
                if s == node_idx and self.retrieval_counts[d] >= 3:
                    neighbor_rewards.append(self.avg_rewards[d])
                elif d == node_idx and self.retrieval_counts[s] >= 3:
                    neighbor_rewards.append(self.avg_rewards[s])
            neighbor_val = np.mean(neighbor_rewards) if neighbor_rewards else 0.0
            return max(neighbor_val * 0.5, 0.1) / math.sqrt(max(age, 1) / 10.0)

        # Too few retrievals: pure exploration
        if count < 3:
            return self.ucb_c * math.sqrt(math.log(max(self.total_retrievals, 1) + 1) / count)

        # Normal UCB: exploitation (EMA avg_reward) + exploration
        exploitation = avg_r
        exploration = self.ucb_c * math.sqrt(math.log(max(self.total_retrievals, 1) + 1) / count)
        return exploitation + exploration

    def _evict_nodes(self, num_to_evict: int):
        if num_to_evict <= 0 or self.num_nodes == 0:
            return
        num_to_evict = min(num_to_evict, self.num_nodes)

        # Compute UCB scores for all nodes
        scores = [self._ucb_score(i) for i in range(self.num_nodes)]
        evict_indices = set(np.argsort(scores)[:num_to_evict].tolist())
        keep_indices = [i for i in range(self.num_nodes) if i not in evict_indices]

        if not keep_indices:
            self.raw_texts, self.embeddings, self.edge_list = [], None, []
            self.retrieval_counts, self.avg_rewards, self.node_ages = [], [], []
            self.num_nodes = 0
            self._rebuild_edge_index()
            return

        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        self.raw_texts = [self.raw_texts[i] for i in keep_indices]
        self.embeddings = self.embeddings[keep_indices]
        self.retrieval_counts = [self.retrieval_counts[i] for i in keep_indices]
        self.avg_rewards = [self.avg_rewards[i] for i in keep_indices]
        self.node_ages = [self.node_ages[i] for i in keep_indices]
        self.edge_list = [
            (old_to_new[s], old_to_new[d])
            for s, d in self.edge_list if s in old_to_new and d in old_to_new
        ]
        self.num_nodes = len(keep_indices)
        self._rebuild_edge_index()
        logger.info(f"[ExperienceGraph] Evicted {num_to_evict} nodes (UCB-based). Now {self.num_nodes} nodes.")

    def _record_retrievals(self, node_indices: List[int]):
        for idx in node_indices:
            if 0 <= idx < len(self.retrieval_counts):
                self.retrieval_counts[idx] += 1
                self.total_retrievals += 1

    def update_rewards(self, node_indices: List[int], reward: float):
        """Update EMA reward for retrieved nodes (called after episode completes)."""
        with self.lock:
            for idx in node_indices:
                if 0 <= idx < self.num_nodes:
                    if self.retrieval_counts[idx] <= 1:
                        # First observation: set directly
                        self.avg_rewards[idx] = reward
                    else:
                        # EMA update
                        self.avg_rewards[idx] = (
                            self.ema_alpha * reward +
                            (1 - self.ema_alpha) * self.avg_rewards[idx]
                        )

    def increment_ages(self):
        """Increment age of all nodes by 1 step."""
        with self.lock:
            for i in range(self.num_nodes):
                self.node_ages[i] += 1
            self.global_step += 1

    def add_nodes(self, texts: List[str], dedup_threshold: float = 0.90, initial_rewards: List[float] = None) -> List[int]:
        if not texts:
            return []
        with self.lock:
            new_embeddings = self.retriever.encode_texts(texts)
            new_node_ids = []
            skipped = 0

            for i, (text, emb) in enumerate(zip(texts, new_embeddings)):
                # Dedup: skip if too similar to existing node
                if self.embeddings is not None and self.num_nodes > 0:
                    sims = np.dot(self.embeddings[:self.num_nodes], emb)
                    if np.max(sims) > dedup_threshold:
                        skipped += 1
                        continue

                # Evict if at capacity
                if self.num_nodes >= self.max_nodes:
                    self._evict_nodes(1)

                node_id = self.num_nodes
                self.raw_texts.append(text)
                init_reward = initial_rewards[i] if initial_rewards and i < len(initial_rewards) else 0.5
                self.retrieval_counts.append(1)
                self.avg_rewards.append(init_reward)
                self.node_ages.append(0)
                new_node_ids.append(node_id)

                if self.embeddings is None:
                    self.embeddings = emb.reshape(1, -1).copy()
                else:
                    self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])
                self.num_nodes += 1

                if node_id > 0:
                    sims = np.dot(self.embeddings[:node_id], emb)
                    top_k = min(self.k_neighbors, len(sims))
                    top_indices = np.argsort(-sims)[:top_k]
                    for neighbor_idx in top_indices:
                        if sims[neighbor_idx] >= self.sim_threshold:
                            self.edge_list.append((node_id, int(neighbor_idx)))
                            self.edge_list.append((int(neighbor_idx), node_id))

            self._rebuild_edge_index()
            added = len(new_node_ids)
            logger.info(f"[ExperienceGraph] Added {added} nodes (skipped {skipped} duplicates). Graph: {self.num_nodes} nodes.")
            return new_node_ids

    def naive_retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Pure cosine similarity retrieval. No PPR, no R parameter. Used as naive baseline."""
        if self.num_nodes == 0 or not query.strip():
            return []
        query_embedding = self.retriever.encode_texts([query])[0]
        with self.lock:
            if self.num_nodes == 0:
                return []
            sims = np.dot(self.embeddings, query_embedding)
            top_k_actual = min(top_k, self.num_nodes)
            top_indices = np.argsort(-sims)[:top_k_actual].tolist()
            return [self.raw_texts[idx] for idx in top_indices]

    _ppr_cache_A_norm_T = None
    _ppr_cache_n = None

    def _ppr_expand(self, seed_indices, alpha, n_expand=20, max_iter=10):
        """Personalized PageRank via cached sparse power iteration.
        Returns set of expanded node indices."""
        from scipy.sparse import coo_matrix
        n = self.num_nodes

        # Cache the normalized adjacency transpose
        cache_key = (id(self), getattr(self, '_edge_version', 0), n)
        if ExperienceGraph._ppr_cache_n != cache_key or ExperienceGraph._ppr_cache_A_norm_T is None:
            row = self.edge_index[0].numpy()
            col = self.edge_index[1].numpy()
            data = np.ones(len(row), dtype=np.float32)
            A = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
            deg = np.array(A.sum(axis=1)).flatten()
            deg[deg == 0] = 1
            inv_deg = 1.0 / deg
            A_norm = A.multiply(inv_deg[:, None]).tocsr()
            ExperienceGraph._ppr_cache_A_norm_T = A_norm.T.tocsr()
            ExperienceGraph._ppr_cache_n = cache_key

        A_norm_T = ExperienceGraph._ppr_cache_A_norm_T
        expanded = set()
        # Only use top-5 seeds (not all 20) for speed
        seeds = list(seed_indices)[:5]
        for seed in seeds:
            p = np.zeros(n, dtype=np.float32)
            p[seed] = 1.0
            for _ in range(max_iter):
                p_new = np.zeros(n, dtype=np.float32)
                p_new[seed] = alpha
                p_new += (1 - alpha) * A_norm_T.dot(p)
                if np.abs(p_new - p).sum() < 1e-6:
                    break
                p = p_new
            top_k = min(n_expand, n)
            top_indices = np.argsort(-p)[:top_k]
            expanded.update(top_indices.tolist())
        return expanded

    def _node_ucb(self, node_idx: int, c: float = UCB_C) -> float:
        """UCB score for retrieval ranking (not eviction)."""
        import math
        count = max(self.retrieval_counts[node_idx], 1)  # avoid division by zero
        avg_r = self.avg_rewards[node_idx]
        N = max(self.total_retrievals, 1)

        exploitation = avg_r
        exploration = c * math.sqrt(math.log(N + 1) / count)
        return exploitation + exploration

    def retrieve(self, query: str, R: float, top_k: int = 5, W: float = 50.0, query_embedding=None):
        """
        Three-stage retrieval: Cosine recall → PPR expansion → UCB+cosine selection.
        W controls UCB vs cosine weight: score = (W/100)*UCB + (1-W/100)*cosine
        When W=0: pure cosine (= naive baseline).
        Returns (texts, node_indices) tuple.
        query_embedding: optional pre-computed embedding to skip redundant encode.
        """
        if self.num_nodes == 0 or not query.strip():
            return [], []

        R = max(0, min(100, R))
        W = max(0, min(100, W))
        alpha = max(0.1, min(0.9, R / 100.0))
        lam = W / 100.0  # UCB weight: 0=pure cosine, 1=pure UCB
        if query_embedding is None:
            query_embedding = self.retriever.encode_texts([query])[0]

        with self.lock:
            if self.num_nodes == 0:
                return [], []

            # ---- Stage 1: Cosine top-m seeds ----
            sims = np.dot(self.embeddings, query_embedding)
            n_seeds = min(self.NUM_SEEDS, self.num_nodes)
            seed_indices = set(np.argsort(-sims)[:n_seeds].tolist())

            # ---- Stage 2: PPR expansion from each seed (skip if R=0) ----
            candidates = set(seed_indices)

            if R > 0 and self.edge_index is not None and self.edge_index.shape[1] > 0:
                try:
                    candidates.update(self._ppr_expand(seed_indices, alpha, n_expand=20))
                except Exception as e:
                    logger.warning(f"PPR expansion failed: {e}, using cosine seeds only")

            # ---- Stage 3: λ * normalized_UCB + (1-λ) * cosine → top-k ----
            candidate_list = list(candidates)

            # Get raw scores
            raw_ucb = np.array([self._node_ucb(idx) for idx in candidate_list])
            raw_cosine = np.array([sims[idx] for idx in candidate_list])

            # Normalize both to [0, 1]
            ucb_min, ucb_max = raw_ucb.min(), raw_ucb.max()
            if ucb_max > ucb_min:
                norm_ucb = (raw_ucb - ucb_min) / (ucb_max - ucb_min)
            else:
                norm_ucb = np.ones_like(raw_ucb) * 0.5

            cos_min, cos_max = raw_cosine.min(), raw_cosine.max()
            if cos_max > cos_min:
                norm_cosine = (raw_cosine - cos_min) / (cos_max - cos_min)
            else:
                norm_cosine = np.ones_like(raw_cosine) * 0.5

            # Combined score
            combined = lam * norm_ucb + (1 - lam) * norm_cosine
            top_indices_in_candidates = np.argsort(-combined)[:min(top_k, len(candidate_list))]
            selected_indices = [candidate_list[i] for i in top_indices_in_candidates]

            self._record_retrievals(selected_indices)
            selected_texts = [self.raw_texts[idx] for idx in selected_indices]

        return selected_texts, selected_indices

    def save(self, path: str):
        with self.lock:
            data = {
                "raw_texts": self.raw_texts,
                "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
                "edge_list": self.edge_list,
                "num_nodes": self.num_nodes,
                "retrieval_counts": self.retrieval_counts,
                "avg_rewards": self.avg_rewards,
                "node_ages": self.node_ages,
                "global_step": self.global_step,
                "total_retrievals": self.total_retrievals,
            }
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)
            logger.info(f"[ExperienceGraph] Saved: {self.num_nodes} nodes (step {self.global_step})")

    def load(self, path: str):
        with self.lock:
            with open(path, "r") as f:
                data = json.load(f)
            self.raw_texts = data["raw_texts"]
            self.embeddings = np.array(data["embeddings"]) if data["embeddings"] else None
            self.edge_list = [tuple(e) for e in data["edge_list"]]
            self.num_nodes = data["num_nodes"]
            self.retrieval_counts = data.get("retrieval_counts", [1] * self.num_nodes)
            self.avg_rewards = data.get("avg_rewards", [0.5] * self.num_nodes)
            # Ensure no zero counts (for old graphs without optimistic init)
            self.retrieval_counts = [max(c, 1) for c in self.retrieval_counts]
            self.node_ages = data.get("node_ages", [0] * self.num_nodes)
            self.global_step = data.get("global_step", 0)
            self.total_retrievals = data.get("total_retrievals", 0)
            self._rebuild_edge_index()
            logger.info(f"[ExperienceGraph] Loaded: {self.num_nodes} nodes (step {self.global_step})")

    def get_stats(self) -> dict:
        return {"num_nodes": self.num_nodes, "num_edges": len(self.edge_list) // 2, "max_nodes": self.max_nodes}


# ============================================================================
# FastAPI Models & Global State
# ============================================================================

class QueryRequest(BaseModel):
    queries: List[str]
    R_values: Optional[List[float]] = None
    W_values: Optional[List[float]] = None
    topk: Optional[int] = 10
    return_scores: bool = False

class AddExperienceRequest(BaseModel):
    experiences: List[str]

class UpdateRewardsRequest(BaseModel):
    node_indices: List[List[int]]  # per-episode list of retrieved node indices
    rewards: List[float]           # per-episode reward

class SaveRequest(BaseModel):
    path: str

# Global graph instance
graph: Optional[ExperienceGraph] = None


def parse_query(raw_query: str):
    """Parse 'R:50 query text' format. Falls back to R=50 if no prefix."""
    raw_query = raw_query.strip()
    if raw_query.startswith("R:"):
        parts = raw_query.split(" ", 1)
        try:
            R = float(parts[0][2:])
        except ValueError:
            R = 50.0
        query = parts[1] if len(parts) > 1 else ""
    else:
        R = 50.0
        query = raw_query
    return R, query


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Batched graph retrieval used by the copilot rollout.
    R/W are passed explicitly via R_values / W_values (or as an 'R:value' query prefix).
    """
    topk = request.topk or 3
    results = []

    # Batch encode all queries at once (Contriever is the bottleneck)
    parsed = []
    for i, raw_query in enumerate(request.queries):
        if request.R_values and i < len(request.R_values):
            R = request.R_values[i]
            query = raw_query
        else:
            R, query = parse_query(raw_query)
        W = request.W_values[i] if request.W_values and i < len(request.W_values) else 50.0
        parsed.append((R, W, query))

    # Encode all queries in one batch
    all_queries = [p[2] for p in parsed]
    all_embeddings = graph.retriever.encode_texts(all_queries) if all_queries else []

    all_node_indices = []
    for i, (R, W, query) in enumerate(parsed):
        query_embedding = all_embeddings[i] if i < len(all_embeddings) else None
        texts, node_indices = graph.retrieve(query, R=R, top_k=topk, W=W, query_embedding=query_embedding)
        all_node_indices.append(node_indices)

        query_results = []
        for idx, skill_text in enumerate(texts):
            doc = {
                "document": {
                    "title": f"Experience {idx + 1}",
                    "text": skill_text,
                    "contents": f"Experience {idx + 1}\n{skill_text}"
                },
                "score": 1.0 - idx * 0.1  # Descending pseudo-score
            }
            if request.return_scores:
                query_results.append(doc)
            else:
                query_results.append(doc["document"])

        # If no results, return empty experiences
        if not query_results:
            empty_doc = {
                "document": {
                    "title": "No Experience Found",
                    "text": "No relevant experience available yet.",
                    "contents": "No Experience Found\nNo relevant experience available yet."
                },
                "score": 0.0
            }
            if request.return_scores:
                query_results.append(empty_doc)
            else:
                query_results.append(empty_doc["document"])

        results.append(query_results)

    return {"result": results, "node_indices": all_node_indices}


@app.post("/add_experience")
def add_experience_endpoint(request: AddExperienceRequest):
    """Add new experiences to the graph."""
    if not request.experiences:
        return {"status": "no experiences to add"}
    new_ids = graph.add_nodes(request.experiences)
    return {"status": "ok", "added": len(new_ids), "total_nodes": graph.num_nodes}


@app.post("/naive_retrieve")
def naive_retrieve_endpoint(request: QueryRequest):
    """Pure cosine similarity retrieval (no diffusion, no utility). Used for ablations."""
    topk = request.topk or 5
    results = []
    for query in request.queries:
        skills = graph.naive_retrieve(query, top_k=topk)
        query_results = []
        for idx, skill_text in enumerate(skills):
            query_results.append({
                "title": f"Experience {idx + 1}",
                "text": skill_text,
                "contents": f"Experience {idx + 1}\n{skill_text}"
            })
        if not query_results:
            query_results.append({
                "title": "No Experience Found",
                "text": "No relevant experience available yet.",
                "contents": "No Experience Found\nNo relevant experience available yet."
            })
        results.append(query_results)
    return {"result": results}


@app.post("/update_rewards")
def update_rewards_endpoint(request: UpdateRewardsRequest):
    """Update bandit rewards for retrieved nodes after episodes complete."""
    updated = 0
    for node_indices, reward in zip(request.node_indices, request.rewards):
        graph.update_rewards(node_indices, reward)
        updated += len(node_indices)
    graph.increment_ages()
    return {"status": "ok", "updated_nodes": updated, "global_step": graph.global_step}


@app.post("/save")
def save_endpoint(request: SaveRequest):
    """Save graph to disk."""
    graph.save(request.path)
    return {"status": "ok", "path": request.path}


@app.get("/stats")
def stats_endpoint():
    """Get graph stats."""
    stats = graph.get_stats()
    stats["global_step"] = graph.global_step
    stats["total_retrievals"] = graph.total_retrievals
    return stats


# ============================================================================
# Main
# ============================================================================

def main():
    global graph

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--graph_path", type=str, default=None)
    parser.add_argument("--contriever_path", type=str, default="facebook/contriever")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_nodes", type=int, default=2000)
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--sim_threshold", type=float, default=0.3)
    args = parser.parse_args()

    retriever = ContrieverRetriever(model_name=args.contriever_path, device=args.device)
    graph = ExperienceGraph(
        retriever=retriever,
        k_neighbors=args.k_neighbors,
        sim_threshold=args.sim_threshold,
        max_nodes=args.max_nodes,
    )

    if args.graph_path and os.path.exists(args.graph_path):
        graph.load(args.graph_path)
        logger.info(f"Loaded graph from {args.graph_path}")
    else:
        logger.info("Starting with empty graph")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
