
import numpy as np
import faiss
import pandas as pd

class KnowledgeBaseIndex:
    """Semantic search over the defect knowledge base using sentence embeddings."""
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.index = None
        self.kb_df = None

    def build(self, kb_df):
        self.kb_df = kb_df.reset_index(drop=True)
        texts = [
            f"{row.defect_type}: {row.cause} {row.impact} {row.solution} {row.prevention}"
            for row in self.kb_df.itertuples()
        ]
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, k=3):
        query_emb = self.embed_model.encode([query], convert_to_numpy=True)
        query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
        distances, indices = self.index.search(query_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            row = self.kb_df.iloc[idx].to_dict()
            row["relevance_distance"] = float(dist)
            results.append(row)
        return results

    def lookup_exact(self, defect_type):
        """Direct lookup when the defect_type is already known (e.g. from a
        specific inspection result) — more reliable than semantic search
        when the exact key is available."""
        match = self.kb_df[self.kb_df["defect_type"] == defect_type]
        if len(match) > 0:
            return match.iloc[0].to_dict()
        return None


def build_rag_prompt(user_question, retrieved_entries, inspection_context=None):
    """Combines retrieved KB entries (+ optional specific inspection context)
    into a grounded prompt for the LLM."""
    context_lines = []
    for i, entry in enumerate(retrieved_entries, 1):
        context_lines.append(
            f"""[{i}] Defect type: {entry['defect_type']}
    Cause: {entry['cause']}
    Impact: {entry['impact']}
    Solution: {entry['solution']}
    Prevention: {entry['prevention']} """
        )
    context_block = "\n\n".join(context_lines)

    inspection_block = ""
    if inspection_context:
        inspection_block = (
            f"""
Current inspection result:
Category: {inspection_context.get('category')}
Severity: {inspection_context.get('severity_level')} ({inspection_context.get('severity_score')}/100)
"""
        )

    system_msg = (
        "You are a factory quality-control knowledge assistant. Answer the "
        "operator's question using ONLY the retrieved knowledge base entries "
        "below. Do not invent causes, statistics, or facts not present in the "
        "provided context. If the retrieved entries do not fully answer the "
        "question, say what is known and note what is uncertain. Keep the "
        "answer to 3-4 sentences."
    )
    user_msg = (
        f"Retrieved knowledge base entries:\n\n{context_block}"
        f"{inspection_block}\n\nOperator question: {user_question}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
