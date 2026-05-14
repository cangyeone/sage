"""
A simple vector database implementation using TF-IDF and cosine similarity.
Designed as a fallback when complex embedding models are not available.
"""
import re
import math
import pickle
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional
from sage_paths import sage_home

_DB_PATH = sage_home("knowledge", "simple_vectordb.pkl")


class VectorItem:
    """
    Represents an item in the vector database.
    """
    def __init__(self, text: str, metadata: Optional[Dict] = None, doc_id: str = None):
        self.text = text
        self.metadata = metadata or {}
        self.doc_id = doc_id


class SimpleVectorDB:
    """
    A simple vector database using TF-IDF and cosine similarity.
    """
    
    def __init__(self):
        self.items: List[VectorItem] = []
        self.vocabulary: Dict[str, int] = {}  # word -> index
        self.idf_values: List[float] = []     # IDF value for each word
        self.doc_vectors: List[List[float]] = []  # TF-IDF vectors for each document
        self.vocab_to_idx: Dict[str, int] = {}  # To map words to indices during processing

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenization supporting mixed Chinese-English text.
        Splits on whitespace/punctuation AND on ASCII↔Unicode boundaries,
        so "SVM方法" → ["svm", "方法"], avoiding long hybrid tokens.
        """
        # Split English alphanumeric runs and Chinese character runs separately
        tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff\u3400-\u4dbf]+', text.lower())
        return [t for t in tokens if len(t) > 1]

    # ── 持久化 ─────────────────────────────────────────────────────────────────

    def save(self, path: Path = _DB_PATH):
        """将全部 item 序列化到磁盘（pickle）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.items, f)

    def load(self, path: Path = _DB_PATH):
        """从磁盘恢复 item，并重建 TF-IDF 向量。"""
        if not path.exists():
            return
        try:
            with open(path, 'rb') as f:
                self.items = pickle.load(f)
            if self.items:
                self._update_vocabulary_and_vectors()
        except Exception as e:
            print(f"SimpleVectorDB load error: {e}")
            self.items = []

    def add_item(self, text: str, metadata: Optional[Dict] = None, doc_id: str = None):
        """
        Add an item to the database.
        """
        item = VectorItem(text, metadata, doc_id)
        self.items.append(item)

        # Update vocabulary and document vectors
        self._update_vocabulary_and_vectors()
    
    def count_items(self) -> int:
        """
        Return the number of items in the database.
        """
        return len(self.items)

    def remove_items_by_doc_id(self, doc_id: str):
        """
        Remove all items associated with a specific document ID.
        """
        # Filter out items with the specified doc_id
        self.items = [item for item in self.items if item.doc_id != doc_id]

        # Update vocabulary and document vectors after removal
        self._update_vocabulary_and_vectors()
        self.save()

    def _update_vocabulary_and_vectors(self):
        """
        Update vocabulary and compute TF-IDF vectors for all documents.
        """
        # Reset vocabulary
        self.vocabulary = {}
        all_tokens = []
        
        # Collect all unique tokens
        for i, item in enumerate(self.items):
            tokens = self.tokenize(item.text)
            all_tokens.append(tokens)
            
            # Add unique tokens to vocabulary
            for token in set(tokens):
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        
        # Create a mapping from vocab to index for quick lookups
        self.vocab_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        
        # Compute IDF values
        N = len(self.items)
        idf = [0.0] * len(self.vocabulary)
        
        for word, idx in self.vocabulary.items():
            df = sum(1 for tokens in all_tokens if word in tokens)
            if df > 0:
                idf[idx] = math.log(N / df)  # IDF formula
        
        self.idf_values = idf
        
        # Compute TF-IDF vectors for each document
        self.doc_vectors = []
        for tokens in all_tokens:
            tfidf_vector = [0.0] * len(self.vocabulary)
            counter = Counter(tokens)
            doc_len = len(tokens)
            
            for token, freq in counter.items():
                if token in self.vocab_to_idx:
                    idx = self.vocab_to_idx[token]
                    tf = freq / doc_len if doc_len > 0 else 0  # TF formula
                    tfidf_vector[idx] = tf * self.idf_values[idx]
            
            # Normalize the vector
            norm = math.sqrt(sum(x*x for x in tfidf_vector))
            if norm > 0:
                tfidf_vector = [x/norm for x in tfidf_vector]
            
            self.doc_vectors.append(tfidf_vector)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[VectorItem, float]]:
        """
        Search for top_k most similar items to the query.
        Returns a list of (VectorItem, similarity_score) tuples.
        """
        if not self.items:
            return []
        
        # Tokenize query and create TF vector
        query_tokens = self.tokenize(query)
        query_tf = Counter(query_tokens)
        query_len = len(query_tokens)
        
        # Create TF-IDF vector for query
        query_tfidf = [0.0] * len(self.vocabulary)
        for token, freq in query_tf.items():
            if token in self.vocab_to_idx:
                idx = self.vocab_to_idx[token]
                tf = freq / query_len if query_len > 0 else 0
                query_tfidf[idx] = tf * self.idf_values[idx]
        
        # Normalize query vector
        norm = math.sqrt(sum(x*x for x in query_tfidf))
        if norm > 0:
            query_tfidf = [x/norm for x in query_tfidf]
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_vector in enumerate(self.doc_vectors):
            similarity = sum(a * b for a, b in zip(query_tfidf, doc_vector))
            similarities.append((i, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:top_k]
        
        # Return VectorItem with similarity score
        return [(self.items[i], score) for i, score in top_similarities]
    
    def clear(self):
        """
        Clear the database.
        """
        self.items = []
        self.vocabulary = {}
        self.idf_values = []
        self.doc_vectors = []
        self.vocab_to_idx = {}
        try:
            _DB_PATH.unlink(missing_ok=True)
        except Exception:
            pass


# Global instance
_vector_db = None


def get_simple_vector_db() -> SimpleVectorDB:
    """
    Get the global instance of the simple vector database.
    首次调用时自动从磁盘恢复数据。
    """
    global _vector_db
    if _vector_db is None:
        _vector_db = SimpleVectorDB()
        _vector_db.load()   # 恢复持久化数据
    return _vector_db
