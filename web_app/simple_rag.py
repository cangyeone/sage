"""
Simple RAG implementation that works as a fallback when complex embeddings fail.
This module provides basic document retrieval functionality without requiring 
complex embedding models.
"""
from typing import List, Tuple
import re
import json
import os
from pathlib import Path
from .simple_vector_db import get_simple_vector_db, VectorItem

# Define DocChunk and DocMeta classes to match rag_engine structure
class DocChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str
    char_start: int = 0
    
    def __init__(self, chunk_id, doc_id, doc_name, page, text, char_start=0):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.page = page
        self.text = text
        self.char_start = char_start

class DocMeta:
    def __init__(self, doc_id, doc_name, file_path, n_pages, n_chunks, added_at, size_bytes):
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.file_path = file_path
        self.n_pages = n_pages
        self.n_chunks = n_chunks
        self.added_at = added_at
        self.size_bytes = size_bytes

# Define constants to match rag_engine
KB_DIR = Path(__file__).parent.parent / "seismo_rag"
SIMPLE_META_FILE = KB_DIR / "simple_meta.json"


class SimpleRAG:
    """
    Simplified RAG implementation that uses basic text matching
    when advanced embeddings are not available.
    """
    
    def __init__(self):
        self.db = get_simple_vector_db()
        self._docs = {}
        self._load_docs()
    
    def _load_docs(self):
        """Load document metadata from disk."""
        if SIMPLE_META_FILE.exists():
            try:
                with SIMPLE_META_FILE.open('r', encoding='utf-8') as f:
                    raw = json.load(f)
                    for doc_id, doc_data in raw.items():
                        self._docs[doc_id] = DocMeta(
                            doc_id=doc_data['doc_id'],
                            doc_name=doc_data['doc_name'],
                            file_path=doc_data['file_path'],
                            n_pages=doc_data['n_pages'],
                            n_chunks=doc_data['n_chunks'],
                            added_at=doc_data['added_at'],
                            size_bytes=doc_data['size_bytes']
                        )
            except Exception as e:
                print(f"Error loading simple metadata: {e}")
    
    def save_docs(self):
        """Save document metadata to disk."""
        try:
            meta_dict = {}
            for doc_id, doc_meta in self._docs.items():
                meta_dict[doc_id] = {
                    'doc_id': doc_meta.doc_id,
                    'doc_name': doc_meta.doc_name,
                    'file_path': doc_meta.file_path,
                    'n_pages': doc_meta.n_pages,
                    'n_chunks': doc_meta.n_chunks,
                    'added_at': doc_meta.added_at,
                    'size_bytes': doc_meta.size_bytes
                }
            
            with SIMPLE_META_FILE.open('w', encoding='utf-8') as f:
                json.dump(meta_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving simple metadata: {e}")
    
    def add_document(self, chunks: List[DocChunk], doc_metadata: dict = None):
        """
        Add a document (as a list of chunks) to the knowledge base.
        """
        doc_id = doc_metadata.get("doc_id", "unknown") if doc_metadata else "unknown"
        doc_name = doc_metadata.get("doc_name", "unknown") if doc_metadata else "unknown"
        
        for chunk in chunks:
            chunk_meta = {
                "doc_name":    doc_name,
                "doc_id":      doc_id,
                "page":        chunk.page,
                "chunk_id":    chunk.chunk_id,
                "chunk_index": int(chunk.chunk_id.split('_')[-1]) if '_' in chunk.chunk_id else 0,
            }
            
            self.db.add_item(
                text=chunk.text,
                metadata=chunk_meta,
                doc_id=doc_id  # Pass the doc_id to the vector database
            )
        
        # Save document metadata and persist vectors to disk
        if doc_metadata:
            self._docs[doc_metadata['doc_id']] = DocMeta(
                doc_id=doc_metadata['doc_id'],
                doc_name=doc_metadata['doc_name'],
                file_path=doc_metadata['file_path'],
                n_pages=doc_metadata['n_pages'],
                n_chunks=doc_metadata['n_chunks'],
                added_at=doc_metadata['added_at'],
                size_bytes=doc_metadata['size_bytes']
            )
            self.save_docs()
            self.db.save()   # 持久化向量到磁盘，重启后自动恢复
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        """
        # Clean up the text a bit
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings near the boundary
                snippet = text[start:end]
                last_period = snippet.rfind('.')
                last_exclamation = snippet.rfind('!')
                last_question = snippet.rfind('?')
                
                break_point = max(last_period, last_exclamation, last_question)
                
                if break_point > len(snippet) // 2:  # Only break at sentence if it's reasonably close
                    end = start + break_point + 1
                else:
                    # Otherwise look for word boundaries
                    last_space = snippet.rfind(' ')
                    if last_space > len(snippet) // 2:
                        end = start + last_space
            
            chunks.append(text[start:end])
            start = end - overlap  # Overlap
            
            # Handle edge case where remaining text is shorter than overlap
            if len(text) - start < overlap:
                start = len(text)
        
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 20]
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Retrieve the most relevant chunks for a given query.
        Returns a list of (text, relevance_score, metadata) tuples.
        """
        results = self.db.search(query, top_k=top_k)
        return [(item.text, score, item.metadata) for item, score in results]
    
    def build_context(self, query: str, top_k: int = 5, max_chars: int = 3000) -> str:
        """
        Build a context string from retrieved documents.
        """
        hits = self.retrieve(query, top_k=top_k)
        
        if not hits:
            return ""
        
        lines = ["以下是从知识库中检索到的相关内容，请参考这些内容回答用户问题：\n"]
        total_chars = 0
        
        for text, score, metadata in hits:
            # Format the chunk with metadata
            doc_name = metadata.get("doc_name", "Unknown document")
            page_num = metadata.get("page", "Unknown")
            
            entry = f"【来源：{doc_name}，第 {page_num} 页，相关度 {score:.2f}】\n{text}\n"
            
            # Check if adding this entry would exceed the limit
            if total_chars + len(entry) > max_chars:
                break
                
            lines.append(entry)
            total_chars += len(entry)
        
        return "\n".join(lines)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the knowledge base.
        """
        if doc_id not in self._docs:
            return False
            
        # Remove from docs
        del self._docs[doc_id]
        
        # Remove from vector database
        self.db.remove_items_by_doc_id(doc_id)
        
        # Update metadata file
        self.save_docs()
        
        return True
    
    def clear(self):
        """
        Clear the knowledge base.
        """
        self.db.clear()
        self._docs.clear()
        if SIMPLE_META_FILE.exists():
            SIMPLE_META_FILE.unlink()  # Remove metadata file


# Global instance
_simple_rag = None


def get_simple_rag() -> SimpleRAG:
    """
    Get the global instance of the simple RAG system.
    """
    global _simple_rag
    if _simple_rag is None:
        _simple_rag = SimpleRAG()
    return _simple_rag