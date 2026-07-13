# phoenix-llm34/rag_system.py
"""
Système RAG (Retrieval Augmented Generation) avancé
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import threading
import pickle

class Document:
    """Document pour le système RAG"""
    
    def __init__(self, content: str, metadata: Dict = None):
        self.id = hashlib.md5(content.encode()).hexdigest()
        self.content = content
        self.metadata = metadata or {}
        self.tokens = self._tokenize(content)
        self.embeddings = None
        self.timestamp = datetime.now().isoformat()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenisation simple"""
        # Nettoyer le texte
        text = text.lower()
        # Enlever la ponctuation et tokenizer
        import re
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "tokens": self.tokens,
            "timestamp": self.timestamp
        }

class RAGSystem:
    """Système RAG complet"""
    
    def __init__(self, config):
        self.config = config
        self.rag_dir = config.rag_dir
        self.documents: Dict[str, Document] = {}
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        
        self._load_index()
    
    def _load_index(self):
        """Charge l'index depuis le disque"""
        index_file = self.rag_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    for doc_data in data.get("documents", []):
                        doc = Document(
                            doc_data["content"],
                            doc_data["metadata"]
                        )
                        doc.id = doc_data["id"]
                        doc.timestamp = doc_data["timestamp"]
                        self.documents[doc.id] = doc
                        for token in doc.tokens:
                            self.index[token].append(doc.id)
            except Exception as e:
                print(f"Erreur chargement index: {e}")
    
    def save_index(self):
        """Sauvegarde l'index"""
        index_file = self.rag_dir / "index.json"
        data = {
            "documents": [doc.to_dict() for doc in self.documents.values()],
            "updated_at": datetime.now().isoformat()
        }
        with open(index_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_document(self, content: str, metadata: Dict = None):
        """Ajoute un document au système"""
        with self.lock:
            doc = Document(content, metadata)
            self.documents[doc.id] = doc
            for token in doc.tokens:
                self.index[token].append(doc.id)
            self.save_index()
            return doc.id
    
    def add_documents(self, documents: List[Dict]):
        """Ajoute plusieurs documents"""
        for doc_data in documents:
            self.add_document(
                doc_data.get("content", ""),
                doc_data.get("metadata")
            )
    
    def remove_document(self, doc_id: str):
        """Supprime un document"""
        with self.lock:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                for token in doc.tokens:
                    if doc_id in self.index[token]:
                        self.index[token].remove(doc_id)
                del self.documents[doc_id]
                self.save_index()
                return True
        return False
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Recherche les documents pertinents"""
        query_tokens = self._tokenize_query(query)
        
        # Calculer les scores
        scores = defaultdict(float)
        for token in query_tokens:
            if token in self.index:
                for doc_id in self.index[token]:
                    # Score basé sur la fréquence
                    tf = self._get_tf(token, self.documents[doc_id])
                    idf = self._get_idf(token)
                    scores[doc_id] += tf * idf
        
        # Trier par score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = self.documents[doc_id]
            doc.score = score  # Ajouter le score pour référence
            results.append(doc)
        
        return results
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenise une requête"""
        import re
        tokens = re.findall(r'\b\w+\b', query.lower())
        # Enlever les stop words communs
        stop_words = {'le', 'la', 'les', 'des', 'et', 'ou', 'pour', 'dans', 'avec', 'sur'}
        tokens = [t for t in tokens if t not in stop_words]
        return tokens
    
    def _get_tf(self, token: str, document: Document) -> float:
        """Calcule la fréquence du terme dans le document"""
        if document:
            return document.tokens.count(token) / len(document.tokens)
        return 0
    
    def _get_idf(self, token: str) -> float:
        """Calcule l'inverse de la fréquence du document"""
        if token in self.index:
            return np.log(len(self.documents) / len(self.index[token]))
        return 0
    
    def generate_context(self, query: str, max_length: int = 1000) -> str:
        """Génère un contexte à partir des documents pertinents"""
        docs = self.search(query, top_k=3)
        if not docs:
            return ""
        
        context = "Contexte:\n"
        for i, doc in enumerate(docs, 1):
            # Prendre les parties pertinentes
            content = doc.content[:max_length]
            context += f"\nDocument {i}:\n{content}\n"
            
        return context
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du système"""
        return {
            "total_documents": len(self.documents),
            "unique_tokens": len(self.index),
            "total_tokens": sum(len(doc.tokens) for doc in self.documents.values()),
            "documents": [{
                "id": doc.id,
                "length": len(doc.content),
                "tokens": len(doc.tokens)
            } for doc in self.documents.values()]
        }