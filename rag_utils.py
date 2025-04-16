import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple, Dict

class RAGProcessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.qa_pairs = []
        self.dimension = 384  # MiniLM-L6-v2 embedding dimension
        
    def load_text(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        qa_blocks = content.split('\n\n')
        for block in qa_blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                question = lines[0].strip()
                answer = ' '.join(lines[1:]).strip()
                if question.startswith('Q:') and answer.startswith('A:'):
                    self.qa_pairs.append({
                        'question': question[2:].strip(),
                        'answer': answer[2:].strip()
                    })
        
        self.texts = [
            f"Question: {qa['question']} Answer: {qa['answer']}"
            for qa in self.qa_pairs
        ]
        
        self._build_index()
    
    def _build_index(self):
        if not self.texts:
            raise ValueError("No texts loaded. Call load_text() first.")
        
        # creating embeddings
        embeddings = self.model.encode(self.texts)
        
        # initializing FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # adding vectors to the index
        self.index.add(np.array(embeddings).astype('float32'))
    
    def query(self, query_text: str, num_results: int = 3) -> Tuple[List[Dict[str, str]], List[float]]:
        if not self.index:
            raise ValueError("Index not built. Call load_text() first.")
        
        # getting query embedding
        query_embedding = self.model.encode([query_text])
        
        # searcing in FAISS
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            num_results
        )
        
        # converting distances to similarity scores with euclidean distance
        scores = [1 / (1 + d) for d in distances[0]]
        
        # getting relevant Q&A pairs
        results = [self.qa_pairs[i] for i in indices[0]]
        
        return results, scores 