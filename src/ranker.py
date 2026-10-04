import torch
from sentence_transformers import CrossEncoder

class Ranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Ranker with a Cross-Encoder model.
        
        Args:
            model_name (str): The name of the Cross-Encoder model.
        """
        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
            
        print(f"Loading Ranker model {model_name} on {self.device}...")
        self.model = CrossEncoder(model_name, device=self.device)
        print("Ranker model loaded.")

    def rank(self, query, candidates, top_k=12):
        """
        Re-rank the candidates using the Cross-Encoder.
        
        Args:
            query (str): The search query.
            candidates (list): List of dictionaries. Each dict must have 'text' key containing the content to rank.
                               It can also have other keys like 'id', 'metadata', etc. which will be preserved.
            top_k (int): Number of top results to return after re-ranking.
            
        Returns:
            list: Re-ranked list of candidates.
        """
        if not candidates:
            return []
            
        # Prepare pairs for the Cross-Encoder
        pairs = [[query, cand['text']] for cand in candidates]
        
        # Predict scores
        with torch.no_grad():
            scores = self.model.predict(pairs)
            
        # Attach scores to candidates
        for i, cand in enumerate(candidates):
            cand['score'] = float(scores[i])
            
        # Sort by score descending
        ranked_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        
        return ranked_candidates[:top_k]
