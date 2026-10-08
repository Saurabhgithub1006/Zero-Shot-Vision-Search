#Indexer class to index embeddings in Pinecone

import os
import time
from pinecone import Pinecone, ServerlessSpec

from src.config import get_settings

class Indexer:
    def __init__(self, index_name=None, namespace=None, dimension=None, metric="cosine"):
        """
        Initialize Pinecone Indexer.
        
        Args:
            index_name (str): Name of the index. Defaults to the configured index.
            namespace (str): Namespace inside the index, one per dataset. Defaults to the configured dataset.
            dimension (int): Dimension of the vectors. Defaults to the configured embedding size.
            metric (str): Metric for similarity search.
        """
        settings = get_settings()
        self.api_key = os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set.")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name or settings.pinecone_index
        self.namespace = namespace or settings.dataset
        self.dimension = dimension or settings.embedding_dim
        self.metric = metric
        self.index = None
        
        self._initialize_index()

    def _initialize_index(self):
        """Create index if it doesn't exist and connect to it."""
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print(f"Index '{self.index_name}' created.")
        else:
            print(f"Index '{self.index_name}' already exists.")
            
        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, vectors, batch_size=100):
        """
        Upsert vectors to Pinecone.
        
        Args:
            vectors (list): List of tuples (id, vector, metadata).
            batch_size (int): Number of vectors to upsert in a single batch.
        """
        total_vectors = len(vectors)
        print(f"Upserting {total_vectors} vectors...")
        
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            print(f"Upserted batch {i // batch_size + 1}/{(total_vectors + batch_size - 1) // batch_size}")
            
    def search(self, vector, top_k=5, metadata_filter=None):
        """
        Search the Pinecone index.
        
        Args:
            vector (list): Query vector.
            top_k (int): Number of results to return.
            metadata_filter (dict): Optional Pinecone metadata filter.
            
        Returns:
            dict: Query results.
        """
        return self.index.query(
            vector=vector, top_k=top_k, include_metadata=True,
            namespace=self.namespace, filter=metadata_filter,
        )

    def fetch_vectors(self, ids):
        """
        Fetch vectors by ID to check existence.
        
        Args:
            ids (list): List of vector IDs.
            
        Returns:
            dict: Dictionary containing the fetched vectors.
        """
        return self.index.fetch(ids=ids, namespace=self.namespace)

    def delete_index(self):
        """Delete the index."""
        if self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted.")
