import os
from typing import List, Dict, Optional
from qdrant_client.models import PointStruct
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client import QdrantClient
import openai

class HierarchicalDataManager:

    def __init__(self, qdrant_client: QdrantClient):
        self.client = qdrant_client
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.hierarchy_collection = "hierarchy_tree"

        # Ensure the hierarchy collection exists
        try:
            self.client.get_collection(collection_name=self.hierarchy_collection)
        except Exception:
            # Create the hierarchy collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.hierarchy_collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),  # Dummy vector size
            )


    def classify_with_gpt(self, text: str, options: List[str], level: str) -> str:
        """
        Use GPT to classify text into one of the given options.
        """
        prompt = (
            f"Based on the following text, decide which {level} it belongs to:\n\n"
            f"Text: {text}\n\n"
            f"Options: {', '.join(options)}\n\n"
            f"Only return the name of the {level}."
        )
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=20,
            temperature=0.0
        )
        return response.choices[0].text.strip()

    def get_hierarchy_node(self, name: str, level: str) -> Optional[Dict]:
        """
        Retrieve a hierarchy node by its name and level.
        """
        results = self.client.search(
            collection_name=self.hierarchy_collection,
            query_vector=[0.0],  # Dummy query vector
            filter={
                "must": [
                    {"key": "name", "match": {"value": name}},
                    {"key": "type", "match": {"value": level}},
                ]
            },
            limit=1,
        )
        return results[0].payload if results else None

    def save_hierarchy_node(self, name: str, level: str, parent: Optional[str] = None, children: Optional[List[str]] = None):
        """
        Save or update a hierarchy node.
        """
        node = self.get_hierarchy_node(name, level)
        if node:
            # Update the existing node
            node["children"] = list(set(node.get("children", []) + (children or [])))
            self.client.upsert(
                collection_name=self.hierarchy_collection,
                points=[
                    PointStruct(
                        id=node["id"],
                        vector=[0.0],  # Dummy vector
                        payload=node,
                    )
                ],
            )
        else:
            # Create a new node
            self.client.upsert(
                collection_name=self.hierarchy_collection,
                points=[
                    PointStruct(
                        id=None,
                        vector=[0.0],  # Dummy vector
                        payload={
                            "type": level,
                            "name": name,
                            "parent": parent,
                            "children": children or [],
                        },
                    )
                ],
            )

    def save_data(self, document, vector: List[float]):
        """
        Save a document in the proper hierarchical structure.
        """
        document_text = ' '.join(document.text)

        # Step 1: Sector Classification
        sectors = [
            node["name"] for node in self.client.search(
                collection_name=self.hierarchy_collection,
                query_vector=[0.0],
                filter={"must": [{"key": "type", "match": {"value": "sector"}}]},
                limit=100,
            )
        ]
        sector = self.classify_with_gpt(document_text, sectors, "sector")
        self.save_hierarchy_node(name=sector, level="sector")

        # Step 2: Company/Subject Classification
        subjects = [
            node["name"] for node in self.client.search(
                collection_name=self.hierarchy_collection,
                query_vector=[0.0],
                filter={
                    "must": [
                        {"key": "type", "match": {"value": "subject"}},
                        {"key": "parent", "match": {"value": sector}},
                    ]
                },
                limit=100,
            )
        ]
        subject = self.classify_with_gpt(document_text, subjects, "subject")
        self.save_hierarchy_node(name=subject, level="subject", parent=sector)

        # Step 3: Event Type Classification
        event_types = [
            node["name"] for node in self.client.search(
                collection_name=self.hierarchy_collection,
                query_vector=[0.0],
                filter={
                    "must": [
                        {"key": "type", "match": {"value": "event_type"}},
                        {"key": "parent", "match": {"value": subject}},
                    ]
                },
                limit=100,
            )
        ]
        event_type = self.classify_with_gpt(document_text, event_types, "event type")
        self.save_hierarchy_node(name=event_type, level="event_type", parent=subject)

        # Step 4: Save the document in its specific Qdrant collection
        collection_name = f"{sector}_{subject}_{event_type}".lower().replace(" ", "_")
        if not self.client.get_collection(collection_name):
            self.client.create_collection(
                collection_name,
                vectors_config=VectorParams(size=len(vector), distance=Distance.COSINE),
            )
        ids, payloads = document.to_payloads()
        points = [
            PointStruct(id=idx, vector=vector, payload=_payload)
            for idx, vector, _payload in zip(ids, document.embeddings, payloads)
        ]

        self.client.upsert(collection_name=collection_name, points=points)



