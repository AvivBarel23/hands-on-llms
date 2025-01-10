import os
from typing import List, Dict, Optional

import openai
from bytewax.outputs import DynamicOutput, StatelessSink
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff
from qdrant_client.models import PointStruct
# Check which ScoredPoint import actually works for your Qdrant version
from qdrant_client.models import ScoredPoint
from qdrant_client.conversions.common_types import ScoredPoint

from streaming_pipeline import constants
from streaming_pipeline.models import Document


class HierarchicalDataManager:
    def __init__(self, qdrant_client: QdrantClient):
        print("[DEBUG] HierarchicalDataManager.__init__ called")
        self.client = qdrant_client
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.hierarchy_collection = "hierarchy_tree"

        # Ensure the hierarchy collection exists
        try:
            print(f"[DEBUG] Checking collection: {self.hierarchy_collection}")
            self.client.get_collection(collection_name=self.hierarchy_collection)
            print("[DEBUG] Hierarchy collection exists.")
        except Exception:
            print("[DEBUG] Hierarchy collection does NOT exist; creating it.")
            self.client.create_collection(
                collection_name=self.hierarchy_collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )


    def classify_with_gpt(self, text: str, options: List[str], level: str) -> str:
        # TODO: What happens when there is no match (like at the beginning when the sectors are empty)
        print("[DEBUG] classify_with_gpt called")
        print(f"[DEBUG] text='{text[:50]}...' options={options} level={level}")

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
        classification = response.choices[0].text.strip().replace(".", "")
        print(f"[DEBUG] GPT classification result: {classification}")

        if classification not in options:
            print(f"[DEBUG] classification '{classification}' not in existing options => treating as new")
            return classification
        return classification


    def get_hierarchy_node(self, name: str, level: str) -> Optional[ScoredPoint]:
        print(f"[DEBUG] get_hierarchy_node called with name='{name}', level='{level}'")
        results = self.client.search(
            collection_name=self.hierarchy_collection,
            query_vector=[1],  # Dummy query vector
            filter={
                "must": [
                    {"key": "name", "match": {"value": name}},
                    {"key": "type", "match": {"value": level}},
                ]
            },
            limit=1,
        )
        if results:
            print("[DEBUG] Found matching node(s). Returning the first one.")
        else:
            print("[DEBUG] No matching node found.")
        return results[0] if results else None


    def save_hierarchy_node(self, name: str, level: str, parent: Optional[str] = None, children: Optional[List[str]] = None):
        """
        Save or update a hierarchy node.
        """
        print(f"[DEBUG] save_hierarchy_node called with name='{name}', level='{level}', parent='{parent}', children={children}")
        node = self.get_hierarchy_node(name, level)
        if node:
            print("[DEBUG] Node exists; updating existing node.")
            # node.payload is the existing payload
            payload = node.payload
            payload["children"] = list(set(payload.get("children", []) + (children or [])))
            self.client.upsert(
                collection_name=self.hierarchy_collection,
                points=[
                    PointStruct(
                        id=node.id,  # use the actual scored point ID
                        vector=[0],
                        payload=payload,
                    )
                ],
            )
        else:
            print("[DEBUG] Node does not exist; creating new node.")
            self.client.upsert(
                collection_name=self.hierarchy_collection,
                points=[
                    PointStruct(
                        id=None,
                        vector=[0],  # Dummy vector
                        payload={
                            "type": level,
                            "name": name,
                            "parent": parent,
                            "children": children or [],
                        },
                    )
                ],
            )


    def save_data(self, document):
        """
        Save a document in the proper hierarchical structure.
        """
        print("[DEBUG] save_data called.")
        document_text = ' '.join(document.text)
        print("[DEBUG] Full document text:", document_text[:100], "...")

        # Step 1: Sector Classification
        sectors = [
            node["name"] for node in self.client.search(
                collection_name=self.hierarchy_collection,
                query_vector=[1.0],
                filter={"must": [{"key": "type", "match": {"value": "sector"}}]}
            )
        ]
        print("[DEBUG] Found existing sectors:", sectors)
        sector = self.classify_with_gpt(document_text, sectors, "sector")
        print(f"[DEBUG] sector => '{sector}'")
        # self.save_hierarchy_node(name=sector, level="sector")

        # Step 2: Company/Subject Classification
        subjects = [
            node["name"] for node in self.client.search(
                collection_name=self.hierarchy_collection,
                query_vector=[1.0],
                filter={
                    "must": [
                        {"key": "type", "match": {"value": "subject"}},
                        {"key": "parent", "match": {"value": sector}},
                    ]
                }
            )
        ]
        print("[DEBUG] Found existing subjects under sector:", subjects)
        subject = self.classify_with_gpt(document_text, subjects, "subject")
        print(f"[DEBUG] subject => '{subject}'")
        # self.save_hierarchy_node(name=subject, level="subject", parent=sector)

        # Step 3: Event Type Classification
        event_types = [
            node["name"] for node in self.client.search(
                collection_name=self.hierarchy_collection,
                query_vector=[1.0],
                filter={
                    "must": [
                        {"key": "type", "match": {"value": "event_type"}},
                        {"key": "parent", "match": {"value": subject}},
                    ]
                }
            )
        ]
        print("[DEBUG] Found existing event_types under subject:", event_types)
        event_type = self.classify_with_gpt(document_text, event_types, "event type")
        print(f"[DEBUG] event_type => '{event_type}'")
        # self.save_hierarchy_node(name=event_type, level="event_type", parent=subject)

        # Step 4: Save the document in its specific Qdrant collection
        collection_name = f"{sector}_{subject}_{event_type}".lower().replace(" ", "_")
        print(f"[DEBUG] Final collection_name => '{collection_name}'")

        if not self.client.get_collection(collection_name):
            print(f"[DEBUG] Collection '{collection_name}' does NOT exist; creating.")
            vector_size = len(document.embeddings[0]) if document.embeddings else 768
            self.client.create_collection(
                collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print("[DEBUG] Created new collection with vector_size=", vector_size)
            # TODO: add to the final tree!

        print("[DEBUG] Upserting the document's embeddings...")
        ids, payloads = document.to_payloads()
        points = [
            PointStruct(id=idx, vector=vector, payload=_payload)
            for idx, vector, _payload in zip(ids, document.embeddings, payloads)
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        print("[DEBUG] Document saved successfully in", collection_name)


class QdrantVectorOutput(DynamicOutput):
    """
    A class representing a Qdrant vector output,
    for at-least-once message processing.
    """

    def __init__(
        self,
        vector_size: int,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        client: Optional[QdrantClient] = None,
    ):
        print("[DEBUG] QdrantVectorOutput.__init__ called")
        self._collection_name = collection_name
        self._vector_size = vector_size

        if client:
            self.client = client
        else:
            self.client = build_qdrant_client()

        try:
            print(f"[DEBUG] Checking collection: {self._collection_name}")
            self.client.get_collection(collection_name=self._collection_name)
            print("[DEBUG] Collection exists. Will not recreate.")
        except (UnexpectedResponse, ValueError):
            print("[DEBUG] Collection does not exist. Recreating from scratch.")
            self.client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
                optimizers_config=OptimizersConfigDiff(max_optimization_threads=1),
            )

    def build(self, worker_index, worker_count):
        """
        Called by Bytewax to build a sink for each worker.
        Returns a QdrantVectorSink instance.
        """
        print(f"[DEBUG] QdrantVectorOutput.build called on worker {worker_index} / {worker_count}")
        return QdrantVectorSink(self.client, self._collection_name)


def build_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
    print("[DEBUG] build_qdrant_client called")
    if url is None:
        try:
            url = os.environ["QDRANT_URL"]
        except KeyError:
            raise KeyError(
                "QDRANT_URL must be set as environment variable or manually passed as an argument."
            )

    if api_key is None:
        try:
            api_key = os.environ["QDRANT_API_KEY"]
        except KeyError:
            raise KeyError(
                "QDRANT_API_KEY must be set as environment variable or manually passed as an argument."
            )

    client = QdrantClient(url, api_key=api_key)
    print("[DEBUG] QdrantClient built successfully")
    return client


class QdrantVectorSink(StatelessSink):
    """
    A sink that writes document embeddings to a Qdrant collection.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):
        print("[DEBUG] QdrantVectorSink.__init__ called")
        self._collection_name = collection_name
        self._openai_client = HierarchicalDataManager(client)

    def write(self, document: Document):
        print("[DEBUG] QdrantVectorSink.write called with a Document")
        self._openai_client.save_data(document)
        print("[DEBUG] Document saved to hierarchical data store!")


