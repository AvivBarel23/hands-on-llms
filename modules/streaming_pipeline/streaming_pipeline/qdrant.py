import os
from typing import List, Dict, Optional

import openai
from bytewax.outputs import DynamicOutput, StatelessSink
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff
from qdrant_client.models import PointStruct
from qdrant_client.models import ScoredPoint
from qdrant_client.conversions.common_types import ScoredPoint


from streaming_pipeline import constants
from streaming_pipeline.models import Document

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
        #TODO: What happens when there is no match (like at the beginning when the sectors are empty for example)
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
        if classification not in options:
            # Optionally, treat it as new
            return classification
        return classification

    def get_hierarchy_node(self, name: str, level: str) -> Optional[ScoredPoint]:
        results = self.client.search(
            collection_name=self.hierarchy_collection,
            query_vector=[1.0],  # Dummy query vector
            filter={
                "must": [
                    {"key": "name", "match": {"value": name}},
                    {"key": "type", "match": {"value": level}},
                ]
            },
            limit=1,
        )
        return results[0] if results else None

    def save_hierarchy_node(self, name: str, level: str, parent: Optional[str] = None, children: Optional[List[str]] = None):
        #TODO: After fixing the classify_with_gpt function, make sure this works, which means create a new level if doesn't exist, and inserts to existing level if exists
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
                        vector=[0.0],
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

    def save_data(self, document):
        """
        Save a document in the proper hierarchical structure.
        """
        document_text = ' '.join(document.text)

        # Step 1: Sector Classification
        sectors = [
            node["name"] for node in self.client.search(
                collection_name=self.hierarchy_collection,
                query_vector=[1.0],
                filter={"must": [{"key": "type", "match": {"value": "sector"}}]}
            )
        ]
        sector = self.classify_with_gpt(document_text, sectors, "sector")
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
        subject = self.classify_with_gpt(document_text, subjects, "subject")
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
        event_type = self.classify_with_gpt(document_text, event_types, "event type")
        # self.save_hierarchy_node(name=event_type, level="event_type", parent=subject)

        # Step 4: Save the document in its specific Qdrant collection
        collection_name = f"{sector}_{subject}_{event_type}".lower().replace(" ", "_")
        if not self.client.get_collection(collection_name):
            # Use document.embeddings to figure out the vector size
            vector_size = len(document.embeddings[0]) if document.embeddings else 768  # or any default
            self.client.create_collection(
                collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

            #TODO: add to the final tree!
        
        #Baseline code!!! don't modify the logic (unless adding something necessary)
        ids, payloads = document.to_payloads()
        points = [
            PointStruct(id=idx, vector=vector, payload=_payload)
            for idx, vector, _payload in zip(ids, document.embeddings, payloads)
        ]

        self.client.upsert(collection_name=collection_name, points=points)

class QdrantVectorOutput(DynamicOutput):
    """A class representing a Qdrant vector output.

    This class is used to create a Qdrant vector output, which is a type of dynamic output that supports
    at-least-once processing. Messages from the resume epoch will be duplicated right after resume.

    Args:
        vector_size (int): The size of the vector.
        collection_name (str, optional): The name of the collection.
            Defaults to constants.VECTOR_DB_OUTPUT_COLLECTION_NAME.
        client (Optional[QdrantClient], optional): The Qdrant client. Defaults to None.
    """

    def __init__(
        self,
        vector_size: int,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        client: Optional[QdrantClient] = None,
    ):
        self._collection_name = collection_name
        self._vector_size = vector_size

        if client:
            self.client = client
        else:
            self.client = build_qdrant_client()

        try:
            self.client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, ValueError):
            self.client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
                # Manuall add this optimizers_config to address issue: https://github.com/iusztinpaul/hands-on-llms/issues/72
                # qdrant_client.http.exceptions.ResponseHandlingException: 1 validation error for ParsingModel[InlineResponse2005] (for parse_as_type)
                # obj -> result -> config -> optimizer_config -> max_optimization_threads
                # none is not an allowed value (type=type_error.none.not_allowed)
                optimizers_config=OptimizersConfigDiff(max_optimization_threads=1),
            )

    def build(self, worker_index, worker_count):
        """Builds a QdrantVectorSink object.

        Args:
            worker_index (int): The index of the worker.
            worker_count (int): The total number of workers.

        Returns:
            QdrantVectorSink: A QdrantVectorSink object.
        """

        return QdrantVectorSink(self.client, self._collection_name)


def build_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
    """
    Builds a QdrantClient object with the given URL and API key.

    Args:
        url (Optional[str]): The URL of the Qdrant server. If not provided,
            it will be read from the QDRANT_URL environment variable.
        api_key (Optional[str]): The API key to use for authentication. If not provided,
            it will be read from the QDRANT_API_KEY environment variable.

    Raises:
        KeyError: If the QDRANT_URL or QDRANT_API_KEY environment variables are not set
            and no values are provided as arguments.

    Returns:
        QdrantClient: A QdrantClient object connected to the specified Qdrant server.
    """

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

    return client


class QdrantVectorSink(StatelessSink):
    """
    A sink that writes document embeddings to a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client to use for writing.
        collection_name (str, optional): The name of the collection to write to.
            Defaults to constants.VECTOR_DB_OUTPUT_COLLECTION_NAME.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):
        self._collection_name = collection_name
        self._openai_client=HierarchicalDataManager(client)

    def write(self, document: Document):
        self._openai_client.save_data(document)

