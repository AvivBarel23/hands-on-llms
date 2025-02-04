import json
import os
import inspect
import datetime

import openai
import time
from typing import List, Optional
from bytewax.outputs import DynamicOutput, StatelessSink
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff
from qdrant_client.models import PointStruct

from streaming_pipeline import constants
from streaming_pipeline.models import Document

# -- Global path to log in the same directory as this file
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "debug.log")
INDICES_PATH = os.path.join(os.path.dirname(__file__), "hierarchy_indices_db.json")



def debug_print(msg: str):
    """
    Logs debug messages to `debug.log` in this directory,
    including timestamp, filename, and line number.
    """
    # Capture call frame info (who called debug_print)
    frame_info = inspect.stack()[1]
    filename = os.path.basename(frame_info.filename)
    lineno = frame_info.lineno

    # Optional timestamp
    now_str = datetime.datetime.now().isoformat()

    formatted_msg = f"[{now_str}][{filename}:{lineno}] {msg}"
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")


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


class HierarchicalDataManager:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str):
        self.new_node_id = 0
        self.client = qdrant_client
        self.hierarchy_file = INDICES_PATH
        self.collection_name = collection_name
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # Load existing hierarchy or initialize a new one
        if os.path.exists(self.hierarchy_file):
            with open(self.hierarchy_file, "r") as f:
                self.hierarchy = json.load(f)
        else:
            self.hierarchy = {"name": "Root", "children": []}

    def save_hierarchy_to_file(self):
        """Save the current hierarchy to the JSON file."""
        with open(self.hierarchy_file, "w") as f:
            json.dump(self.hierarchy, f, indent=4)

    def get_all_sectors(self) -> List[str]:
        """Retrieve all sector names from the hierarchy."""
        return [child["name"] for child in self.hierarchy.get("children", [])]

    def get_subjects_under_sector(self, sector_name: str) -> List[str]:
        """Retrieve all subjects under a specific sector."""
        sector_node = self.find_node(sector_name)
        if not sector_node or sector_node.get("level") != "sector":
            raise ValueError(f"Sector {sector_name} not found.")
        return [child["name"] for child in sector_node.get("children", [])]

    def get_event_types_under_subject(self, sector_name: str, subject_name: str) -> List[str]:
        """Retrieve all event types under a specific sector and subject."""
        sector_node = self.find_node(sector_name)
        if not sector_node or sector_node.get("level") != "sector":
            raise ValueError(f"Sector '{sector_name}' not found.")

        subject_node = next(
            (child for child in sector_node.get("children", []) if child["name"] == subject_name), None
        )
        if not subject_node or subject_node.get("level") != "subject":
            raise ValueError(f"Subject {subject_name} not found under sector {sector_name}.")
        return [child["name"] for child in subject_node.get("children", [])]

    def find_node(self, name, current_node=None):
        """Recursively search for a node in the hierarchy by name."""
        if current_node is None:
            current_node = self.hierarchy

        if current_node["name"] == name:
            return current_node

        for child in current_node.get("children", []):
            result = self.find_node(name, child)
            if result:
                return result

        return None

    def save_hierarchy_node(self, name: str, level: str, parent: Optional[str] = None):
        """Add a node to the hierarchy in the JSON file."""
        if parent:
            parent_node = self.find_node(parent)
            if not parent_node:
                raise ValueError(f"Parent node '{parent}' not found.")
            # Check if node already exists
            existing_node = next(
                (child for child in parent_node.get("children", []) if child["name"] == name),
                None,
            )
            if not existing_node:
                parent_node.setdefault("children", []).append({"name": name, "level": level, "children": []})
        else:
            # Adding to root if no parent is provided
            if not any(child["name"] == name for child in self.hierarchy.get("children", [])):
                self.hierarchy["children"].append({"name": name, "level": level, "children": []})

        self.save_hierarchy_to_file()

    def classify_with_gpt(self, text: str, options: List[str], level: str, sector: Optional[str] = None, subject: Optional[str] = None) -> str:
                system_prompt = f"""
                    You are tasked with classifying a document into the following three categories:

                    ### 1. **Sector**:
                    The broad industry or field to which the document belongs (e.g., Finance, Healthcare, Technology).
                    - **Explanation**: The **Sector** is the broadest classification. For instance, if the document is about healthcare services or innovations in the medical field, it would fall under **Healthcare**. If it's about technology products or software development, it would fall under **Technology**. Please pick the sector that fits best based on the text.

                    ### 2. **Company/Subject**:
                    The specific company or subject mentioned in the document (e.g., Google, Tesla, artificial intelligence, climate change).
                    - **Explanation**: The **Company/Subject** level focuses on the specific company or subject mentioned. For example, if the document mentions a new technology by **Apple** or discusses **Artificial Intelligence**, the response should reflect that. If the subject doesn’t match the options, suggest a fitting one. You can classify topics like "climate change" under the **Subject** category even if it isn't a company.

                    ### 3. **Event Type**:
                    The type of event or activity described in the document (e.g., merger, financial report, product launch, acquisition, scientific discovery).
                    - **Explanation**: The **Event Type** categorizes what the document describes in terms of events or activities. For example, if the document talks about a company merger, it should be classified under **Merger**. If it's about a product release by **Apple**, it should be classified as a **Product Launch**. If no event type matches the options, suggest one based on the document's context.
                            """
                user_prompt=""
                # Building the prompt based on the level
                if level == "subject":
                    user_prompt += (
                        f"you need to decide which subject the following text belongs under the sector '{sector}':\n\n"
                    )
                elif level == "event type":
                    user_prompt += (
                        f"Based on the following text, decide which event type it belongs to under the sector '{sector}' and subject '{subject}':\n\n"
                    )
                else:
                    user_prompt += (
                        f"Based on the following text, decide which {level} it belongs to:\n\n"
                    )
                user_prompt += f"Text: {text}\n\n"
                user_prompt += f"Options: {', '.join(options)}\n\n"
                user_prompt += (
                    "If the list of options is empty, or none of the options seem appropriate for any of the categories, "
                    "**suggest an appropriate one** based on the content of the query. "
                    "Your suggestions should be **specific and relevant** to the content. "
                    "**Do not reply **neither of the options** or **none of them** or anything of the sort! this is not valid answer. "
                    f"Always provide an answer, even if it means suggesting a new category that fits better.\nThe answer must be only the name of the {level} without any garbage!")



                # Request GPT classification
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a financial classifier for data , {system_prompt}"
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0.8,
                    max_tokens=10,
                    top_p=1
                )


                classification = response.choices[0].message.content.strip().replace(".", "")

                return classification


    def save_data(self, document):
        """Save document data to Qdrant and hierarchy nodes to a JSON file."""
        document_text = ' '.join(document.text)

        # Step 1: Sector Classification
        sectors = self.get_all_sectors()
        sector = self.classify_with_gpt(document_text, sectors, "sector")
        self.save_hierarchy_node(name=sector, level="sector")

        # Step 2: Subject Classification
        subjects = self.get_subjects_under_sector(sector)
        subject = self.classify_with_gpt(document_text, subjects, "subject",sector=sector)
        self.save_hierarchy_node(name=subject, level="subject", parent=sector)

        # Step 3: Event Type Classification
        event_types = self.get_event_types_under_subject(sector, subject)
        event_type = self.classify_with_gpt(document_text, event_types, "event type",sector=sector,subject=subject)
        self.save_hierarchy_node(name=event_type, level="event_type", parent=subject)

        # Save the hierarchy to JSON
        self.save_hierarchy_to_file()

        # Step 4: Save the document in its specific Qdrant collection
        doc_collection_name = f"{sector}__{subject}__{event_type}".lower().replace(" ", "_")
        # Upsert the document embeddings into Qdrant
        try:
            ids, payloads = document.to_payloads(doc_collection_name)
            points = [
                PointStruct(id=idx, vector=vector, payload=_payload)
                for idx, vector, _payload in zip(ids, document.embeddings, payloads)
            ]
            # if len(points) > 1 and document.metadata['headline']:
            #     debug_print(f"[DEBUG] Number of points in doc with header {document.metadata['headline']} is {len(points)}")
            self.client.upsert(collection_name=self.collection_name, points=points)

        except Exception as e:
            debug_print(f"[DEBUG] Exception when upserting to new collection: {e}")


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
        self.hierarchical_data_manager = HierarchicalDataManager(client, collection_name)

    def write(self, document: Document):
        self.hierarchical_data_manager.save_data(document)


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
            response =self.client.get_collection(collection_name=self._collection_name)
        except Exception as e:
            if 'vectors_count' not in str(e):
                self.client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE
                    ),

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

        sink = QdrantVectorSink(self.client, self._collection_name)
        return sink
