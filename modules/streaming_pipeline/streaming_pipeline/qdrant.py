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
from qdrant_client.conversions.common_types import ScoredPoint

from streaming_pipeline import constants
from streaming_pipeline.models import Document

# -- Global path to log in the same directory as this file
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "debug.log")

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
    debug_print("[DEBUG] build_qdrant_client START")

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
    debug_print("[DEBUG] QdrantClient built successfully")

    debug_print("[DEBUG] build_qdrant_client END")
    return client


class HierarchicalDataManager:
    def __init__(self, qdrant_client: QdrantClient,collection_name):
        self.new_node_id=0
        debug_print("[DEBUG] HierarchicalDataManager.__init__ START")
        self.client = qdrant_client
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.indices_collection:str = collection_name

    def classify_with_gpt(self, text: str, options: List[str], level: str) -> str:
        max_retries = 5  # Maximum number of retry attempts
        retry_delay = 2  # Initial delay between retries (in seconds)
        attempt = 0

        while attempt < max_retries:
            try:
                debug_print("[DEBUG] classify_with_gpt START")
                debug_print(f"[DEBUG] text='{text[:50]}...' options={options} level={level}")

                prompt = (
                    f"Based on the following text, decide which {level} it belongs to:\n\n"
                    f"Text: {text}\n\n"
                    f"Options: {', '.join(options)}\n\n"
                    f"Only return the name of the {level}. If there is no correct option, please suggest one."
                )
                #debug_print(prompt)

                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a financial classifier for data"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.8,
                    max_tokens=10,
                    top_p=1
                )

                debug_print("[DEBUG] classify_with_gpt before ")
                classification = response.choices[0].message.content.strip().replace(".", "")
                debug_print("[DEBUG] classify_with_gpt after")
                debug_print(f"[DEBUG] GPT classification result: {classification}")

                if classification not in options:
                    debug_print(
                        f"[DEBUG] classification '{classification}' not in existing options => treating as new"
                    )
                    debug_print("[DEBUG] classify_with_gpt END (NEW LABEL)")
                    return classification

                debug_print("[DEBUG] classify_with_gpt END (EXISTING LABEL)")
                return classification

            except Exception as e:
                attempt += 1
                wait_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                debug_print(f"Exception :{e}, [DEBUG] Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt}/{max_retries})")
                time.sleep(wait_time)


        # If all retries fail, return an appropriate error message or raise an exception
        debug_print("[DEBUG] classify_with_gpt END (FAILED AFTER RETRIES)")
        raise Exception("Rate limit exceeded. Maximum retries reached.")


    def get_hierarchy_node(self, name: str, level: str) -> Optional[ScoredPoint]:
        debug_print("[DEBUG] get_hierarchy_node START")
        debug_print(f"[DEBUG] name='{name}', level='{level}'")

        results = self.client.search(
            collection_name=self.indices_collection,
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
            debug_print("[DEBUG] Found matching node(s). Returning the first one.")
        else:
            debug_print("[DEBUG] No matching node found.")

        debug_print("[DEBUG] get_hierarchy_node END")
        return results[0] if results else None


    def save_hierarchy_node(self,
                            name: str,
                            level: str,
                            parent: Optional[str] = None,
                            children: Optional[List[str]] = None):
        debug_print("[DEBUG] save_hierarchy_node START")
        debug_print(
            f"[DEBUG] name='{name}', level='{level}', parent='{parent}', children={children}"
        )

        node = self.get_hierarchy_node(name, level)
        if node:
            debug_print("[DEBUG] Node exists; updating existing node.")
            payload = node.payload
            payload["children"] = list(set(payload.get("children", []) + (children or [])))
            self.client.upsert(
                collection_name=self.indices_collection,
                points=[
                    PointStruct(
                        id=node.id,  # use the actual scored point ID
                        vector=[1.0],
                        payload=payload,
                    )
                ],
            )
        else:
            debug_print("[DEBUG] Node does not exist; creating new node.")
            try:
                self.new_node_id +=1
                self.client.upsert(
                    collection_name=self.indices_collection,
                    points=[
                        PointStruct(
                            id=self.new_node_id,
                            vector=[1.0],  # Dummy vector
                            payload={
                                "type": level,
                                "name": name,
                                "parent": parent,
                                "children": children or [],
                            },
                        )
                    ],
                )
            except Exception as e:
                debug_print(f"[DEBUG] exception :{e} , tried to insert node to collection {self.indices_collection}  ")


        debug_print("[DEBUG] save_hierarchy_node END")


    def save_data(self, document):
        debug_print("[DEBUG] save_data START")
        document_text = ' '.join(document.text)
        debug_print("[DEBUG] Full document text: " + document_text[:100] + "...")
        try:
            # Step 1: Sector Classification
            sectors = [
                node.payload["name"] for node in self.client.search(
                    collection_name=self.indices_collection,
                    query_vector=[1.0],
                    filter={"must": [{"key": "type", "match": {"value": "sector"}}]}
                )
            ]
        except Exception as e:
            debug_print("[DEBUG] exception :" + str(e))
            sectors=[]
        debug_print(f"[DEBUG] Found existing sectors: {sectors}")
        sector = self.classify_with_gpt(document_text, sectors, "sector")
        debug_print(f"[DEBUG] sector => '{sector}'")
        self.save_hierarchy_node(name=sector, level="sector")
        debug_print(f"[DEBUG] saved sector")


    # Step 2: Company/Subject Classification
        subjects_raw =  self.client.search(
                collection_name=self.indices_collection,
                query_vector=[1.0],
                filter={
                    "must": [
                        {"key": "type", "match": {"value": "subject"}},
                        {"key": "parent", "match": {"value": sector}},
                    ]
                }
            )
        debug_print(f"[DEBUG] subjects => '{subjects_raw}'")
        subjects =[ node.payload["name"] for node in subjects_raw]
        debug_print(f"[DEBUG] Found existing subjects under sector: {subjects}")
        subject = self.classify_with_gpt(document_text, subjects, "subject")
        debug_print(f"[DEBUG] subject => '{subject}'")
        self.save_hierarchy_node(name=subject, level="subject", parent=sector)
        debug_print(f"[DEBUG] saved subject")

        # Step 3: Event Type Classification
        event_types = [
            node.payload["name"] for node in self.client.search(
                collection_name=self.indices_collection,
                query_vector=[1.0],
                filter={
                    "must": [
                        {"key": "type", "match": {"value": "event_type"}},
                        {"key": "parent", "match": {"value": subject}},
                    ]
                }
            )
        ]
        debug_print(f"[DEBUG] Found existing event_types under subject: {event_types}")
        event_type = self.classify_with_gpt(document_text, event_types, "event type")
        debug_print(f"[DEBUG] event_type => '{event_type}'")
        self.save_hierarchy_node(name=event_type, level="event_type", parent=subject)

        debug_print(f"[DEBUG] saved event_type: {event_type}")


        # Step 4: Save the document in its specific Qdrant collection
        collection_name = f"alpaca_financial_news_{sector}_{subject}_{event_type}".lower().replace(" ", "_")
        debug_print(f"[DEBUG] Final collection_name => '{collection_name}'")
        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            debug_print(f"[DEBUG] Collection '{collection_name}' does NOT exist; creating.")
            if not document.embeddings:
              raise ValueError("document.embeddings is missing or empty.")
            vector_size = len(document.embeddings[0])
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            debug_print("[DEBUG] Created new collection with vector_size=" + str(vector_size))


        debug_print("[DEBUG] Upserting the document's embeddings...")
        ids, payloads = document.to_payloads()
        points = [
            PointStruct(id=idx, vector=vector, payload=_payload)
            for idx, vector, _payload in zip(ids, document.embeddings, payloads)
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        debug_print(f"[DEBUG] Document saved successfully in {collection_name}")

        debug_print("[DEBUG] save_data END")


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
        debug_print("[DEBUG] QdrantVectorSink.__init__ START")
        self.hierarchical_data_manager = HierarchicalDataManager(client,collection_name)
        debug_print("[DEBUG] QdrantVectorSink.__init__ END")

    def write(self, document: Document):
        debug_print("[DEBUG] QdrantVectorSink.write START")
        self.hierarchical_data_manager.save_data(document)
        debug_print("[DEBUG] Document saved to hierarchical data store!")
        debug_print("[DEBUG] QdrantVectorSink.write END")


class QdrantVectorOutput(DynamicOutput):
    def __init__(
        self,
        vector_size: int,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        client: Optional[QdrantClient] = None,
    ):
        debug_print("[DEBUG] QdrantVectorOutput.__init__ START")
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
        debug_print(f"[DEBUG] QdrantVectorOutput.build START on worker {worker_index}/{worker_count}")
        sink = QdrantVectorSink(self.client, self._collection_name)
        debug_print("[DEBUG] QdrantVectorOutput.build END - returning sink")
        return sink
