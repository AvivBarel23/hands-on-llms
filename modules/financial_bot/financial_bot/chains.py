import os
import time
from typing import Any, Dict, List, Optional
import json
import inspect
import datetime
import requests
import openai

import qdrant_client
from qdrant_client.models import PointStruct
from langchain import chains
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline
from unstructured.cleaners.core import (
    clean,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    replace_unicode_quotes,
)

from financial_bot.embeddings import EmbeddingModelSingleton
from financial_bot.template import PromptTemplate

INDICES_PATH = os.path.join(os.path.dirname(__file__), "../../streaming_pipeline/streaming_pipeline/hierarchy_indices_db.json")
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

class StatelessMemorySequentialChain(chains.SequentialChain):
    """
    A sequential chain that uses a stateless memory to store context between calls.

    This chain overrides the _call and prep_outputs methods to load and clear the memory
    before and after each call, respectively.
    """

    history_input_key: str = "to_load_history"

    def _call(self, inputs: Dict[str, str], **kwargs) -> Dict[str, str]:
        """
        Override _call to load history before calling the chain.

        This method loads the history from the input dictionary and saves it to the
        stateless memory. It then updates the inputs dictionary with the memory values
        and removes the history input key. Finally, it calls the parent _call method
        with the updated inputs and returns the results.
        """

        to_load_history = inputs[self.history_input_key]
        for (
            human,
            ai,
        ) in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key: human},
                outputs={self.memory.output_key: ai},
            )
        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)

        del inputs[self.history_input_key]

        return super()._call(inputs, **kwargs)

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """
        Override prep_outputs to clear the internal memory after each call.

        This method calls the parent prep_outputs method to get the results, then
        clears the stateless memory and removes the memory key from the results
        dictionary. It then returns the updated results.
        """

        results = super().prep_outputs(inputs, outputs, return_only_outputs)

        # Clear the internal memory.
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key] = ""

        return results


class ContextExtractorChain(Chain):
    """
    Encode the question, search the vector store for top-k articles and return
    context news from documents collection of Alpaca news.

    Attributes:
    -----------
    top_k : int
        The number of top matches to retrieve from the vector store.
    embedding_model : EmbeddingModelSingleton
        The embedding model to use for encoding the question.
    vector_store : qdrant_client.QdrantClient
        The vector store to search for matches.
    vector_collection : str
        The name of the collection to search in the vector store.
    hierarchy_file: str
        The path to the hierarchy json file created in the streaming pipeline.
    """

    top_k: int = 10
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str
    hierarchy_file: str = INDICES_PATH
    hierarchy: Dict[str, Any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vector_collection = kwargs.get("vector_collection", None)

        # Load existing hierarchy or initialize a new one
        if os.path.exists(self.hierarchy_file):
            with open(self.hierarchy_file, "r") as f:
                self.hierarchy = json.load(f)
        else:
            raise ValueError(f"Could not find architecture json in path {INDICES_PATH}")

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "question"]

    @property
    def output_keys(self) -> List[str]:
        return ["context"]


    def classify_with_gpt(self, text: str, options: List[str], level: str, top_k: int = 1, sector: Optional[str] = None, subject: Optional[str] = None) -> List[str]:
        debug_print(f"[DEBUG] query is: {text}")
        system_prompt = f"""
            You are tasked with classifying the following user query into the following three categories:
    
            ### 1. **Sector**:
            The broad industry or field to which the query talks about (e.g., Finance, Healthcare, Technology).
            - **Explanation**: The **Sector** is the broadest classification. For instance, if the query is about healthcare services or innovations in the medical field, it would fall under **Healthcare**. If it's about technology products or software development, it would fall under **Technology**. Please pick the sector that fits best based on the text.
    
            ### 2. **Company/Subject**:
            The specific company or subject of the query (e.g., Google, Tesla, artificial intelligence, climate change).
            - **Explanation**: The **Company/Subject** level focuses on the specific company or subject mentioned. For example, if the query mentions a new technology by **Apple** or discusses **Artificial Intelligence**, the response should reflect that. If the subject doesn’t match the options, suggest a fitting one. You can classify topics like "climate change" under the **Subject** category even if it isn't a company.
    
            ### 3. **Event Type**:
            The type of event or activity described in the query (e.g., merger, financial report, product launch, acquisition, scientific discovery).
            - **Explanation**: The **Event Type** categorizes what the query describes in terms of events or activities. For example, if the query talks about a company merger, it should be classified under **Merger**. If it's about a product release by **Apple**, it should be classified as a **Product Launch**. If no event type matches the options, suggest one based on the query's context.
        """

        user_prompt = ""
        # Building the prompt based on the level
        if level == "subject":
            user_prompt += (
                f"you need to return the top {top_k} {level}s the following text belongs under the sector '{sector}':\n\n"
            )
        elif level == "event type":
            user_prompt += (
                f"Based on the following text, return the top {top_k} {level}s it belongs to under the sector '{sector}' and subject '{subject}':\n\n"
            )
        else:
            user_prompt += (
                f"Based on the following text, return the top {top_k} {level}s it belongs to:\n\n"
            )

        debug_print(f"[DEBUG] Options are {options}, type is {type(options)}")
        user_prompt += f"Text: {text}\n\n"
        user_prompt += f"Options: {', '.join(options)}\n\n"
        user_prompt += (
            f"Your task is to return the top {top_k} most relevant options from the given list. "
            "**Do not reply with 'neither of the options' or 'none of them' or anything of the sort! This is not a valid answer! "
            "You must choose from the provided options. "
            f"Return the top {top_k} options as a comma-separated list, ordered by relevance. "
            "Do not include any additional text or explanations."
        )

        # Request GPT classification
        response = openai.chat.completions.create(
            model="gpt-4",  # Replace with your desired model
            messages=[
                {
                    "role": "system",
                    "content": f"You are a financial classifier for data, {system_prompt}"
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.8,
            max_tokens=50,  # Increased to allow for multiple options
            top_p=1
        )

        # Extract the top k options from the GPT response
        classification = response.choices[0].message.content.strip().replace(".", "")
        top_k_options = [opt.strip() for opt in classification.split(",")][:top_k]  # Split and limit to top k

        debug_print(f"[DEBUG] user prompt :{user_prompt}, GPT classification result: {top_k_options}")
        return top_k_options

    def summarize_with_gpt(self, text: str) -> str:
        # Request GPT classification
        limit_tokens=100
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this text no longer than {limit_tokens} tokens: {text}"
                }
            ],
            temperature=0.8,
            max_tokens=limit_tokens,
            top_p=1
        )


        classification = response.choices[0].message.content.strip().replace(".", "")

        return classification


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


    def search(self, query: str, query_vector: List[float], top_k):
        """
        Searches for the most relevant data based on the hierarchical structure with top-k branch exploration.

        Args:
            query (str): The user's search query.
            query_vector (List[float]): Embedding vector for the query.
            top_k (int): The number of top branches to explore at each level.

        Returns:
            List[dict]: List of relevant data points.
        """
        sectors = self.get_all_sectors()
        debug_print(f"[DEBUG] Found sectors: {sectors}")

        top_sectors = self.classify_with_gpt(query, sectors, "sector", top_k=top_k)
        debug_print(f"[DEBUG] Top sectors: {top_sectors}")

        all_results = []

        # Step 2: Iterate over the top-k sectors
        for sector in top_sectors:
            debug_print(f"[DEBUG] Exploring sector: {sector}")

            # Get the subjects under the sector
            subjects = self.get_subjects_under_sector(sector)
            debug_print(f"[DEBUG] Found subjects in sector '{sector}': {subjects}")

            # Classify the query to find relevant subjects in the sector
            top_subjects = self.classify_with_gpt(query, subjects, "subject", sector=sector, top_k=top_k)
            debug_print(f"[DEBUG] Top subjects under sector '{sector}': {top_subjects}")

            # Step 3: Iterate through the top-k subjects under each sector
            for subject in top_subjects:
                debug_print(f"[DEBUG] Exploring subject: {subject}")

                # Get event types under this subject
                event_types = self.get_event_types_under_subject(sector, subject)
                debug_print(f"[DEBUG] Found event types for subject '{subject}': {event_types}")

                # Classify the query to find the most relevant event type
                top_event_types = self.classify_with_gpt(query, event_types, "event type", sector=sector, subject=subject, top_k=top_k)
                debug_print(f"[DEBUG] Top event types for subject '{subject}': {top_event_types}")

                # Step 4: Iterate through the top-k event types under each subject
                for event_type in top_event_types:
                    debug_print(f"[DEBUG] Exploring event type: {event_type}")

                    # Formulate the collection name as a triplet: sector_subject_event_type
                    doc_collection_name = f"{sector}__{subject}__{event_type}".lower().replace(" ", "_")
                    debug_print(f"[DEBUG] Collection name: {doc_collection_name}")

                    # Step 5: Search within the specific collection (sector_subject_event_type)
                    try:
                        debug_print(f"[DEBUG] Searching in collection '{doc_collection_name}'...")
                        results = self.vector_store.search(
                            query_vector=query_vector,
                            collection_name=self.vector_collection,
                            limit=top_k,
                            timeout=360,
                            query_filter={
                                "must": [
                                    {
                                        "key": "collection_name",
                                        "match": {
                                            "value": doc_collection_name
                                        }
                                    }
                                ]
                            }
                        )
                        debug_print(f"[DEBUG] Retrieved {len(results)} results from collection '{doc_collection_name}'.")
                        all_results.extend(results)

                    except Exception as e:
                        debug_print(f"[ERROR] Search failed for collection '{doc_collection_name}': {e}")
                # Step 6: Combine all results and sort them by relevance score
        debug_print(f"[DEBUG] Combining and sorting all results ({len(all_results)} total)...")
        debug_print(f"all results:!!!!!!!!!!!!!!!!!!{all_results}")
        # Step 6: Combine all results and sort them by relevance score
        debug_print(f"[DEBUG] Combining and sorting all results ({len(all_results)} total)...")

        # Sort results by score (descending order)
        sorted_results = sorted(all_results, key=lambda x: x.score, reverse=True)

        # Debug print the sorted results
        debug_print(f"[DEBUG] Sorted results: {sorted_results}")

        return sorted_results[:top_k]  # Return the top-k documents based on score


    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        _, quest_key = self.input_keys
        question_str = inputs[quest_key]

        cleaned_question = self.clean(question_str)
        # TODO: Instead of cutting the question at 'max_input_length', chunk the question in 'max_input_length' chunks,
        # pass them through the model and average the embeddings.
        cleaned_question = cleaned_question[: self.embedding_model.max_input_length]
        embeddings = self.embedding_model(cleaned_question)

        # TODO: Using the metadata, use the filter to take into consideration only the news from the last 24 hours
        # (or other time frame).
        matches = self.search(question_str, embeddings, top_k=self.top_k)

        debug_print(f"[DEBUG]\n" + "\n".join(f"match: {item}" for item in matches))

        text=""
        for match in matches:
            payload = match.payload
            text+=payload.get("text", "")

        summary=self.summarize_with_gpt(text)
        debug_print(f"[DEBUG] summary with gpt: {summary}")
        return {
                "context": summary,
            }

    def clean(self, question: str) -> str:
        """
        Clean the input question by removing unwanted characters.

        Parameters:
        -----------
        question : str
            The input question to clean.

        Returns:
        --------
        str
            The cleaned question.
        """
        question = clean(question)
        question = replace_unicode_quotes(question)
        question = clean_non_ascii_chars(question)

        return question


class FinancialBotQAChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""

    hf_pipeline: HuggingFacePipeline
    template: PromptTemplate

    @property
    def input_keys(self) -> List[str]:
        """Returns a list of input keys for the chain"""

        return ["context"]

    @property
    def output_keys(self) -> List[str]:
        """Returns a list of output keys for the chain"""

        return ["answer"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Calls the chain with the given inputs and returns the output"""

        inputs = self.clean(inputs)
        prompt = self.template.format_infer(
            {
                "user_context": inputs["about_me"],
                "news_context": inputs["context"],
                "chat_history": inputs["chat_history"],
                "question": inputs["question"],
            }
        )

        start_time = time.time()
        response = self.hf_pipeline(prompt["prompt"])
        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000

        if run_manager:
            run_manager.on_chain_end(
                outputs={
                    "answer": response,
                },
                # TODO: Count tokens instead of using len().
                metadata={
                    "prompt": prompt["prompt"],
                    "prompt_template_variables": prompt["payload"],
                    "prompt_template": self.template.infer_raw_template,
                    "usage.prompt_tokens": len(prompt["prompt"]),
                    "usage.total_tokens": len(prompt["prompt"]) + len(response),
                    "usage.actual_new_tokens": len(response),
                    "duration_milliseconds": duration_milliseconds,
                },
            )

        return {"answer": response}

    def clean(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Cleans the inputs by removing extra whitespace and grouping broken paragraphs"""

        for key, input in inputs.items():
            cleaned_input = clean_extra_whitespace(input)
            cleaned_input = group_broken_paragraphs(cleaned_input)

            inputs[key] = cleaned_input

        return inputs
