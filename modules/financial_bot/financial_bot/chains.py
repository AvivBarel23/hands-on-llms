import os
import time
from typing import Any, Dict, List, Optional

import openai
import qdrant_client
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
    """

    top_k: int = 1
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "question"]

    @property
    def output_keys(self) -> List[str]:
        return ["context"]

    def classify_hierarchy_level(self, query: str, options: List[str], level: str) -> str:
        """
        Classifies a query into the top-k options at a given hierarchy level.

        Args:
            query (str): The user's search query.
            options (List[str]): Available options at the current hierarchy level.
            level (str): The current hierarchy level (sector, subject, event type).
            top_k (int): The number of top options to return.

        Returns:
            List[str]: The top-k most relevant options for the query.
        """
        prompt = (
            f"Given the query '{query}', which of these {level}s is it most related to? "
            f"Options: {', '.join(options)}"
        )
        response = (openai.chat.completions.create(
            model="gpt-4",
            messages=prompt,
            max_tokens=100,
            stop=None,
            n=1,
            temperature=0.7,
        ))
        classification = response.choices[0].text.strip().replace(".", "")
        return classification


    def get_current_level_options(self, level: str, parent: Optional[str] = None) -> List[str]:
        """
        Fetches available options at the current hierarchy level.

        Args:
            level (str): The hierarchy level (sector, subject, event_type).
            parent (Optional[str]): The parent node (if applicable).

        Returns:
            List[str]: The names of the available options at the current level.
        """
        query_filter = {"must": [{"key": "type", "match": {"value": level}}]}
        if parent:
            query_filter["must"].append({"key": "parent", "match": {"value": parent}})

        options = [
            node.payload["name"]
            for node in self.vector_store.search(
                collection_name=self.vector_collection,
                query_vector=[0.0],  # Dummy vector as we are using filters
                query_filter=query_filter
            )
        ]
        return options


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
        sectors = self.get_current_level_options("sector")
        sector = self.classify_hierarchy_level(query, sectors, "sector")

        subjects = self.get_current_level_options("subject", parent=sector)
        subject = self.classify_hierarchy_level(query, subjects, "subject")

        event_types = self.get_current_level_options("event_type", parent=subject)
        event_type = self.classify_hierarchy_level(query, event_types, "event_type")

        collection_name = f"alpaca_financial_news_{sector}_{subject}_{event_type}".lower().replace(" ", "_")
        data = self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=self.top_k,
        )
        return data

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

        context = ""
        for match in matches:
            context += match.payload["summary"] + "\n"

        return {
            "context": context,
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
