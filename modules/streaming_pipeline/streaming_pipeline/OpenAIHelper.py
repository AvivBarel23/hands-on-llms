from typing import List, Optional, Dict
import openai
from qdrant_client import QdrantClient


class HierarchicalCategoryManager:
    def __init__(
            self,
            qdrant_client: QdrantClient,
            collection_name: str,
            max_sectors: int = 20,  # Limit the number of sectors
            max_event_types: int = 15,  # Limit the number of event types
            max_subjects: int = 100,  # Limit the number of subjects
            default_model: str = "gpt-4",
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.max_sectors = max_sectors
        self.max_event_types = max_event_types
        self.max_subjects = max_subjects
        self.default_model = default_model

    def fetch_existing_categories(self, level: str) -> List[str]:
        """
        Fetch existing categories for a specific hierarchy level (e.g., sector, subject, event_type).
        """
        response = self.client.get_collection(self.collection_name)
        return response["payload_schema"].get(level, {}).get("values", [])

    def classify_hierarchy(self, document_text: str, level: str, existing_categories: List[str]) -> Optional[str]:
        """
        Classify a document_text into a specific hierarchy level using OpenAI.
        If no category fits, return None.
        """
        prompt = f"""
        You are a classifier for financial data.
        Existing {level} categories are: {', '.join(existing_categories)}.
        Classify the following text into one of these {level} categories (return only the name of the category):
        {document_text}
        """
        response = openai.ChatCompletion.create(
            model=self.default_model,
            messages=[
                {"role": "system", "content": f"You are a financial data {level} classifier."},
                {"role": "user", "content": prompt},
            ],
        )
        category = response["choices"][0]["message"]["content"].strip()
        return category if category in existing_categories else None

    def generate_new_category(self, document_text: str, level: str, existing_categories:List[str]) -> str:
        """
        Generate a new category for a specific hierarchy level using OpenAI.
        """
        prompt = f"""
        You are a financial expert. The following text doesn't fit into any existing {level} categories which are {', '.join(existing_categories)}.
        Suggest a concise and relevant {level} category for this text:
        {document_text}
        """
        response = openai.ChatCompletion.create(
            model=self.default_model,
            messages=[
                {"role": "system", "content": f"You are a financial {level} expert."},
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"].strip()

    def categorize_hierarchically(self, document_text: str) -> Dict[str, str]:
        """
        Categorize a document_text hierarchically into sector, subject, and event_type.
        """
        hierarchy = [
            {"level": "sector", "max_categories": self.max_sectors},
            {"level": "subject", "max_categories": self.max_subjects},
            {"level": "event_type", "max_categories": self.max_event_types},
        ]
        metadata = {}

        for step in hierarchy:
            level = step["level"]
            max_categories = step["max_categories"]

            # Fetch existing categories for the current level
            existing_categories = self.fetch_existing_categories(level)

            # Step 1: Attempt to classify the document_text into the existing categories
            category = self.classify_hierarchy(document_text, level, existing_categories)

            # Step 2: If no suitable category exists, generate a new one (with limits if applicable)
            if not category:
                if max_categories and len(existing_categories) >= max_categories:
                    raise ValueError(f"Category limit reached for {level}. Cannot add new categories.")
                category = self.generate_new_category(document_text, level,existing_categories)

                # Add the new category to the metadata schema
                self.client.update_collection(
                    self.collection_name,
                    optimizers_config={
                        level: {"values": existing_categories + [category]}
                    },
                )

            # Store the category in the metadata
            metadata[level] = category

        return metadata
