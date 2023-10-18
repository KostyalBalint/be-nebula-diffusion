import os

from algoliasearch.search_client import SearchClient
from dotenv import load_dotenv

load_dotenv(".env")

# Connect and authenticate with your Algolia app
client = SearchClient.create(os.getenv("ALGOLIA_ID"), os.getenv("ALGOLIA_TOKEN"))

# Create a new index and add a record
index = client.init_index(os.getenv("ALGOLIA_INDEX_NAME", "nebula_diffusion"))


def search_algolia(query: str):
    # Search the index and print the results
    results = index.search("test_record")
    return results["hits"]