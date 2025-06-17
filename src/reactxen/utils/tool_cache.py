import hashlib
import json
import os
from typing import Any


class ToolInvocationCache:
    def __init__(self, cache_file: str = "tool_cache.json"):
        self.cache = {}
        self.cache_file = cache_file
        self._load_cache_from_file()

    def _generate_cache_key(self, tool_name: str, tool_parameters: str) -> str:
        """Generate a unique cache key based on the tool name and tool parameters."""
        # Cache key is just the tool_name and tool_parameters concatenated
        return f"{tool_name}::{tool_parameters}"

    def get_from_cache(self, tool_name: str, tool_parameters: str) -> Any:
        """Retrieve the result from the cache if it exists."""
        cache_key = self._generate_cache_key(tool_name, tool_parameters)
        return self.cache.get(cache_key, None)

    def add_to_cache(self, tool_name: str, tool_parameters: str, result: Any):
        """Add a result to the cache and persist to file."""
        cache_key = self._generate_cache_key(tool_name, tool_parameters)
        self.cache[cache_key] = result
        #print (f'came here ---- {self.cache.keys()}')
        #print ('---------------------------')
        self._save_cache_to_file()  # Persist the cache to file every time we add a new invocation

    def clear_cache(self):
        """Clear the entire cache and also the file cache."""
        self.cache.clear()
        self._save_cache_to_file()

    def query_cache(self, tool_name: str, tool_parameters: str) -> bool:
        """Check if a cache entry exists for the given tool name and parameters."""
        cache_key = self._generate_cache_key(tool_name, tool_parameters)
        return cache_key in self.cache

    def _save_cache_to_file(self):
        """Save the current cache to a file."""
        with open(self.cache_file, "w") as file:
            json.dump(self.cache, file)

    def _load_cache_from_file(self):
        """Load the cache from a file if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as file:
                self.cache = json.load(file)
        else:
            self.cache = {}
