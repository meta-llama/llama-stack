# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import os
import pickle
import re
from pathlib import Path


class RecordableMock:
    """A mock that can record and replay API responses."""

    def __init__(self, real_func, cache_dir, func_name, record=False):
        self.real_func = real_func
        self.pickle_path = Path(cache_dir) / f"{func_name}.pickle"
        self.json_path = Path(cache_dir) / f"{func_name}.json"
        self.record = record
        self.cache = {}

        # Load existing cache if available and not recording
        if self.pickle_path.exists():
            try:
                with open(self.pickle_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                print(f"Error loading cache from {self.pickle_path}: {e}")

    async def __call__(self, *args, **kwargs):
        """
        Returns a coroutine that when awaited returns the result or an async generator,
        matching the behavior of the original function.
        """
        # Create a cache key from the arguments
        key = self._create_cache_key(args, kwargs)

        if self.record:
            # In record mode, always call the real function
            real_result = self.real_func(*args, **kwargs)

            # If it's a coroutine, we need to create a wrapper coroutine
            if hasattr(real_result, "__await__"):
                # Define a coroutine function that will record the result
                async def record_coroutine():
                    try:
                        # Await the real coroutine
                        result = await real_result

                        # Check if the result is an async generator
                        if hasattr(result, "__aiter__"):
                            # It's an async generator, so we need to record its chunks
                            chunks = []

                            # Create and return a new async generator that records chunks
                            async def recording_generator():
                                nonlocal chunks
                                async for chunk in result:
                                    chunks.append(chunk)
                                    yield chunk
                                # After all chunks are yielded, save to cache
                                self.cache[key] = {"type": "generator", "chunks": chunks}
                                self._save_cache()

                            return recording_generator()
                        else:
                            # It's a regular result, save it to cache
                            self.cache[key] = {"type": "value", "value": result}
                            self._save_cache()
                            return result
                    except Exception as e:
                        print(f"Error in recording mode: {e}")
                        raise

                return await record_coroutine()
            else:
                # It's already an async generator, so we need to record its chunks
                async def record_generator():
                    chunks = []
                    async for chunk in real_result:
                        chunks.append(chunk)
                        yield chunk
                    # After all chunks are yielded, save to cache
                    self.cache[key] = {"type": "generator", "chunks": chunks}
                    self._save_cache()

                return record_generator()
        elif key not in self.cache:
            # In replay mode, if the key is not in the cache, throw an error
            raise KeyError(
                f"No cached response found for key: {key}\nRun with --record-responses to record this response."
            )
        else:
            # In replay mode with a cached response
            cached_data = self.cache[key]

            # Check if it's a value or chunks
            if cached_data.get("type") == "value":
                # It's a regular value
                return cached_data["value"]
            else:
                # It's chunks from an async generator
                async def replay_generator():
                    for chunk in cached_data["chunks"]:
                        yield chunk

                return replay_generator()

    def _create_cache_key(self, args, kwargs):
        """Create a hashable key from the function arguments, ignoring auto-generated IDs."""
        # Convert args and kwargs to a string representation directly
        args_str = str(args)
        kwargs_str = str(sorted([(k, kwargs[k]) for k in kwargs]))

        # Combine into a single key
        key = f"{args_str}_{kwargs_str}"

        # Post-process the key with regex to replace IDs with placeholders
        # Replace UUIDs and similar patterns
        key = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<UUID>", key)

        # Replace temporary file paths created by tempfile.mkdtemp()
        key = re.sub(r"/var/folders/[^,'\"\s]+", "<TEMP_FILE>", key)

        return key

    def _save_cache(self):
        """Save the cache to disk in both pickle and JSON formats."""
        os.makedirs(self.pickle_path.parent, exist_ok=True)

        # Save as pickle for exact object preservation
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.cache, f)

        # Also save as JSON for human readability and diffing
        try:
            # Create a simplified version of the cache for JSON
            json_cache = {}
            for key, value in self.cache.items():
                if value.get("type") == "generator":
                    # For generators, create a simplified representation of each chunk
                    chunks = []
                    for chunk in value["chunks"]:
                        chunk_dict = self._object_to_json_safe_dict(chunk)
                        chunks.append(chunk_dict)
                    json_cache[key] = {"type": "generator", "chunks": chunks}
                else:
                    # For values, create a simplified representation
                    val = value["value"]
                    val_dict = self._object_to_json_safe_dict(val)
                    json_cache[key] = {"type": "value", "value": val_dict}

            # Write the JSON file with pretty formatting
            with open(self.json_path, "w") as f:
                json.dump(json_cache, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"Error saving JSON cache: {e}")

    def _object_to_json_safe_dict(self, obj):
        """Convert an object to a JSON-safe dictionary."""
        # Handle enum types
        if hasattr(obj, "value") and hasattr(obj.__class__, "__members__"):
            return {"__enum__": obj.__class__.__name__, "value": obj.value}

        # Handle Pydantic models
        if hasattr(obj, "model_dump"):
            return self._process_dict(obj.model_dump())
        elif hasattr(obj, "dict"):
            return self._process_dict(obj.dict())

        # Handle regular objects with __dict__
        try:
            return self._process_dict(vars(obj))
        except Exception as e:
            print(f"Error converting object to JSON-safe dict: {e}")
            # If we can't get a dict, convert to string
            return str(obj)

    def _process_dict(self, d):
        """Process a dictionary to make all values JSON-safe."""
        if not isinstance(d, dict):
            return d

        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = self._process_dict(v)
            elif isinstance(v, list):
                result[k] = [
                    self._process_dict(item)
                    if isinstance(item, dict)
                    else self._object_to_json_safe_dict(item)
                    if hasattr(item, "__dict__")
                    else item
                    for item in v
                ]
            elif hasattr(v, "value") and hasattr(v.__class__, "__members__"):
                # Handle enum
                result[k] = {"__enum__": v.__class__.__name__, "value": v.value}
            elif hasattr(v, "__dict__"):
                # Handle nested objects
                result[k] = self._object_to_json_safe_dict(v)
            else:
                # Basic types
                result[k] = v

        return result
