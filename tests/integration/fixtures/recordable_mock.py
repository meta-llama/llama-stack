# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import importlib
import json
import os
import re
from datetime import datetime
from enum import Enum
from pathlib import Path


class RecordableMock:
    """A mock that can record and replay API responses."""

    def __init__(self, real_func, cache_dir, func_name, record=False):
        self.real_func = real_func
        self.json_path = Path(cache_dir) / f"{func_name}.json"
        self.record = record
        self.cache = {}

        # Load existing cache if available and not recording
        if self.json_path.exists():
            try:
                with open(self.json_path, "r") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Error loading cache from {self.json_path}: {e}")
                raise

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
                return self._reconstruct_object(cached_data["value"])
            else:
                # It's chunks from an async generator
                async def replay_generator():
                    for chunk in cached_data["chunks"]:
                        yield self._reconstruct_object(chunk)

                return replay_generator()

    def _create_cache_key(self, args, kwargs):
        """Create a hashable key from the function arguments, ignoring auto-generated IDs."""
        # Convert to JSON strings with sorted keys
        key = json.dumps((args, kwargs), sort_keys=True, default=self._json_default)

        # Post-process the key with regex to replace IDs with placeholders
        # Replace UUIDs and similar patterns
        key = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<UUID>", key)

        # Replace temporary file paths created by tempfile.mkdtemp()
        key = re.sub(r"/var/folders/[^,'\"\s]+", "<TEMP_FILE>", key)

        # Replace /tmp/ paths which are also commonly used for temporary files
        key = re.sub(r"/tmp/[^,'\"\s]+", "<TEMP_FILE>", key)

        return key

    def _save_cache(self):
        """Save the cache to disk in JSON format."""
        os.makedirs(self.json_path.parent, exist_ok=True)

        # Write the JSON file with pretty formatting
        try:
            with open(self.json_path, "w") as f:
                json.dump(self.cache, f, indent=2, sort_keys=True, default=self._json_default)
                # write another empty line at the end of the file to make pre-commit happy
                f.write("\n")
        except Exception as e:
            print(f"Error saving JSON cache: {e}")

    def _json_default(self, obj):
        """Default function for JSON serialization of objects."""

        if isinstance(obj, datetime):
            return {
                "__datetime__": obj.isoformat(),
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
            }

        if isinstance(obj, Enum):
            return {
                "__enum__": obj.__class__.__name__,
                "value": obj.value,
                "__module__": obj.__class__.__module__,
            }

        # Handle Pydantic models
        if hasattr(obj, "model_dump"):
            model_data = obj.model_dump()
            return {
                "__pydantic__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "data": model_data,
            }

    def _reconstruct_object(self, data):
        """Reconstruct an object from its JSON representation."""
        if isinstance(data, dict):
            # Check if this is a serialized datetime
            if "__datetime__" in data:
                try:
                    module_name = data.get("__module__", "datetime")
                    class_name = data.get("__class__", "datetime")

                    # Try to import the specific datetime class
                    module = importlib.import_module(module_name)
                    dt_class = getattr(module, class_name)

                    # Parse the ISO format string
                    dt = dt_class.fromisoformat(data["__datetime__"])
                    return dt
                except (ImportError, AttributeError, ValueError) as e:
                    print(f"Error reconstructing datetime: {e}")
                    return data

            # Check if this is a serialized enum
            elif "__enum__" in data:
                try:
                    module_name = data.get("__module__", "builtins")
                    enum_class = self._import_class(module_name, data["__enum__"])
                    return enum_class(data["value"])
                except (ImportError, AttributeError) as e:
                    print(f"Error reconstructing enum: {e}")
                    return data

            # Check if this is a serialized Pydantic model
            elif "__pydantic__" in data:
                try:
                    module_name = data.get("__module__", "builtins")
                    model_class = self._import_class(module_name, data["__pydantic__"])
                    return model_class(**self._reconstruct_object(data["data"]))
                except (ImportError, AttributeError) as e:
                    print(f"Error reconstructing Pydantic model: {e}")
                    return data

            # Regular dictionary
            return {k: self._reconstruct_object(v) for k, v in data.items()}

        # Handle lists
        elif isinstance(data, list):
            return [self._reconstruct_object(item) for item in data]

        # Return primitive types as is
        return data

    def _import_class(self, module_name, class_name):
        """Import a class from a module."""
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
