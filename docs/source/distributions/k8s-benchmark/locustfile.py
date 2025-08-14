# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Locust load testing script for Llama Stack with Prism mock OpenAI provider.
"""

import random
from locust import HttpUser, task, between
import os

base_path = os.getenv("LOCUST_BASE_PATH", "/v1/openai/v1")

MODEL_ID = os.getenv("INFERENCE_MODEL")

class LlamaStackUser(HttpUser):
    wait_time = between(0.0, 0.0001)
    
    def on_start(self):
        """Setup authentication and test data."""
        # No auth required for benchmark server
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Test messages of varying lengths
        self.test_messages = [
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "What is the capital of France?"}],
            [{"role": "user", "content": "Explain quantum physics in simple terms."}],
            [{"role": "user", "content": "Write a short story about a robot learning to paint."}],
            [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI..."},
                {"role": "user", "content": "Can you give me a practical example?"}
            ]
        ]

    @task(weight=100)
    def chat_completion_streaming(self):
        """Test streaming chat completion (20% of requests)."""
        messages = random.choice(self.test_messages)
        payload = {
            "model": MODEL_ID, 
            "messages": messages,
            "stream": True,
            "max_tokens": 100
        }
        
        with self.client.post(
            f"{base_path}/chat/completions",
            headers=self.headers,
            json=payload,
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                chunks_received = 0
                try:
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                chunks_received += 1
                                if line_str.strip() == 'data: [DONE]':
                                    break
                    
                    if chunks_received > 0:
                        response.success()
                    else:
                        response.failure("No streaming chunks received")
                except Exception as e:
                    response.failure(f"Streaming error: {e}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
