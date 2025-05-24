### Running test_milvus.py

Since we are using a remote Milvus configuration, you must start the Milvus containers before running the test. Follow these steps:

1. Navigate to the directory containing the docker-compose file:
   ```bash
   cd tests/unit/providers/vector_io/config
   ```

2. Start the Milvus containers using the docker-compose file:
   ```bash
   docker compose -f docker-compose.milvus.yaml up -d
   ```

3. Once the containers are running, you can run the test:
   ```bash
   pytest tests/unit/providers/vector_io/remote/test_milvus.py -v -s --tb=short --disable-warnings --asyncio-mode=auto
   ```

4. After running the test, you can stop the containers:
   ```bash
   docker compose -f docker-compose.milvus.yaml down
   ```
