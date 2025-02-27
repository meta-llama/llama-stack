## Summary
As discussed in #1061, this RFC introduces the design and endpoint specification for managing, invoking and serving document preprocessors. 
A preprocessor is an entity that can handle one or more of the following document-related tasks:

- Fetching - e.g., downloading a list of documents from a given URI;
- Converting - e.g., from a binary format (PDF, DOCX, etc.) to JSON or plain text;
- Chunking - i.e., dividing a large document into multiple, possibly overlapping segments for later ingestion.

Preprocessors are most notably used in the RAG ingestion process. However, they also have uses outside of RAG, such as in synthetic data generation.
We would like to introduce support for preprocessors as a new type of Llama Stack providers. The user will be able to dynamically add and remove them similarly to, e.g., models and vector DBs. Our proposed extension introduces document preprocessing as an independent function and additionally incorporates it into the existing RAG ingestion mechanism.


## Design
Most document preprocessors do not strictly fall under one of the aforementioned categories but provide some subset of these capabilities. Thus, we will treat the different preprocessors according to their declared _input/output types_. For example:

- A pure fetcher downloading the content from the given URIs will specify _“URI”_ as the input type and _“document”_ as the output type;
- A pure converter that does not handle neither fetching nor chunking will specify _“document”_ as the input type and _“text”_ as the output type;
- A pure chunker will specify _“text”_ as the input type and _“chunks”_ as the output type;
- A universal all-in-one tool that handles the entire process or parts thereof will specify _[“URI”, “document”, “text”]_ as the input type and _“chunks”_ as the output type.

The available input/output types will cover two categories of use cases:

1. Data passed by value: the data is explicitly provided to the document preprocessor, e.g., “document”, “text”, “chunks” as in the above examples.
2. Data passed by reference: the data is not explicitly provided to the document preprocessor - instead, the preprocessor is in charge of retrieving the data according to the given reference or identifier. Examples of such identifiers include, but are not limited to web links, S3 bucket links, URIs, regexes, links to directory services, and more.

Depending on the capabilities of the underlying tool, a document preprocessor may support either or both cases.

In addition, a preprocessor dealing with documents or text will provide a list of supported formats, for example _[“pdf”, “docx”]_ for documents and _[“md”, “json”]_ for text.


### Preprocessing as a standalone service
A dedicated endpoint will be defined for document preprocessing in a way similar to the ‘/v1/inference’ endpoints.
In order to fully leverage the existing document preprocessors’ functionality and for the sake of user convenience, the API will make it possible to also specify paths to local/remote directories or document stores containing multiple documents. This is in contrast to the current interface for RAG ingestion that only accepts paths to individual documents, i.e., the documents can only be specified one-by-one.
Please refer to the endpoint definition below for more details.

### Preprocessing as a part of the RAG ingestion process
To utilize the preprocessing functionality, the RAG insert call will be extended to include a chain of document preprocessors as an optional parameter. The chain must end with a chunk-supporting preprocessor, and every two adjacent preprocessors must agree on the input-output format. Typically, but not necessarily, the first preprocessor in the chain will declare a by-reference input type (e.g., a URI as shown above).
Using the examples from above:

- [“fetcher”, “converter”, “chunker”] is a valid chain since: 1) the fetcher’s output is a subset of the converter’s input, and the same relationship holds for the converter and the chunker; 2) it ends with the chunker.
- [“converter”, “chunker”] is also a valid chain as long as the user provides raw documents rather than URLs as input.
- [“fetcher”, “chunker”] and [“converter”, “fetcher”, “chunker”] are invalid chains since the fetcher’s output and the chunker’s input do not necessarily match (unless the deployment is limited to plain text-only URIs).
- [“all-in-one-tool”] constitutes a valid preprocessing chain.

If no chain is provided, the default chain defined in the runtime configuration will be used.
The Llama Stack server will be in charge of validating the consistency of the input chain.

### Adding, managing and removing document preprocessors
The Llama Stack server will expose an interface for registering and unregistering preprocessors similar to the currently available APIs for models and vector DBs. As with other provider types, both inline and remote preprocessors will be supported. For details please check out the endpoint definitions below.

## Endpoints
### Preprocessing endpoint
URL: `POST /v1/preprocess`
Request Body: 

```
{
	"preprocessor_id": "string",
	"preprocessor_inputs": [
		{
			"preprocessor_input_id": "string",
			"preprocessor_input_type": "string",
			"path_or_content": "string",
		}
	],
	"options": {
		"option1": null,
		"option2": null,
	}
}

```

- preprocessor_id - the ID of the document preprocessor to use.
- preprocessor_inputs - a list of documents or document paths to process. Each input specification contains the following:
    - preprocessor_input_id - a unique identifier of the document/path
    - path_or_content - can contain one of the following:
        - The URI of a single document
        - A string identifier for multiple documents, including, but not limited to a remote directory path, a directory service URL, a regex, etc.
        - The document itself
    - preprocessor_input_type - an optional parameter explicitly specifying the type of path_or_content
- options - an optional dictionary of preprocessor-specific parameters. Some examples may include:
    - Desired conversion output format
    - Chunk size
    - The path to write/store the results

A sample llama-stack-client API call:

```
remote_dirs = [ ... ]
preprocessor_inputs = [
	DocumentDirPath(document_path_id=f"path_{i}",
   path_or_content=dir_path
) 
for i, dir_path in enumerate(remote_dirs)
]
chunks = client.preprocessing.preprocess(
	preprocessor_id="docling_remote",
	preprocessor_inputs=preprocessor_inputs,
	chunk_size=512,
)
```

### Adding a new document preprocessor
URL: `POST /v1/preprocessors`
Request Body: 

```
{
	"preprocessor_id": "string",
	"provider_id": "string",
	"metadata": {
		"option1": null,
		"option2": null,
	}
}
```

- preprocessor_id - the ID of the preprocessor to register, unique among all preprocessors.
- provider_id - the globally unique ID of this provider.
- metadata - an optional dictionary of preprocessor-specific settings. For remote preprocessors, this will contain a URI alongside other parameters.

A sample llama-stack-client API call:

```
client.preprocessors.register(
    preprocessor_id="docling_remote",
    provider_id="docling-remote"
    args={"defaut_output_format": "json", "default_chunk_size": 512},
)
```

### Viewing the registered document preprocessors
URL: `GET /v1/preprocessors`
Response: 

```
{
  "data": [
    {
      "preprocessor_id": "string",
      "metadata": {
        "option1": null,
        "option2": null,
      }
    }
  ]
}
```

A sample llama-stack-client API call:

`client.preprocessors.list()`

### Viewing the settings of a given document preprocessor
URL: `GET /v1/preprocessors/{preprocessor_id}`
Response: 

```
{
  "preprocessor_id": "string",
  "metadata": {
    "option1": null,
    "option2": null,
  }
}
```

A sample llama-stack-client API call:

`client.preprocessors.retrieve("docling_remote")`

### Unregistering a document preprocessor
URL: `DELETE /v1/preprocessors/{preprocessor_id}`

A sample llama-stack-client API call:

`client.preprocessors.unregister("docling_remote")`

### Modified RAG ingestion endpoint
The RAG ingestion API (/v1/tool-runtime/rag-tool/insert) will be modified as follows:

- A new list parameter `preprocessing_chain` will be added. Each entry on the list will contain a preprocessor ID and a dictionary of runtime parameters as expected by the preprocessing endpoint defined above.
- The `chunk_size_in_tokens` parameter will be deprecated and an equivalent parameter will be obtained from the respective document preprocessor parameters.
- The documents list will also be allowed to contain URIs of directories and document collections.

The following is the full definition of the updated ingestion request parameters:

```
{
  "documents": [
    {
      "document_id": "string",
      "content": "string",
      "mime_type": "string",
      "metadata": {
        "property1": null,
        "property2": null
      }
    }
  ],
  "vector_db_id": "string",
  "preprocessing_chain": [
    {
      "preprocessor_id": "string",
      "options": {
	  "option1": null,
	  "option2": null,
	}
    }
  ]
}
```

The following is a sample llama-stack-client API call illustrating the changes above. It assumes that the components of the currently hard-coded preprocessing process (downloading remote documents with HTTPX, parsing PDF files with pypdf and using a simple chunking scheme with overlapping chunks) are defined and registered as preprocessors.

```
remote_dirs = [ ... ]
document_dir_paths = [
	DocumentDirPath(document_path_id=f"path_{i}",
   path_or_content=dir_path
) 
for i, dir_path in enumerate(remote_dirs)
]
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id="my_vector_id",
    preprocessing_chain=[
	Preprocessor(id="inline_httpx_fetcher"),
	Preprocessor(id="inline_pypdf_converter"),
Preprocessor(id="inline_overlapping_chunks_chunker", chunk_size=512)
    ]
)
```

### Sample provider configuration

```
...
providers:
  preprocessors:
  - provider_id: docling
    provider_type: inline::docling
    config: {}      
  - provider_id: docling-remote
    provider_type: remote::docling
      config:
        uri: 'http://localhost:8888/'
	 to_format: 'md'
```

### Limitations
This proposal solely focuses on loading, converting and chunking the documents, but leaves embedding calculation, storing the documents and the embeddings to the vector (or non-vector) DB, and advanced management of the vector DB out of its scope. These concerns will be addressed in future RFCs.
