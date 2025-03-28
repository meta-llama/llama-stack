---
orphan: true
---
# SQLite-Vec

[SQLite-Vec](https://github.com/asg017/sqlite-vec) is an inline vector database provider for Llama Stack. It
allows you to store and query vectors directly within an SQLite database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features

- Lightweight and easy to use
- Fully integrated with Llama Stacks
- Uses disk-based storage for persistence, allowing for larger vector storage

### Comparison to faiss 

SQLite-Vec is a lightweight alternative to Faiss, which is a popular vector database provider.
While faiss is a powerful, fast, and lightweight in line provider, faiss reindexes the 
entire database when a new vector is added. SQLite-Vec is a disk-based storage provider 
that allows for larger vector storage and handles incremental writes more efficiently.

sqlite-vec is a great alternative to faiss when you need to execute several writes to the 
database.

Consider the histogram below in which 10,000 randomly generated strings were inserted 
in batches of 100 into both `faiss` and `sqlite-vec` using `client.tool_runtime.rag_tool.insert()`.

```{image} ../../../../_static/providers/vector_io/write_time_comparison_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss write times
:width: 400px
```

You will notice that the average write time for `sqlite-vec` was 788ms, compared to 
47,640ms for faiss. While the number is jarring, if you look at the distribution, you'll notice that it is rather uniformly spread across the [1500, 100000] interval.

```{image} ../../../../_static/providers/vector_io/write_time_sequence_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss write times
:width: 400px
```
For more information about this discussion see [the GitHub Issue](https://github.com/meta-llama/llama-stack/issues/1165) 
where this was discussed.

## Usage

To use sqlite-vec in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use SQLite-Vec.
3. Start storing and querying vectors.

## Installation

You can install SQLite-Vec using pip:

```bash
pip install sqlite-vec
```

## Documentation

See [sqlite-vec's GitHub repo](https://github.com/asg017/sqlite-vec/tree/main) for more details about sqlite-vec in general.
