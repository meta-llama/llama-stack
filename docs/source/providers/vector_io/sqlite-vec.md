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

### Comparison to Faiss

The choice between Faiss and sqlite-vec should be made based on the needs of your application,
as they have different strengths.

#### Choosing the Right Provider

Scenario | Recommended Tool | Reason
-- |-----------------| --
Online Analytical Processing (OLAP) | Faiss           | Fast, in-memory searches
Online Transaction Processing (OLTP) | sqlite-vec      | Frequent writes and reads
Frequent writes | sqlite-vec      | Efficient disk-based storage and incremental indexing
Large datasets | sqlite-vec      | Disk-based storage for larger vector storage
Datasets that can fit in memory, frequent reads | Faiss | Optimized for speed, indexing, and GPU acceleration

#### Empirical Example

Consider the histogram below in which 10,000 randomly generated strings were inserted
in batches of 100 into both Faiss and sqlite-vec using `client.tool_runtime.rag_tool.insert()`.

```{image} ../../../../_static/providers/vector_io/write_time_comparison_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss write times
:width: 400px
```

You will notice that the average write time for `sqlite-vec` was 788ms, compared to
47,640ms for Faiss. While the number is jarring, if you look at the distribution, you can see that it is rather
uniformly spread across the [1500, 100000] interval.

Looking at each individual write in the order that the documents are inserted you'll see the increase in
write speed as Faiss reindexes the vectors after each write.
```{image} ../../../../_static/providers/vector_io/write_time_sequence_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss write times
:width: 400px
```

In comparison, the read times for Faiss was on average 10% faster than sqlite-vec.
The modes of the two distributions highlight the differences much further where Faiss
will likely yield faster read performance.

```{image} ../../../../_static/providers/vector_io/read_time_comparison_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss read times
:width: 400px
```

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
