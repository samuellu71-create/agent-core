# Storage Architecture: Why SQLite

## Decision

SQLite (WAL mode, FTS5, better-sqlite3) is the correct storage engine for agent-core's memory system.

## Workload Profile

agent-core is an **embeddable library** consumed by a future platform. It is:
- Single-process (one Node.js runtime)
- Single-tenant (one user/org per instance, or one SQLite file per user in model 2)
- Read-heavy with bursty writes (agent sessions produce memories, search is frequent)
- Small-to-medium cardinality (<100K memories per scope for initial deployment)

## Why SQLite Is Correct

### 1. Zero Infrastructure

SQLite is an in-process library — no server process, no Docker container, no credentials, no port management. agent-core starts with `import { getDb } from './db'` and the database exists.

Every alternative requires infrastructure:
- PostgreSQL: server process, TCP socket, connection pool, pg_hba.conf
- Qdrant: server process, gRPC/REST endpoint, Docker or binary
- LanceDB: native Arrow/Parquet libraries (OS-specific binaries)
- Neo4j: JVM, 1-4GB RAM baseline, bolt:// protocol

This aligns with `intention.md`: agent-core is a library, not a service. The platform chooses deployment topology.

### 2. FTS5 Gives Free BM25

SQLite FTS5 is a production-quality full-text search engine with BM25 ranking — the second-highest-weighted signal (W_BM25 = 0.8) in our hybrid scorer. No other embedded database includes this:

- LanceDB: no full-text search
- Qdrant: no full-text search (vector-only)
- Pinecone: no full-text search
- Weaviate: has BM25 but requires a server

To get BM25 without FTS5, we'd need a second database (like cognee, which runs LanceDB + SQLite + Ladybug simultaneously).

### 3. mem0 Made the Same Choice

mem0's TypeScript SDK (`vector_stores/memory.ts:17`) defaults to SQLite + `better-sqlite3` with brute-force cosine similarity scan. Their architecture: embeddings stored as BLOBs, full table scan on search, optional swap to Qdrant/Redis/PGVector.

This validates the architecture at production scale — mem0 is the most widely deployed memory framework.

### 4. Test Speed

`:memory:` mode enables 176 tests in ~600ms with zero external dependencies. No test containers, no Docker, no network mocks. `beforeEach` calls `closeDb()` to force re-initialization — each test gets a clean database in microseconds.

Every alternative degrades test ergonomics:
- PostgreSQL: requires `docker-compose up` or test containers (~5s startup)
- Qdrant: requires Docker or mocked client
- LanceDB: requires native binaries (CI matrix complications)

### 5. WAL Mode for Concurrent Reads

SQLite WAL (Write-Ahead Logging) allows concurrent reads during writes. For a single-process agent-core handling one session at a time, this eliminates contention entirely. Multiple search queries can execute while a write is in progress.

## Why Each Alternative Is Worse

### Dedicated Vector DBs (Qdrant, Pinecone, Weaviate, Milvus)

- **Operational dependency**: Contradicts embeddable-library design
- **Network latency**: ~2-20ms per search roundtrip vs <0.1ms in-process
- **No FTS5**: Would lose BM25 signal or require a second database
- **Premature optimization**: Brute-force cosine over SQLite rows is faster than network I/O for <100K rows. ANN indices only matter at >1M vectors
- **Test breakage**: 176 tests can't run without Docker or mocks

### LanceDB (cognee's choice)

- **Native dependency**: Requires Arrow/Parquet binaries (OS-specific, CI complications)
- **No FTS**: cognee runs a separate SQLite for relational data and LanceDB for vectors — two databases instead of one
- **Adapter complexity**: cognee's `LanceDBAdapter.py` is 1,354 lines with threading, subprocess proxies, and schema marshaling. Our entire memory module is simpler
- **No value-add at our scale**: LanceDB's advantage is columnar storage for large vector datasets. We do exact cosine (more accurate than ANN) on <100K rows

### PostgreSQL / PGVector

- **Requires a server**: Breaks embeddable-library constraint
- **Overkill for single-tenant**: PostgreSQL's advantages — concurrent connections, row-level security, connection pooling — solve multi-tenant problems we don't have
- **Slower for our workload**: In-process SQLite with memory-mapped I/O is faster than PGVector over TCP for single-tenant, single-process access
- **When it becomes correct**: Platform Phase 3 multi-tenancy (many users sharing one database). At that point, the `MemoryProvider` interface allows swapping storage without changing the scoring/hotness/dedup/stats layers

### MongoDB / JSON File Stores (Parlant's choice)

- **No FTS5 equivalent**: MongoDB `$text` search lacks BM25 ranking quality
- **No SQL**: Triggers for FTS5 index maintenance, `UPDATE ... SET active_count = active_count + 1` for hotness, and scope-filtered queries are natural in SQL, awkward in document stores
- **JSON files aren't production-grade**: Parlant's `json_file.py` stores work for small, rarely-written guideline sets — not for high-frequency memory writes during agent sessions

### Neo4j / Graph DBs (cognee uses for knowledge graph)

- **Wrong abstraction**: agent-core stores flat memories with metadata, not entity-relationship graphs. Forcing a graph model adds complexity without value
- **Massive overhead**: JVM, 1-4GB RAM baseline, Bolt protocol, separate process
- **When it becomes correct**: If agent-core adds a knowledge graph layer (entity resolution, relationship edges, graph traversal). Currently out of scope

## Deployment Models and Storage Choice

| Model | Description | Correct Storage |
|-------|-------------|----------------|
| **Single user, multiple sessions** | One user runs many agent sessions sharing a SQLite file | SQLite (current) |
| **Many users, isolated instances** | Each user gets their own agent-core + SQLite | SQLite (scales horizontally) |
| **Many users, shared database** | One database serves all users with row-level isolation | PostgreSQL / PGVector |

agent-core currently targets models 1 and 2. The `MemoryProvider` interface ensures that model 3 (PostgreSQL) is a storage-layer swap, not an architecture change. The 5-signal scorer, hotness tracking, dedup, stats collector, and reranker all operate on the `ScoringCandidate` struct — they don't know or care about the storage backend.

## Summary

SQLite is not a compromise — it is the architecturally correct choice for an embeddable library at current scale. It provides FTS5 (which no alternative offers in-process), zero infrastructure, sub-millisecond test initialization, and a clean upgrade path to PostgreSQL when multi-tenancy demands it.
