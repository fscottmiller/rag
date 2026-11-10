* **What was simplified:**
  - Consolidated cascading delete logic into a single SQL statement in `SQLiteStore`.
  - Optimized existence checks by replacing `SELECT COUNT(*)` with `SELECT 1 ... LIMIT 1`.

* **Why the previous implementation was unnecessarily complex:**
  - **Cascading deletes:** The prior implementation fetched `chunk_ids` using a `SELECT` query into a Python list, then used `executemany` with a `DELETE` statement for each ID. This introduced unnecessary Python-side memory overhead and N+1 database round-trips.
  - **Existence checks:** `SELECT COUNT(*)` requires a full index or table scan to compute an aggregate number, even when only a boolean existence check was needed. This became a bottleneck for `_chunk_count_all` and `_drop_empty_vector_table`.

* **What changed:**
  - Modified `replace_document` and `delete_document` to use a single query for cascading deletion from `vec_chunks`: `DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)`.
  - Modified `_chunk_count_all` and `_drop_empty_vector_table` to use `SELECT 1 FROM ... LIMIT 1`, and evaluate `fetchone() is not None` to derive the boolean response.

* **Validation performed and its result:**
  - All tests pass (run using `uv run --python 3.11 pytest tests/`).
  - Formatting correctly checked with `ruff`.
  - Coverage verified at `96.36%`, fulfilling the minimum `95%` requirement.

* **Approximate lines or concepts removed, when meaningful:**
  - Removed ~20 lines of manual Python mapping/fetching logic.
  - Eliminated the concept of manual Python N+1 deletion for SQL cascading cleanup.

* **Any limitations or follow-up threshold:**
  - The single subquery deletion approach relies on `document_id` being correctly filtered which is safe here. If there are other multi-layer dependent tables added in the future, similar optimizations must be consistently applied instead of falling back to Python-side processing.
