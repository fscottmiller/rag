## Correlated Subqueries Over Group By
**Learning:** In SQLite, a correlated scalar subquery can replace `LEFT JOIN ... GROUP BY` for per-row aggregate counts (as in `list_documents`), avoiding an intermediate grouped table by leveraging an existing index such as `UNIQUE(document_id, ordinal)` on the chunks table.
**Action:** When a query groups an entire table just to retrieve simple per-row aggregates, consider a correlated scalar subquery if a suitable index exists.
