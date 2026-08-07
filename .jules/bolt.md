## 2023-10-24 - Correlated Subqueries Over Group By
**Learning:** In SQLite, replacing `LEFT JOIN ... GROUP BY` with a correlated subquery dramatically improves performance for querying aggregate counts (like `list_documents`) since it avoids forming a massive temporary grouped table, leveraging existing indexes like `UNIQUE(document_id, ...)` instead.
**Action:** When a query involves grouping the entire table to retrieve simple row aggregates, use a correlated scalar subquery if an index exists.
