You are answering a question whose answer is a grounded list or set of items using the retrieved memory below.

{context}

Question: {question}

Rules:
1. Identify every item type requested by the question.
2. If the question names multiple item types with words like "or" or punctuation like "/", include grounded items from each named type.
3. Return all distinct grounded items that answer the question, not just the single strongest item.
4. Prefer explicit items supported by retrieved facts or nearby raw context from the same source.
5. Deduplicate aliases or repeated mentions of the same grounded item.
6. Do not say an item is missing if it is explicitly present in the context.
7. If no grounded items are present, say so briefly.
8. Prefer a compact comma-separated list unless a short numbered list is clearly easier to read.

Answer based only on the context above. Be concise and direct.
Answer:
