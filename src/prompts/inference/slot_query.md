You are answering a direct slot-filling or attribute question using the retrieved memory below.

{context}

Question: {question}

Rules:
1. Identify the requested slot, value, item type, or named attribute from the question.
2. Use RETRIEVED FACTS first.
3. If the context includes RAW SLOT CANDIDATES, treat them as grounded candidate strings extracted from the same source text, not as guesses.
4. Prefer a candidate that is explicitly supported by nearby same-episode context.
5. If multiple grounded candidates appear, choose the one that best matches the requested slot and the local evidence.
6. If a direct grounded candidate is present in the context, do not answer that it is missing.
7. If the context truly does not contain a grounded candidate, say so briefly.

Answer based only on the context above. Be concise and direct.
Answer:

CONFLICT RESOLUTION:
If multiple candidates conflict, prefer the candidate with the strongest same-episode grounding and the clearest match to the requested slot. Do not invent a value to break the tie.
