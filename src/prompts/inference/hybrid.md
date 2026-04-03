You are answering a question using the retrieved memory below.

{context}

Question: {question}

The context has two sections:
- RETRIEVED FACTS: structured extracted facts (check these first)
- RAW CONTEXT: source text excerpts with full details

Step-by-step:
1. Find relevant facts in RETRIEVED FACTS
2. If facts lack specific details (step numbers, coordinates, exact names, sequences) — check RAW CONTEXT for the missing details
3. Combine both to answer precisely
4. If the answer requires a short bridge across 2-3 facts or raw lines, do that explicitly instead of saying the answer is missing
5. If multiple lines from the SAME EPISODE describe the same event, entity, or outcome, combine them into one grounded answer
6. If RAW CONTEXT explicitly names an action, command, or ordered step outcome, trust that explicit label over a second-order inference from surrounding state descriptions
7. For short ordered action sequences, compute the net effect from the stated actions unless the question explicitly asks about a separate object/state change

CONFLICT RESOLUTION:
If multiple facts contradict each other, do NOT treat a higher session number as newer by default.
Use recency only when the context explicitly supports it, such as:
- later dated or clearly later conversational updates
- explicit supersedes/replaced/updated wording
- currentness markers
- later explicit updates inside the SAME EPISODE
For static document bundles or unordered fact lists, treat session numbers as identifiers, not time.
If no explicit update signal resolves the conflict, answer from the strongest locally connected evidence in the retrieved context and do NOT use outside world knowledge to break the tie.

Answer based only on the context above. Be concise and direct.
If the facts don't contain the answer, say what you can infer from them.
Answer:
