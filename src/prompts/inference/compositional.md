You are answering a multi-constraint question using the retrieved memory below.

{context}

Question: {question}

Rules:
1. Identify the full constraint set in the question before answering.
2. Return the single best answer that satisfies the most important constraints jointly, not a generic answer that satisfies only one side.
3. Prefer a concrete grounded option over a broader category or environment description.
4. If the answer requires combining two or three nearby grounded clues from the same episode, do that explicitly.
5. If one candidate satisfies only the activity constraint and another satisfies only the secondary constraint, do not answer with either partial match as if it fully solves the question.
6. Use RAW CONTEXT only to recover specific wording or locally linked details for an answer already supported by the retrieved facts.
7. If no single grounded answer satisfies the full constraint set, say so directly instead of promoting a partial match.
8. If the question asks for an activity, action, hobby, or thing to do, answer with the activity itself, not a place, room, venue, or facility where some activity could happen.
9. Prefer a candidate whose core activity is explicitly grounded for the target person over an environment description that only supports the secondary constraint.

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
Answer:
