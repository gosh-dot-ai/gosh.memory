You are answering a question using the retrieved memory below.

{context}

MEMORY METADATA:
- Sessions searched: {sessions_in_context} of {total_sessions} total

Question: {question}

The context has two sections:
- RETRIEVED FACTS: structured extracted facts
- RAW CONTEXT: source text excerpts with full details

Step-by-step:
1. Check RETRIEVED FACTS for the answer
2. If facts lack details (step numbers, coordinates, names, sequences), check RAW CONTEXT
3. If BOTH facts and raw context still don't contain enough to answer confidently,
   call get_more_context with a session number from the context (e.g. S12)
   to retrieve the FULL text of that session. You get ONE call maximum.
4. Answer based on all available information

CONFLICT RESOLUTION:
If multiple facts contradict each other on the same topic, ALWAYS choose
the fact with the HIGHEST session number (e.g. S75 beats S41) — it is
the most recent update. Ignore superseded values.

Answer concisely and directly.
If the facts don't contain the answer, say what you can infer.