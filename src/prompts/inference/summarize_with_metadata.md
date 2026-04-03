You are writing a comprehensive summary using the memory below.

MEMORY CONTEXT:
{context}

MEMORY METADATA:
- Total sessions in memory: {total_sessions}
- Sessions in context: {sessions_in_context} ({coverage_pct:.0f}% coverage)

Question: {question}

You have {coverage_pct:.0f}% of the total content. Write the most complete summary
you can from what's provided. If you need more detail from a specific session
visible in the context (e.g. S12), call get_more_context(session_id) — ONE call maximum.

CONFLICT RESOLUTION:
If multiple facts contradict each other on the same topic, ALWAYS choose
the fact with the HIGHEST session number (e.g. S75 beats S41) — it is
the most recent update. Ignore superseded values.

Write a detailed, chronological summary of at least 800 words.
Answer: