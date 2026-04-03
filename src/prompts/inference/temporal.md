You are answering a question using ONLY the retrieved memory facts below.
{context}
Question: {question}

To answer temporal questions:
1. Find all relevant dates and events in the facts above
2. List them chronologically
3. If asked "how long between X and Y": identify both dates, compute the difference
4. reference_date for relative calculations: use the most recent date mentioned in facts
5. Answer with exact dates/durations from facts

CONFLICT RESOLUTION:
If multiple facts contradict each other on the same topic, ALWAYS choose
the fact with the HIGHEST session number (e.g. S75 beats S41) — it is
the most recent update. Ignore superseded values.

Answer: