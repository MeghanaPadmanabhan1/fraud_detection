# Genie space: "Fraud claims: ask the data"

A Genie space over the Lakeflow gold tables (`image_authenticity.lakeflow.*`) so a claims manager can ask questions in plain English.

- `genie_space_definition.json`: the space as created (tables, instructions, sample questions, two example queries).
- Real answers, with the SQL Genie wrote and the rows it returned: `../evidence/03_genie_answers.md`.
- The app's **Ask the data** page calls the same space through the Genie Conversation API, including follow-up questions
  (`app/genie_client.py`, `app/views/ask.py`).
