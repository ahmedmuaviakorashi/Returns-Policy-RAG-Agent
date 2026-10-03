# Returns Policy RAG Agent

A Streamlit support assistant that retrieves answers from a local returns-policy knowledge base and calls a deterministic refund calculator when a customer provides enough information.

The application works without an API key. A Groq model can optionally fill structured fields that the local parser cannot extract. Model output never performs the refund arithmetic and must pass the same validation as locally extracted values.

## What it demonstrates

- TF-IDF retrieval over a versioned JSON policy collection
- category- and policy-aware ranking with visible source identifiers
- structured extraction from ordinary customer questions
- optional Groq JSON extraction with a safe local fallback
- decimal-based refund calculations and input validation
- a Streamlit chat interface with inspectable execution details
- automated tests and GitHub Actions checks

## Architecture

```text
Customer message
      |
      +--> local parameter parser
      |         |
      |         +--> optional Groq extraction for missing fields
      |
      +--> TF-IDF policy retriever --> ranked policy citation
      |
      +--> validated refund tool --> eligibility, fee, refund
                |
                +--> grounded response
```

Retrieval and calculation are intentionally separate. Policies determine what information is relevant; the calculator applies the configured return window and restocking rate. This keeps monetary decisions deterministic and testable.

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

No credentials are required for local retrieval and refund calculations.

To enable Groq-assisted extraction:

```bash
cp .env.example .env
```

Add your key to `.env` as `GROQ_API_KEY`. The default model is `openai/gpt-oss-120b`; it can be changed in the sidebar. Never commit the `.env` file.

## Test

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

The test suite covers parameter ambiguity, invalid monetary inputs, policy ranking, missing-field prompts, return-window enforcement, and restocking-fee calculations.

## Example

Input:

```text
Opened headphones cost $200 and arrived 12 days ago. How much can I get back?
```

Result:

```text
Estimated merchandise refund: $180.00. Restocking fee: $20.00.
```

The response also identifies the retrieved policy, while the execution panel shows extracted parameters, retrieval scores, and the calculator result.

## Project structure

```text
agents/                 Request orchestration and response composition
data/                   Policy text and refund configuration
models/                 Retrieval and optional Groq integration
tools/                  Deterministic refund calculation
utils/                  File loading and local parameter extraction
tests/                  Unit and integration tests
app.py                  Streamlit entry point
```

## Scope

The bundled rules are demonstration data, not a real retailer's terms. Refund estimates exclude tax, discounts, shipping, payment processing, and exceptions not represented in `data/config.json`. The application does not approve or issue refunds.

## License

MIT
