import os

import streamlit as st
from dotenv import load_dotenv

from agents.returns_agent import ReturnsPolicyAgent
from models.groq_provider import GroqProvider

load_dotenv()

EXAMPLES = (
    "What is the return window for electronics?",
    "Do opened electronics have a restocking fee?",
    "I paid $300 for a sealed blender delivered 10 days ago. How much is my refund?",
    "Opened headphones cost $200 and arrived 12 days ago. How much can I get back?",
    "Is the shipping fee refundable?",
)


def build_agent(api_key: str, model: str) -> ReturnsPolicyAgent:
    provider = GroqProvider(api_key, model) if api_key else None
    return ReturnsPolicyAgent(provider)


def submit(agent: ReturnsPolicyAgent, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        result = agent.run(prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "details": result}
        )
    except ValueError as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"The request could not be processed: {exc}"}
        )


def main() -> None:
    st.set_page_config(page_title="Returns Policy RAG Agent", page_icon="↩", layout="wide")
    st.title("Returns Policy RAG Agent")
    st.caption("Grounded policy answers and deterministic refund estimates")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Optional Groq extraction")
        api_key = os.getenv("GROQ_API_KEY") or st.text_input("Groq API key", type="password")
        model = st.selectbox(
            "Model",
            ("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
            index=0,
        )
        st.write("The local retriever and refund calculator work without an API key.")
        st.divider()
        st.subheader("Examples")
        selected_example = next(
            (example for example in EXAMPLES if st.button(example, use_container_width=True)),
            None,
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    agent = build_agent(api_key, model)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            details = message.get("details")
            if details:
                with st.expander("Execution details"):
                    st.write("Extracted parameters", details["parameters"])
                    st.write("LLM extraction", details["llm_status"])
                    if details["tool_result"]:
                        st.write("Refund tool", details["tool_result"])
                    if details["policy_matches"]:
                        st.write(
                            "Retrieved policies",
                            [
                                {
                                    "id": match["policy"]["id"],
                                    "score": match["score"],
                                }
                                for match in details["policy_matches"]
                            ],
                        )

    prompt = st.chat_input("Ask about a return policy or request a refund estimate")
    if selected_example:
        submit(agent, selected_example)
        st.rerun()
    if prompt:
        submit(agent, prompt)
        st.rerun()


if __name__ == "__main__":
    main()
