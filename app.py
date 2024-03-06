import html
import streamlit as st

from lc_helpers import get_rag_with_sources

# from lc_helpers_hub import get_rag_with_sources
from dotenv import load_dotenv

load_dotenv()  # for LangSmith


def escape_dollars(title):
    # Replace $ with \$ to escape Markdown interpretation for LaTeX
    return title.replace("$", "\$")


def generate_links_html(urls_and_titles):
    # CSS styles
    styles = """
    <style>
    .link-box {
        display: inline-block;
        margin: 5px;
        padding: 8px;
        border: 1px solid #ccc;
        border-radius: 4px;
        background-color: #f0f0f0;
        font-size: 16px;
    }
    </style>
    """

    # HTML template using the class for styling
    link_template = '<a href="{url}" target="_blank" class="link-box">{title}</a><br>'

    # Generating the links HTML
    links_html = "".join(
        [
            link_template.format(url=url, title=escape_dollars(html.escape(title)))
            for url, title in urls_and_titles
        ]
    )

    # Combining styles with the links HTML
    full_html = styles + links_html
    return full_html


# Set up the Streamlit page
st.set_page_config(page_title="Justin Welsh AI")
st.title("Chat with Justin Welsh AI")
st.subheader("Trained on Saturday Solopreneur Newsletter")

# Initialize session state for messages if not already done
# if "messages" not in st.session_state:
#     st.session_state.messages = []

if user_prompt := st.chat_input("Ask Justin AI..."):
    # st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        links_placeholder = st.empty()
        answer_placeholder = st.empty()
        full_response = ""

        for content_type, content in get_rag_with_sources(user_prompt):
            if content_type == "metadata":
                urls_markdown = generate_links_html(content)
                links_placeholder.markdown(
                    f"Newsletter issues found: <br/>{urls_markdown}",
                    unsafe_allow_html=True,
                )

            if content_type == "answer":
                full_response += content
                answer_placeholder.markdown(full_response + "▌")

        answer_placeholder.markdown(full_response)
