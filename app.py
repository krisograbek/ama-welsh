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
    link_template = '<a href="{url}" target="_blank" style="display: inline-block; margin: 2px; padding: 2px; border: 1px solid #ccc; border-radius: 4px;">{title}</a><br>'
    links_html = "".join(
        [
            link_template.format(url=url, title=escape_dollars(html.escape(title)))
            for url, title in urls_and_titles
        ]
    )
    return links_html


# Set up the Streamlit page
st.set_page_config(page_title="Justin Welsh AI")
st.title("Justin Welsh AI AMA 📈")

# Initialize session state for messages if not already done
# if "messages" not in st.session_state:
#     st.session_state.messages = []

if user_prompt := st.chat_input("Ask Justin AI..."):
    # st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response, urls_and_titles = get_rag_with_sources(user_prompt)
        # Display the response
        st.markdown(response)
        # Generate HTML for the URLs and display them in a separate markdown call
        urls_markdown = generate_links_html(urls_and_titles)
        # print(urls_markdown)
        st.markdown(
            f"For more details, check them out:<br/> {urls_markdown}",
            unsafe_allow_html=True,
        )

    # st.session_state.messages.append({"role": "assistant", "content": response})


# else:
#     st.write("Please upload a CSV file to proceed.")
