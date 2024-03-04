import streamlit as st
from lc_helpers import get_rag_response, get_rag_with_sources
from dotenv import load_dotenv

load_dotenv()  # for LangSmith

# Set up the Streamlit page
st.set_page_config(page_title="Justin Welsh AI")
st.title("Justin Welsh AI AMA 📈")

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

if user_prompt := st.chat_input("Ask Justin AI..."):
    # st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate and display response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response, urls = get_rag_with_sources(user_prompt)
        # Create a markdown string with clickable URLs
        urls_markdown = "".join(
            [
                f'<a href="{url}" target="_blank" style="display: inline-block; margin: 2px; padding: 2px; border: 1px solid #ccc; border-radius: 4px;">{url}</a><br>'
                for url in urls
            ]
        )
        markdown_content = f"{response}<br><br>{urls_markdown}"

        message_placeholder.markdown(markdown_content, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})


# else:
#     st.write("Please upload a CSV file to proceed.")
