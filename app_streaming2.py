import html
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

index_name = "welsh-full"
embeddings = OpenAIEmbeddings()
model_name = "gpt-3.5-turbo"
# model_name = "gpt-4-0125-preview"

vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

template = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context (delimited by XML tags) to answer the question. \
<context>
{context}
</context>

Question: {question}

If the provided context doesn't include the answer to the question, just say that you don't know. \
Do not force answering the question. \

When possible, extract actionable tips from the context. \

--
When you answer the question, use the following structure: \

**Overview**: <Quick overview here> \
**Key points**: <key elements (e.g. tips, mistakes, lessons, etc) of the answer in bullet points here> \

**Actionable Tips**: <actionable tips here> \

--
Your writing style and language must be the same as in the provided context.
Use simple and direct language.

"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name=model_name, temperature=0.1)


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


def format_docs(docs):
    # Assuming 'format_docs' formats the documents for the context,
    # You might need to adapt this function based on how you intend to format your documents.
    return " ".join([doc.page_content for doc in docs])


def get_advanced_response(question, retriever, llm):
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # output = ""
    for chunk in rag_chain_with_source.stream(question):
        # streaming metadata (urls)
        if "context" in chunk:
            urls = [
                (cnt.metadata["url"], cnt.metadata["header"])
                for cnt in chunk["context"]
            ]
            yield "metadata", urls

        # Streaming only "answer" content
        if "answer" in chunk:
            # output += chunk["answer"]
            yield "answer", chunk["answer"]
    # yield "answer", output  # At the end, yield the full accumulated answer


user_query = st.text_input("Type your question here...")
if user_query:
    with st.chat_message("assistant"):

        links_placeholder = st.empty()
        answer_placeholder = st.empty()
        full_response = ""

        for content_type, content in get_advanced_response(user_query, retriever, llm):
            if content_type == "metadata":
                urls_markdown = generate_links_html(content)
                links_placeholder.markdown(
                    f"Sources found: <br/>{urls_markdown}", unsafe_allow_html=True
                )

            if content_type == "answer":
                full_response += content
                answer_placeholder.markdown(full_response + "▌")

        answer_placeholder.markdown(full_response)
