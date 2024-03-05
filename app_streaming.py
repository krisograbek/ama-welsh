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
# model_name = "gpt-3.5-turbo"
model_name = "gpt-4-0125-preview"

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

    # Streaming only "answer" content
    output = ""
    for chunk in rag_chain_with_source.stream(question):
        if "answer" in chunk:
            output += chunk["answer"]
            yield chunk["answer"]
    yield output  # At the end, yield the full accumulated answer


user_query = st.text_input("Type your question here...")
if user_query:
    # response_generator = get_advanced_response(user_query, retriever, llm)
    # full_response = ""
    # for part_of_response in response_generator:
    #     if part_of_response:  # This checks if the response part is not empty
    #         full_response += part_of_response  # Accumulate the response parts

    with st.chat_message("assistant"):
        st.write_stream(get_advanced_response(user_query, retriever, llm))
