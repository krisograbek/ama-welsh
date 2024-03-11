from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

index_name = "welsh-full"
embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_with_sources(query, model=3):
    model_name = "gpt-3.5-turbo" if model == 3 else "gpt-4-0125-preview"
    llm = ChatOpenAI(model_name=model_name, temperature=0.1)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    for chunk in rag_chain_with_source.stream(query):
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


if __name__ == "__main__":
    print(get_rag_with_sources("How to bake an apple pie?"))
