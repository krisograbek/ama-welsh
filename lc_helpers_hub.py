from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

index_name = "welsh-full"
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_response(query):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(query)
    return response


def get_rag_with_sources(query):
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    response = rag_chain_with_source.invoke(query)

    context = response["context"]
    answer = response["answer"]

    urls = [(cnt.metadata["url"], cnt.metadata["header"]) for cnt in context]

    # print(urls)
    # print(context)
    return answer, urls


if __name__ == "__main__":
    print(get_rag_response("How to bake an apple pie?"))
