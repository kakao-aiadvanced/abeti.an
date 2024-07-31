import os
from typing import List
from dotenv import load_dotenv

from tavily import TavilyClient

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict

load_dotenv()

retriever: VectorStoreRetriever
# llm = ChatOllama(model="llama3", format="json")
llm = ChatOpenAI(model="gpt-4o-mini")


class GraphState(TypedDict):
    question: str
    is_relevant: bool
    generation: str
    is_web_search: bool
    documents: List[str]
    is_hallucinated: bool


def index():
    global retriever

    urls = ("https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/")

    loader = WebBaseLoader(web_paths=urls)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=50)
    text_splits = text_splitter.split_documents(docs)

    vector_store = FAISS.from_documents(text_splits, OpenAIEmbeddings())
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


index()


# node1
def docs_retrieval(state: GraphState):
    print("[ Docs Retrieval ]")

    documents = retriever.invoke(state.get("question"))
    return GraphState(question=state.get("question"), documents=[doc.page_content for doc in documents])




class Relevance(BaseModel):
    relevance: bool = Field(description="True if user query and context is relatied, else False")


# node2
def relevance_check(state: GraphState):
    print("[ Relevance Check ]")

    def format_docs(docs):
        return "\n\n".join(docs)

    parser = JsonOutputParser(pydantic_object=Relevance)

    prompt = PromptTemplate(
        template="Evaluate the relevance between given question and context. The answer should be True or False, bool type. \n{format_instructions}\nQuestion: {question}\nContext: {context}\n",
        input_variables=["question", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = (
            {"context": lambda x: format_docs(state.get("documents")),
             "question": RunnablePassthrough()} | prompt | llm | parser
    )

    output = chain.invoke(state.get("question"))
    state.update({"is_relevant": output.get("relevance")})
    return GraphState(**state)


# node3
def web_search(state: GraphState):
    print("[ Web Search ]")

    tavily = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
    response = tavily.search(state.get("question"), max_results=3)
    # context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
    state.update({"documents": [obj["content"] for obj in response['results']], "is_web_search": True})
    return GraphState(**state)


# node4
def generate(state: GraphState):
    print("[ Generate ]")

    system = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "question: {question}\n\ncontext: {context}\n")]
    )

    chain = prompt | llm | StrOutputParser()

    generation = chain.invoke({"question": state.get("question"), "context": state.get("documents")})

    return GraphState(**state, generation=generation)


class Hallucination(BaseModel):
    hallucination: str = Field(description="True if hallucination exists, else False")


# node5
def hallucination_check(state: GraphState):
    print("[ Hallucination Check ]")

    parser = JsonOutputParser(pydantic_object=Hallucination)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(
        template="Create an answer to question and evaluate if hallucination exists your answer.\n{format_instructions}\nQuestion: {question}\nContext: {context}\n",
        input_variables=["question", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | parser
    )

    output = chain.invoke(state.get("question"))

    return GraphState(**state, is_hallucinated=output.get("hallucination"))


def web_search_and_retry(state, retry=3):
    search_state = web_search(state)
    eval = relevance_check(search_state)

    while retry > 0 and eval.get("is_relevant") == 'False':
        print("[ Web Search and Retry ]")

        search_state = web_search(search_state)
        eval = relevance_check(search_state)
        retry -= 1

    return eval


def answer_and_retry(state, retry=3):
    output = generate(state)
    eval = hallucination_check(output)

    while retry > 0 and eval.get("is_hallucinated") == 'True':
        print("[ Answer and Retry ]")

        output = generate(state)
        eval = hallucination_check(output)
        retry -= 1

    return eval


def print_output(state: GraphState):
    answer_format = ("\n"
                     "-----------\n"
                     "> {}\n"
                     "-----------\n"
                     "> is_relevant: {}\n"
                     "> is_web_search: {}\n"
                     "> is_hallucinated: {}\n"
                     "-----------\n")
    print(answer_format.format(
        state.get("generation"),
        state.get("is_relevant"),
        state.get("is_web_search"),
        state.get("is_hallucinated")
    ))


def main(question):
    docs_state = docs_retrieval({"question": question})

    evaluate_docs_state = relevance_check(docs_state)

    if not evaluate_docs_state.get("is_relevant"):
        evaluate_docs_state = web_search_and_retry(evaluate_docs_state)

    output = answer_and_retry(evaluate_docs_state)

    print_output(output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Question to ask", default="What is prompt?")
    args = parser.parse_args()

    main(args.question)
