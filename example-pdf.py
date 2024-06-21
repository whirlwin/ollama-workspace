from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

from langchain_community.vectorstores import Chroma
from flask import Flask, request
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

llm = Ollama(model="llama3")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    #chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
    chunk_size=256, chunk_overlap=80, length_function=len, is_separator_regex=False
)

#response = llm.invoke("Tell me a cat joke")
#print(response)

@app.route("/pdf", methods=['POST'])
def pdfPost():
    json = request.json
    query = json.get("query")
    loader = PDFPlumberLoader("data/example-invoice.pdf")
    docs = loader.load_and_split()
    chunks = text_splitter.split_documents(docs)
    print(len(chunks))
    print("FOOBAR")

    #vectorstore = Chroma(persist_directory="data", embedding_function=embedding)
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory="db"
    )
    vectorstore.persist()

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 1,
            "score_threshold": 0.5,
        }
    )

    raw_prompt = PromptTemplate.from_template("""
        <s>[INST] You are a machine good at searching documents.
        If you don't have an answer from the provided information, say so [/INST]
        [INST] {input}                              
               Context: {context}
               Answer:
        [/INST]
    """)

    document_chain = create_stuff_documents_chain(llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})

    sources = []
    for doc in result["context"]:
        sources.append({"source": doc.metadata["source"], "page_content": doc.page_content})

    return {"answer": result["answer"], "sources": sources}
    return { "docs len": len(docs) }

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True);

if __name__ == "__main__":
    start_app();



