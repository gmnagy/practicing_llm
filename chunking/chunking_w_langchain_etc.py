from root import data_input_path

# Process the PDF Content
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#
loader = PyPDFLoader(data_input_path() + "1810.04805v2.pdf")
documents = loader.load()
#
print(len(documents))
print(documents[0:1000])

# Perform Native Chunking(RecursiveCharacterTextSplitting)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)
#
naive_chunks = text_splitter.split_documents(documents)
for chunk in naive_chunks[0:1]:
  print(chunk.page_content+ "\n")

# Perform Semantic Chunking
# Instantiate Embedding Model
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Setup the API Key for LLM
# TODO: pip google.colab does not work
# from google.colab import userdata
# from groq import Groq
# from langchain_groq import ChatGroq
#
# groq_api_key = userdata.get("GROQ_API_KEY")


from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
#
semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])
#
for semantic_chunk in semantic_chunks[8:10]:
  # if "Effect of Pre-training Tasks" in semantic_chunk.page_content:
  #   print(semantic_chunk.page_content)
  #   print(len(semantic_chunk.page_content))
    print(len(semantic_chunk.page_content))
    print(semantic_chunk.page_content)

# Instantiate the Vectorstore
from langchain_community.vectorstores import Chroma
semantic_chunk_vectorstore = Chroma.from_documents(semantic_chunks, embedding=embed_model)




