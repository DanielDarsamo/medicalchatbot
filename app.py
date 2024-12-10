from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.llms.base import LLM
from pydantic import BaseModel
from groq import Groq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings=download_hugging_face_embeddings()


index_name= "medicalbot"

#doc upsert
from langchain.vectorstores import Pinecone
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever =docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

class GroqLLM(LLM, BaseModel):  # Inherit from both LLM and BaseModel
    model: str = "llama3-8b-8192"  # Define as a pydantic field
    
    # Use __annotations__ to indicate that 'client' is not a pydantic field
    client: Groq = None  # This ensures that 'client' is not managed by pydantic

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Ensure BaseModel initialization
        # Initialize the Groq client inside __init__
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def _call(self, prompt: str, stop=None):
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self):
        return "Groq"

# Initialize the custom LLM
llm = GroqLLM()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


