from flask import Flask, render_template, request, redirect, url_for
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import json

app = Flask(__name__)

#LLMs 

clerk = OpenAI(temperature=0.7, max_tokens=750, presence_penalty=0.25, frequency_penalty=0.1 )
assistant = OpenAI(temperature=0.7, max_tokens=256, model_name="text-davinci-003")

#Prompt Templates
clerk_default_template = """
You are a music nerd working as a clerk in a digital record store. Customers often ask you to teach them about different genres, sub-genres, and how various albums have influenced each other. Your job is to provide them with relevant and insightful editorial based on their questions. If the question asks for recommendations based on a specific artist or album, avoid recommending the same artist or album in your response, since they are obviously already familiar with them. Your response should be between 500 and 700 words and formatted with HTML.

To help users find albums, you should format your response so that each album and artist is wrapped in the <strong> tag. 

Question: {question}
Response:
"""
clerk_default_prompt = PromptTemplate(input_variables=["question"], template=clerk_default_template)
clerk_chain = LLMChain(llm=clerk, prompt=clerk_default_prompt)

assistant_template = """
You are an assistant that is vetting questions from an audience before presenting them to an expert about music history to answer. Given an input from the audience, form a valid question with proper grammar that appoximates the intent of the audience member 

input: {input}
assistant:
"""
assistant_prompt = PromptTemplate(input_variables=["input"], template=assistant_template)
assistant_chain = LLMChain(llm=assistant, prompt=assistant_prompt)

response_chain = SimpleSequentialChain(chains=[assistant_chain,clerk_chain], verbose=True)

@app.route("/", methods=["POST", "GET"])
def index():
  if request.method == "POST": 
    input = request.form["input"]
    response = response_chain.run(input)
    print(response)
    return render_template('index.html', response=response)
  else: 
    return render_template('index.html')

@app.route("/query", methods=["POST"])
def query():
  # First we convert the query into a question
  input = json.loads(request.data.decode())['query']
  response = response_chain.run(input)

  return response