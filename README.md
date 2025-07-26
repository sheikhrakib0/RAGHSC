# AI Engineer (Level-1) — Technical Assessment

## It's a Simple Multilingual Retrieval-Augmented Generation (RAG) System for a online education system.


### Objective:
Designing and implementing a basic RAG pipeline capable of understanding and responding to both English and Bengali queries.
The system should fetch relevant information from a pdf document corpus and generate a meaningful answer grounded in retrieved content.

### Setup Guide:
Work Environment: Google Colab

Language: Python 3.11

Tesseract: Locally downloaded Tesseract engine to extract text from the PDF.

## Used Tools, Library, and Packages:
NumPy, Pandas
!pip install gdown  (for downloading pdf from a shared location)

##The technique used is Optical Character Recognition (OCR) with tesseract, which extracts text from the image representation of the PDF rather than relying on embedded text data.

!sudo apt install tesseract-ocr -y

!sudo apt install libtesseract-dev -y

!pip install pymupdf  #for pdf to image before OCR

!sudo apt install tesseract-ocr-ben -y  # For Bengali (ben)

!pip install pytesseract

!pip install PyPDF2  (for loading pdf)

!pip install re  (for regular expression-based operation)

!pip install transformers (to use the transformer models, here we used an embedding model)

!pip install faiss-cpu   (for vector store)

!pip install -U langchain (we used the langchain framework)

!pip install -U langchain-google-genai  (to use google gemini model)


## Sample Queries and Output:

User Question: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
Expected Answer: শুম্ভুনাথ
Model Answer: context এ এই প্রশ্নের উত্তর নেই।

User Question: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
Expected Answer: মামাকে
Model Answer: মামাকে

User Question: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
Expected Answer: ১৫ বছর
Model Answer: ১৬ বছর


User Question English: Who is called a good man in Anupam's language?
Expected Answer: শুম্ভুনাথ
Model Answer: হরিশের

User Question English: Who is referred to as the god of luck of Anupam?
Expected Answer: মামাকে
Model Answer: মামাকে

User Question English: What was Kalyani's actual age at the time of marriage??
Expected Answer: ১৫ বছর
Model Answer: ১৬ বছর

## Google Chat Model from Langchain API

ChatGoogleGenerativeAI

Google AI chat models integration:

1. To use, you must have either:

* The GOOGLE_API_KEY environment variable set with your API key, or

* Pass your API key using the google_api_key kwarg to the ChatGoogleGenerativeAI constructor.

2. Import chat model
   
from langchain.chat_models import init_chat_model

from langchain_core.prompts import PromptTemplate

4. Construct Prompt
   
def prompt_template(query, context):
  prompt_temp = PromptTemplate(
    template = """
    You are a helpful assistant.
    Answer only from the provided context.
    context: {context}
    Question: {question}
    """,
    input_variables = ["context", "question"]
  )
  prompt =  prompt_temp.invoke({"context": context, "question": query})
  return prompt

6. Initialize model
   
model = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

8. Get the output

model.invoke(prompt)

## Query From HR

## 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

Answer: The technique used is Optical Character Recognition (OCR) with tesseract, which extracts text from the image representation of the PDF rather than relying on embedded text data. This is an advance method extracting text like Bengali language.

I spent so much time to extract text from the pdf. I tried some typical method of pdf reading but none of them was helpfull. They can't extract bengali text correctly from the pdf. 


## 2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

Answer: I have used character based chunking here. I applied sentence based chunking but the result didn't improve. But I think for this kind of project we should heading-based or paragraph-based chunking to keep semantic maening. For this we need to preprocess the text file in depth. Due to the time shortage I couldn't do it.


## 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

Answer: I have used Transformer model intfloat/multilingual-e5-base for embedding. It supports English and Bengali language. 


## 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

Answer: I have used FAISS (Facebook AI Similarity Search), that is a library developed by Facebook for efficient similarity search and clustering of dense vectors. FAISS index that uses L2 (Euclidean) distance for similarity. It's useful for smaller datasets or when exact matching is required.

## 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

Answer: Same embedding for chunks and query that ensure most relevant chunks retrieve. I have used rich embedding model that works perfect on English and Bengali language. 

Vague queries may match unrelated chunks and the language model may hallucinate answers based on weakly related information.

## 6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?

Answer: Not perfectly. The Pipeline has so much to improve. 

* Most of Data Preprocessing
* Heading-based chunking
* Fine-tuning LLM with a larger document.
* I could use Normalization in embedding step
* Trial and error for embedding model, simiilarity, other powerfull llm model.



It’s a brilliant initiative that reflects your commitment to innovation in education through AI, and I’m genuinely excited about the possibilities this role offers. I am very much interested in joining your team and contributing to projects that make learning more intelligent and accessible. I would be glad to further discuss my approach to the assessment and how my background aligns with your goals.

## Please feel free to reach out if you’d like to set up a time to discuss further.











