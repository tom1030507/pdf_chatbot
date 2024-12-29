import os
import logging
import base64
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
import fitz  # PyMuPDF
import requests
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
os.environ['PINECONE_API_KEY'] = 'YOUR_PINECONE_API_KEY'

# Image save directory
IMAGE_DIRECTORY = Path("./images")
IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Create OpenAI Embeddings
embeddings = OpenAIEmbeddings()

def extract_pdf_text_images_tables(pdf_path):
    logger.info("Extracting text, images, and tables from PDF...")

    # Extract text
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = " ".join([doc.page_content for doc in documents])
    
    # Extract images and tables
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    table_texts = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            pix = fitz.Pixmap(pdf_document, xref)
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_filename = IMAGE_DIRECTORY / f"page_{page_num + 1}_image_{img_index + 1}.png"
            pix.save(str(image_filename))
            image_paths.append(str(image_filename))
        
        # Extract tables (assuming tables are detected as text blocks)
        blocks = page.get_text("blocks")
        for block in blocks:
            if "Table" in block[4]:  # Simple heuristic to detect tables
                table_texts.append(block[4])
    
    pdf_document.close()
    logger.info(f"Extracted {len(image_paths)} images and {len(table_texts)} tables from PDF.")
    return text, image_paths, table_texts

def image_to_base64(local_path):
    with open(local_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def chat_with_gpt(content, content_type="image"):
    api_key = os.environ['OPENAI_API_KEY']
    model = "gpt-4o"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if content_type == "image":
        base64_image = image_to_base64(content)
        content_data = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    else:
        content_data = {"type": "text", "text": content}
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [content_data]
            },
            {
                "role": "system",
                "content": "Describe this content in detail"
            } 
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

def generate_descriptions(image_paths, table_texts):
    logger.info("Generating descriptions for images and tables...")
    descriptions = []

    for idx, image_path in enumerate(image_paths):
        try:
            response = chat_with_gpt(image_path, content_type="image")
            description = f"image description {idx + 1}: {response if isinstance(response, str) else str(response)}"
            descriptions.append((description, f"image{idx + 1}"))
            logger.debug(f"Generated description for {image_path}: {response}")
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            descriptions.append((f"Error generating description for {image_path}", f"image{idx + 1}"))
    
    for idx, table_text in enumerate(table_texts):
        try:
            response = chat_with_gpt(table_text, content_type="text")
            description = f"table description {idx + 1}: {response if isinstance(response, str) else str(response)}"
            descriptions.append((description, f"table_{idx + 1}"))
            logger.debug(f"Generated description for table {idx + 1}: {response}")
        except Exception as e:
            logger.error(f"Error processing table {idx + 1}: {e}")
            descriptions.append((f"Error generating description for table {idx + 1}", f"table_{idx + 1}"))
    
    logger.info(f"Generated descriptions for {len(descriptions)} items.")
    return descriptions

def save_to_pinecone(descriptions, index_name):
    logger.info("Saving data to Pinecone...")

    if 'PINECONE_API_KEY' not in os.environ:
        logger.error("Pinecone API Key not set.")
        return

    docs = [Document(page_content=desc[0], metadata={"source": desc[1]}) for desc in descriptions]
    
    try:
        vectorstore = PineconeVectorStore.from_documents(
            docs,
            index_name=index_name,
            embedding=embeddings
        )
        logger.info("Data saved to Pinecone successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error saving data to Pinecone: {e}")
        return None

def create_qa_model(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

if __name__ == "__main__":
    pdf_path = "./1706.03762v7.pdf"
    index_name = "test"
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    try:
        pdf_text, image_paths, table_texts = extract_pdf_text_images_tables(pdf_path)
        descriptions = generate_descriptions(image_paths, table_texts)
        vectorstore = save_to_pinecone(descriptions, index_name)
        qa_model = create_qa_model(vectorstore, llm)
        
        while True:
            question = input("Enter a question: ")
            answer = qa_model.invoke(question)
            print(f"Answer: {answer}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")