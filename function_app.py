import os
import logging
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Azure configurations
FORM_RECOGNIZER_ENDPOINT = os.getenv("FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY = os.getenv("FORM_RECOGNIZER_KEY")
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

document_analysis_client = DocumentAnalysisClient(endpoint=FORM_RECOGNIZER_ENDPOINT, credential=AzureKeyCredential(FORM_RECOGNIZER_KEY))
blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

def semantic_chunking(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def main(event: dict):
    logging.info("Event received: %s", event)

    # Get Blob URL from Event Grid event
    blob_url = event['data']['url']
    logging.info(f"Processing blob: {blob_url}")

    # Parse the Blob URL to get container and blob name
    blob_name = blob_url.split("/")[-1]
    container_name = blob_url.split("/")[-2]

    # Download the Blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob().readall()

    # Send to Azure Form Recognizer
    poller = document_analysis_client.begin_analyze_document("custom-document-model", document=blob_data)
    result = poller.result()

    # Extract and chunk content
    for page_num, page in enumerate(result.pages, start=1):
        page_text = " ".join([line.content for line in page.lines])
        chunks = semantic_chunking(page_text)
        logging.info(f"Page {page_num} has {len(chunks)} chunks")
        for chunk in chunks:
            logging.info(f"Chunk: {chunk}")

    logging.info("Processing complete.")