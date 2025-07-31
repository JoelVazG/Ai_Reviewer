# app.py - Final, Cleaned, and Working Version

import streamlit as st
import time
import re
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import google.generativeai as genai
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Live Product Analyzer",
    page_icon="🚀",
    layout="wide"
)

@st.cache_data(show_spinner=False)
def scrape_flipkart_reviews(url, max_pages=5):
    """
    Scrapes reviews from a Flipkart URL by directly constructing the reviews page URL.
    This is the most stable and definitive method.
    """
    st.info(f"Initializing Flipkart scraper...")

    # --- The Definitive Fix: Construct the reviews URL directly ---
    if "/p/" not in url:
        st.error("This does not appear to be a valid Flipkart product page URL.")
        return []

    reviews_url = url.replace("/p/", "/product-reviews/")
    st.write(f"Constructed reviews URL. Navigating directly...")

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, 10)
    reviews = []

    try:
        driver.get(reviews_url)

        # Loop through the pages of reviews
        for page in range(max_pages):
            st.write(f"Scraping review page {page + 1}...")
            
            # This selector is for the main review text content on Flipkart
            review_elements = wait.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "div.ZmyHeo"))
            )

            for review in review_elements:
                if review.text.strip():
                    reviews.append(review.text.strip())

            # Find and click the "Next" button
            try:
                next_button = driver.find_element(
                    By.XPATH, "//a/span[text()='Next']")
                # A more robust way to click, especially in headless mode
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(2)
            except NoSuchElementException:
                st.warning(
                    f"Reached the last page of reviews at page {page+1}.")
                break

    except TimeoutException:
        st.error(
            "Could not load reviews. The product may have no reviews or the page structure has changed.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        driver.quit()

    unique_reviews = list(set(reviews))
    if unique_reviews:
        st.success(
            f"Scraping complete! Found {len(unique_reviews)} unique reviews.")
    else:
        st.error("Found 0 reviews.")

    return unique_reviews

# --- AI and RAG Functions ---
def get_ai_summary(reviews_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""You are a product review analyst. Based on the following customer reviews, generate a bulleted list of the top 5 pros and a bulleted list of the top 5 cons. Reviews: --- {reviews_text} ---"""
    response = model.generate_content(prompt)
    return response.text

def setup_rag_system(reviews, api_key):
    genai.configure(api_key=api_key)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(" ".join(reviews))
    model_id = 'models/embedding-001'
    embeddings = genai.embed_content(model=model_id, content=chunks, task_type="RETRIEVAL_DOCUMENT")['embedding']
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return chunks, index, model_id

def get_rag_answer(question, index, chunks, model_id, api_key):
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    question_embedding = genai.embed_content(model=model_id, content=question, task_type="RETRIEVAL_QUERY")['embedding']
    distances, indices = index.search(np.array(question_embedding).reshape(1, -1), 5)
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n---\n".join(relevant_chunks)
    prompt = f"""Based ONLY on the following context from customer reviews, answer the user's question. If the context doesn't contain the answer, say "I couldn't find information about that in the reviews." Context: --- {context} --- Question: {question}"""
    response = llm.generate_content(prompt)
    return response.text

# --- Streamlit UI ---
st.title("🚀 Live E-commerce Review Analyzer")
st.write("Paste a Flipkart product URL to scrape its reviews, get an AI summary, and ask questions.")

# Session state initialization
if 'reviews' not in st.session_state:
    st.session_state.reviews = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'rag_ready' not in st.session_state:
    st.session_state.rag_ready = False

# --- Hardcoded API Key (For Development) ---
# 🚨 WARNING: Revoke and remove this key before sharing your code!
api_key = "AIzaSyBzFL0D0AF9reimU1KbyRKW3axzdhjUD6A" 

# --- Main App Logic ---
url = st.text_input("Enter the Flipkart Product URL", "")

if st.button("Analyze Product"):
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        st.error("Please replace 'YOUR_API_KEY_HERE' in the code with your actual Gemini API key.")
    elif not url:
        st.error("Please enter a product URL.")
    else:
        # **THIS IS THE FIX: Calling the correct scraper function**
        st.session_state.reviews = scrape_flipkart_reviews(url, max_pages=30)

        if st.session_state.reviews:
            st.metric(label="Total Unique Reviews Analyzed", value=len(st.session_state.reviews))
            with st.spinner("AI is summarizing..."):
                summary_text = " ".join(st.session_state.reviews)
                st.session_state.summary = get_ai_summary(summary_text, api_key)
            with st.spinner("Building AI chatbot..."):
                chunks, index, model_id = setup_rag_system(st.session_state.reviews, api_key)
                st.session_state.rag_chunks = chunks
                st.session_state.rag_index = index
                st.session_state.rag_model = model_id
                st.session_state.rag_ready = True
        else:
             st.session_state.summary = ""
             st.session_state.rag_ready = False

# --- Display Results ---
if st.session_state.summary:
    st.subheader("AI-Generated Summary")
    st.markdown(st.session_state.summary)
    st.markdown("---")

# --- RAG Chatbot ---
if st.session_state.rag_ready:
    st.subheader("💬 Chat with the Reviews")
    user_question = st.text_input("Ask a question about the product (e.g., 'What did people say about the camera?')")

    if user_question:
        with st.spinner("Finding an answer..."):
            answer = get_rag_answer(user_question, st.session_state.rag_index, st.session_state.rag_chunks, st.session_state.rag_model, api_key)
            st.markdown(answer)