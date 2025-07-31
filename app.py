# app.py - Live E-commerce Review Analyzer

# FINAL Scraper Function for FLIPKART (v6)

import streamlit as st
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# NOTE: No need for regex or ASIN extraction for this simpler approach
@st.cache_data(show_spinner=False)
def scrape_flipkart_reviews(url, max_pages=5):
    """
    Scrapes product reviews from a Flipkart product URL.
    This is more reliable for an educational project.
    """
    st.info(f"Initializing Flipkart scraper...")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, 10)
    
    reviews = []
    
    try:
        driver.get(url)
        
        # On Flipkart, the reviews are on the main page, but we need to find the "All reviews" button/link
        all_reviews_link_element = wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[text()='All reviews']/parent::a"))
        )
        reviews_url = all_reviews_link_element.get_attribute('href')
        
        st.write("Found dedicated reviews page. Navigating...")
        driver.get(reviews_url)

        # Loop through the pages of reviews
        for page in range(max_pages):
            st.write(f"Scraping review page {page + 1}...")
            
            # This selector is for the review text on Flipkart
            review_elements = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.ZmyHeo"))
            )
            
            for review in review_elements:
                if review.text.strip():
                    reviews.append(review.text.strip())
            
            # Find the "Next" button and click it
            try:
                # Find the "Next" button by looking for the one that is NOT disabled
                next_button = driver.find_element(By.XPATH, "//a/span[text()='Next']/parent::a")
                driver.execute_script("arguments[0].click();", next_button) # Use JS click for reliability
                time.sleep(2)
            except NoSuchElementException:
                st.warning(f"Reached the last page of reviews at page {page+1}.")
                break
            
    except TimeoutException:
        st.error("Could not find the reviews on the Flipkart page.")
        st.warning("Please ensure you are using a valid Flipkart product URL with reviews.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    
    finally:
        driver.quit()
    
    unique_reviews = list(set(reviews))
    
    if unique_reviews:
        st.success(f"Scraping complete! Found {len(unique_reviews)} unique reviews.")
    else:
        st.error("Found 0 reviews.")
        
    return unique_reviews
    """
    Scrapes Amazon.in product reviews by directly constructing the reviews URL from the product ASIN.
    This is the most stable method.
    """
    st.info(f"Initializing scraper...")

    # --- NEW: Extract Product ASIN from URL ---
    asin_match = re.search(r'/(dp|gp/product)/(\w{10})', url)
    if not asin_match:
        st.error("Could not find a valid Product ID (ASIN) in the URL. Please use a standard Amazon product URL.")
        return []
    
    asin = asin_match.group(2)
    reviews_url = f"https://www.amazon.in/product-reviews/{asin}/?reviewerType=all_reviews&pageNumber=1"
    st.write(f"Found Product ASIN: {asin}. Navigating directly to reviews page...")

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, 10)
    
    reviews = []
    
    try:
        driver.get(reviews_url)
        
        # Loop through the pages of reviews
        for page in range(max_pages):
            st.write(f"Scraping review page {page + 1}...")
            
            # This selector is the correct one for the dedicated reviews page
            review_elements = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span[data-hook='review-body']"))
            )
            
            for review in review_elements:
                if review.text.strip():
                    reviews.append(review.text.strip())
            
            # Try to find and click the "Next page" button
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, 'ul.a-pagination li.a-last a')
                next_button.click()
                time.sleep(2) # Wait for the next page to load
            except NoSuchElementException:
                st.warning(f"Reached the last page of reviews at page {page+1}.")
                break
            
    except TimeoutException:
        st.error("Could not load the review elements on the page.")
        st.warning("This can happen if the product has zero reviews or if Amazon changed its page structure.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    
    finally:
        driver.quit()
    
    # Remove duplicates by converting to a set and back to a list
    unique_reviews = list(set(reviews))
    
    if unique_reviews:
        st.success(f"Scraping complete! Found {len(unique_reviews)} unique reviews.")
    else:
        st.error("Found 0 reviews.")
        
    return unique_reviews
    
    # Remove duplicates by converting to a set and back to a list
    unique_reviews = list(set(reviews))
    
    if unique_reviews:
        st.success(f"Scraping complete! Found {len(unique_reviews)} unique reviews.")
    else:
        st.error("Found 0 reviews. The scraper could not find any review content.")
        
    return unique_reviews


# --- AI Summary ---
def get_ai_summary(reviews_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    prompt = f"""
    You are a product review analyst. Based on the following customer reviews,
    generate a bulleted list of the top 5 pros and a bulleted list of the top 5 cons.

    Reviews:
    ---
    {reviews_text}
    ---
    """
    response = model.generate_content(prompt)
    return response.text


# --- RAG Setup ---
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
    prompt = f"""
    Based ONLY on the following context from customer reviews, answer the user's question.
    If the context doesn't contain the answer, say "I couldn't find information about that in the reviews."

    Context:
    ---
    {context}
    ---

    Question: {question}
    """
    response = llm.generate_content(prompt)
    return response.text


# --- Streamlit UI ---
st.title("🚀 Live E-commerce Review Analyzer")
st.write("Paste an Amazon product URL to scrape its reviews, get an AI summary, and ask questions.")

# Session state
if 'reviews' not in st.session_state:
    st.session_state.reviews = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'rag_ready' not in st.session_state:
    st.session_state.rag_ready = False

# S# --- Hardcoded API Key (For Development) ---
# WARNING: Do not share this code publicly with your key in it.
# Replace "YOUR_API_KEY_HERE" with your actual Gemini API key.
api_key = "AIzaSyBzFL0D0AF9reimU1KbyRKW3axzdhjUD6A"

# You can comment out or remove the sidebar if you no longer need it
# st.sidebar.header("Configuration")

# Main inputs
url = st.text_input("Enter the Amazon Product URL", "")

if st.button("Analyze Product"):
    if not api_key:
        st.error("Please enter your Google Gemini API Key in the sidebar.")
    elif not url:
        st.error("Please enter a product URL.")
    else:
        st.session_state.reviews = scrape_amazon_reviews(url)

        if st.session_state.reviews:
            with st.spinner("AI is summarizing..."):
                summary_text = " ".join(st.session_state.reviews)
                st.session_state.summary = get_ai_summary(summary_text, api_key)

            with st.spinner("Building AI chatbot..."):
                chunks, index, model_id = setup_rag_system(st.session_state.reviews, api_key)
                st.session_state.rag_chunks = chunks
                st.session_state.rag_index = index
                st.session_state.rag_model = model_id
                st.session_state.rag_ready = True

# Display summary
if st.session_state.summary:
    st.subheader("AI-Generated Summary")
    st.markdown(st.session_state.summary)
    st.markdown("---")

# RAG Chatbot
if st.session_state.rag_ready:
    st.subheader("💬 Chat with the Reviews")
    user_question = st.text_input("Ask a question about the product (e.g., 'What did people say about the battery?')")

    if user_question:
        with st.spinner("Finding an answer..."):
            answer = get_rag_answer(
                user_question,
                st.session_state.rag_index,
                st.session_state.rag_chunks,
                st.session_state.rag_model,
                api_key
            )
            st.markdown(answer)
