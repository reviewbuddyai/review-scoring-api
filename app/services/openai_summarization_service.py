import os
import openai
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
MAX_INPUT_TOKENS = 128000  # Maximum input tokens for GPT-4o-mini
MAX_OUTPUT_TOKENS = 4096   # Maximum output tokens for GPT-4o-mini
MODEL_NAME = "gpt-4o-mini"

def chunk_reviews_by_token_limit(reviews, max_tokens):
    """Chunks reviews to fit within the specified token limit."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for review in reviews:
        review_tokens = tokenizer.encode(review)
        review_token_count = len(review_tokens)

        if current_chunk_tokens + review_token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [review]
            current_chunk_tokens = review_token_count
        else:
            current_chunk.append(review)
            current_chunk_tokens += review_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"Chunked reviews into {len(chunks)} chunks")
    return chunks

def query_openai(client, prompt):
    """Queries the OpenAI GPT-4o-mini model with the provided prompt."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.7,
            top_p=0.95
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"Error querying OpenAI GPT-4o-mini: {e}")
        return "An error occurred while processing the request."

def summarize_chunks(client, chunks, prompt):
    """Summarizes each chunk of reviews."""
    summaries = []
    for idx, chunk in enumerate(chunks):
        try:
            summary = query_openai(client, prompt + chunk)
            summaries.append(summary)
            print(f"Summarized chunk {idx+1}/{len(chunks)}")
        except Exception as ex:
            print(f"Error summarizing chunk {idx+1}/{len(chunks)}: {ex}")
    return summaries

def get_google_place_summary(reviews):
    """Generates a summary for Google Place reviews."""
    reviews_text = [review['text'] for review in reviews]
    chunks = chunk_reviews_by_token_limit(reviews_text, MAX_INPUT_TOKENS)
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Summarize each chunk
    prompt = """Summarize the following reviews which came from multiple reviewers in the following format,
      (if there is nothing that fits the category based on the reviews then leave it empty):
      **Summary:**
       * **Positive Reviews:**
            - postive topic from reviews number 1
            - positive topic from reviews number 2

       * **Neutral Reviews:**
            - netrual topic 1

       * **Negative Reviews:**
            - negative topic, or constructive criticism 1

       * **Reccomended Dishes:**
            - most reccommended dish
            - slightly less reccomended dish
       """
    chunk_summaries = summarize_chunks(client, chunks, prompt)
    
    # If there's only one chunk, return its summary
    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        # Combine chunk summaries with '|' separator
        combined_summary_text = " | ".join(chunk_summaries)
        print("Combined all chunk summaries")
        
        # Summarize the combined summaries
        final_prompt = "Each summary separated by '|' is generated for about 500 reviews. Summarize the summaries into one summary and keep the same format: "
        final_summary = query_openai(client, final_prompt + combined_summary_text)
    
    print("Final summary completed")
    print(final_summary)
    return final_summary
