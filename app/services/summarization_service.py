from together import Together
from huggingface_hub import login
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def initialize_clients(huggingface_token, together_api_key):
    client = Together(api_key=together_api_key)
    login(token=huggingface_token)
    return client

def chunk_reviews_by_token_limit(reviews, max_tokens, tokenizer):
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for review in reviews:
        review_tokens = tokenizer.tokenize(review)
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

def query_together(client, payload):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": payload}],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>"],
        stream=False
    )
    return response.choices[0].message.content

def summarize_chunks(client, chunks, max_len):
    summaries = []
    for idx, chunk in enumerate(chunks):
        try:
            summary = query_together(client, chunk)
            summaries.append(summary)
            print(f"Summarized chunk {idx+1}/{len(chunks)}")
        except Exception as ex:
            print(f"Error summarizing chunk {idx+1}/{len(chunks)}: {ex}")
    return " ".join(summaries)

def get_google_place_summary(reviews):
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    together_api_key = os.getenv('TOGETHER_API_KEY')
    client = initialize_clients(huggingface_token, together_api_key)
    
    # Chunk reviews
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    max_tokens = 128000  # Adjust based on the model's token limit
    reviews_text = [review['text'] for review in reviews]
    chunks = chunk_reviews_by_token_limit(reviews_text, max_tokens, tokenizer)
    
    # Summarize each chunk
    chunk_summaries = []
    for idx, chunk in enumerate(chunks):
        summary = summarize_chunks(client, [chunk], max_len=150)  # Adjust max_len based on the model's limit
        chunk_summaries.append(summary)
        print(f"Summarized review chunk {idx+1}/{len(chunks)}")
    
    # Combine chunk summaries
    combined_summary_text = " ".join(chunk_summaries)
    print("Combined all chunk summaries")
    
    # Summarize the combined summaries if needed
    final_summary = summarize_chunks(client, [combined_summary_text], max_len=150)
    print("Final summary completed")
    print(final_summary)
    return final_summary
