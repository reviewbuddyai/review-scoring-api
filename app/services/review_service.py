import datetime
import math
import torch
import requests
import re
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_url = "https://google-maps-scraper-api-bnwzz3dieq-zf.a.run.app/place"

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model.config.problem_type = "regression"

model.load_state_dict(torch.load('reviewbody_scoring_v1.pth', map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def round_to_one_decimal_place(value):
    return math.floor(value * 10) / 10


def preprocess_reviews(data):
    english_and_numbers_pattern = re.compile(r'[^a-zA-Z0-9\s]')
    english_word_pattern = re.compile(r'[a-zA-Z]+')
    
    for review in data.get('reviews', []):
        cleaned_text = english_and_numbers_pattern.sub('', review.get('text', ''))
        review['text'] = cleaned_text

    filtered_reviews = []
    for review in data.get('reviews', []):
        cleaned_text = english_and_numbers_pattern.sub('', review.get('text', ''))
        if english_word_pattern.search(cleaned_text):
            review['text'] = cleaned_text
            filtered_reviews.append(review)
    
    return filtered_reviews

def predict_score(review: str):
    inputs = tokenizer(review, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits.item() * 4 + 1

def get_google_place_reviews(place_name: str, number_of_reviews: int = 10):
    url_for_reviews = f"{base_url}/{place_name}/reviews"
    try:
        response = requests.get(url_for_reviews, params={"max_reviews": number_of_reviews})
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        return preprocess_reviews(data=data), data['id'], data['name']
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while accessing reviews for places api: {e}")
        return None, None, None
    
def get_google_place_score(reviews):
    total_score = 0
    total_weight = 0
    
    for review in reviews:
        score = predict_score(review['text'])
        likes_weight = (review['likes'] + 1) ** 0.5
        reviews_by_reviewer_weight = (review['reviews_by_reviewer'] + 1) ** 0.25
        date_formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
        ]
        
        datetime_obj = None
        
        for date_format in date_formats:
            try:
                datetime_obj = datetime.datetime.strptime(review['publish_date'], date_format)
                break
            except Exception as e:
                pass
        
        if not datetime_obj:
             publish_date_weight = 1
        else:
            days_ago = (datetime.datetime.now() - datetime_obj).days
            publish_date_weight = 4 * (1/(0.01*(days_ago**1.5)+1)) + 1
                    
        weight = likes_weight * reviews_by_reviewer_weight * publish_date_weight
        total_score += score * weight
        total_weight += weight

    return total_score / total_weight
