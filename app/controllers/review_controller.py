from fastapi import APIRouter
from app.services import openai_summarization_service, review_service

router = APIRouter()

@router.get("/predict_score_for_text")
async def predict_score_for_text(review: str):
    score = review_service.predict_score(review)
    return {"score": score}

@router.get("/get_review_data")
async def get_review_data(place_name: str, number_of_reviews: int = 10):
    reviews, place_id, place_name = review_service.get_google_place_reviews(place_name=place_name, number_of_reviews=number_of_reviews)
    if not reviews: return { "score": 0, "summary": f"Reviews not found for place: {place_name}"}
    score = review_service.get_google_place_score(reviews=reviews)
    summary = openai_summarization_service.get_google_place_summary(reviews)
    return {
            "place_id": place_id,
            "place_name": place_name,
            "score": review_service.round_to_one_decimal_place(value=score),
            "summary": summary,
            "best_review": {"score": 5, "review": "twas a funtime"},
            "worst_review": {"score": 0, "review": "dafuq was this"} 
            }
