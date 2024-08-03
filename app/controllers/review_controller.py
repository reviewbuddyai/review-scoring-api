from fastapi import APIRouter
from app.services import review_service, summarization_service

router = APIRouter()

@router.get("/predict_score_for_text")
async def predict_score_for_text(review: str):
    score = review_service.predict_score(review)
    return {"score": score}

@router.get("/predict_google_place")
async def predict_google_place(place_name: str, number_of_reviews: int = 10):
    reviews, place_id, place_name = review_service.get_google_place_reviews(place_name=place_name, number_of_reviews=number_of_reviews)
    if not reviews: return None
    score = review_service.get_google_place_score(reviews=reviews)
    return {"place_id": place_id, "place_name": place_name, "score": review_service.round_to_half_step(value=score)}

@router.get("/summary_for_place")
async def summary_for_place(place_name: str, number_of_reviews: int = 10):
    reviews, place_id, place_name = review_service.get_google_place_reviews(place_name=place_name, number_of_reviews=number_of_reviews)
    if not reviews: return None
    summary = summarization_service.get_google_place_summary(reviews)
    return {"summary": summary}
