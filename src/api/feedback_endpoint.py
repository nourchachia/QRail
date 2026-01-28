"""
API endpoint for handling expert feedback on anomaly detections.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
import json
from pathlib import Path
from datetime import datetime

router = APIRouter()

class FeedbackRequest(BaseModel):
    """Request model for anomaly feedback"""
    incident_text: str
    action: Literal['valid', 'invalid', 'dismiss']
    timestamp: str

class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    status: str
    message: str
    action_taken: str

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_anomaly_feedback(request: FeedbackRequest):
    """
    Record operator feedback on anomaly detection.
    
    Actions:
    - valid: Incident is real and unprecedented, operator will create solution
    - invalid: Incident is a data entry error or nonsense input
    - dismiss: Test/training data, no action needed
    """
    try:
        # Store feedback in JSON file
        feedback_file = Path("data/anomaly_feedback.json")
        
        # Load existing feedback
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        else:
            feedback_data = {"feedback": []}
        
        # Add new feedback
        feedback_entry = {
            "incident_text": request.incident_text,
            "action": request.action,
            "timestamp": request.timestamp,
            "processed_at": datetime.now().isoformat()
        }
        
        feedback_data["feedback"].append(feedback_entry)
        
        # Save
        feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        
        # Determine response
        if request.action == "valid":
            message = "Feedback recorded. Please create a solution for this incident."
            action_taken = "logged_for_training"
        elif request.action == "invalid":
            message = "Marked as invalid input. This pattern will be filtered in future."
            action_taken = "added_to_blocklist"
        else:  # dismiss
            message = "Dismissed. No action taken."
            action_taken = "ignored"
        
        return FeedbackResponse(
            status="success",
            message=message,
            action_taken=action_taken
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")
