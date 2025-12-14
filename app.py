import os
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- CONFIGURATION ---
# Render will load this from the "Environment Variables" section in the dashboard
AI_SERVICE_TOKEN = os.environ.get("AI_SERVICE_TOKEN")
API_URL = "https://models.inference.ai.azure.com/chat/completions"
MODEL_NAME = "gpt-4o-mini"

app = FastAPI(
    title="AI Backend Service",
    description="FastAPI service running on Render.com"
)

# --- MODELS ---
class AnalyzeRequest(BaseModel):
    filename: str

# --- ENDPOINTS ---

@app.get("/")
def home():
    """Health check endpoint for Render."""
    return {"status": "active", "platform": "Render"}

@app.get("/check-limit")
def check_limit():
    """
    Checks the rate limit status of the configured AI Service Token.
    """
    if not AI_SERVICE_TOKEN:
        raise HTTPException(status_code=500, detail="AI_SERVICE_TOKEN environment variable is missing.")

    headers = {
        "Authorization": f"Bearer {AI_SERVICE_TOKEN}",
        "Content-Type": "application/json"
    }

    # Minimal payload to trigger headers without using many tokens
    payload = {
        "model": MODEL_NAME, 
        "messages": [{"role": "user", "content": "Ping."}],
        "temperature": 0.1,
        "max_tokens": 1
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        # Extract rate limit headers (standard Azure/OpenAI headers)
        remaining = response.headers.get('x-ratelimit-remaining-requests') or response.headers.get('x-ratelimit-remaining') or 'N/A'
        limit = response.headers.get('x-ratelimit-limit-requests') or response.headers.get('x-ratelimit-limit') or 'N/A'
        reset = response.headers.get('x-ratelimit-reset-requests') or response.headers.get('x-ratelimit-reset') or 'N/A'
        
        result = {
            "status_code": response.status_code,
            "rate_limit_info": {
                "remaining_requests": remaining,
                "limit_requests": limit,
                "reset_time": reset
            }
        }

        if response.status_code == 200:
            result["message"] = "Token is valid."
        elif response.status_code == 429:
            result["message"] = "CRITICAL: Rate limit exceeded (429)."
            try:
                result["error_details"] = response.json()
            except:
                result["error_details"] = response.text
        else:
            result["message"] = f"Request failed with status {response.status_code}"
            result["error_details"] = response.text

        return result

    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
def analyze_filename(request: AnalyzeRequest):
    """
    Main endpoint to analyze filenames.
    """
    if not AI_SERVICE_TOKEN:
        raise HTTPException(status_code=500, detail="AI_SERVICE_TOKEN environment variable is missing.")

    headers = {
        "Authorization": f"Bearer {AI_SERVICE_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Movie and TV metadata analyst. Return ONLY raw JSON in the format: {\"title\": \"...\", \"year\": \"...\", \"isSeries\": false/true}. Analyze the following filename and extract the data."
            },
            {
                "role": "user",
                "content": f"Analyze: \"{request.filename}\""
            }
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 429:
             error_data = response.json() if response.content else {"error": "Rate limit exceeded"}
             raise HTTPException(status_code=429, detail=error_data)
        
        response.raise_for_status()
        
        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content')
        
        if content:
            # Clean up markdown if present
            clean_content = content.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(clean_content)
            except json.JSONDecodeError:
                return {"error": "AI returned malformed JSON", "raw_content": clean_content}
        
        return {"error": "No content returned from AI"}

    except requests.exceptions.RequestException as e:
        print(f"External API Error: {e}")
        raise HTTPException(status_code=503, detail="External AI service unavailable")
    except Exception as e:
        print(f"Internal Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
