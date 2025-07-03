import pandas as pd
import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import io
import uuid
from typing import Dict
from cachetools import TTLCache
from huggingface_hub import InferenceClient
import os
from datetime import datetime
import logging
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app and cache
app = FastAPI()
cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize HuggingFace Inference API client
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Set this in your environment or directly
if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN environment variable not set")
llm_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=HF_API_TOKEN)

# Pydantic model for query input
class QueryRequest(BaseModel):
    session_id: str
    prompt: str

# Validate and preprocess uploaded data
def validate_and_process_file(file_content: bytes, file_type: str) -> pd.DataFrame:
    try:
        if file_type == "csv":
            df = pd.read_csv(io.BytesIO(file_content))
        elif file_type == "excel":
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            raise ValueError("Unsupported file format")

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Find required columns by pattern
        date_col = next((col for col in df.columns if 'date' in col), None)
        amount_col = next((col for col in df.columns if 'amount' in col), None)
        # Look for a column likely to contain categories like "Groceries", "Rent"
        possible_category_cols = [col for col in df.columns if 'categor' in col.lower()]
        if not possible_category_cols:
            possible_category_cols = [col for col in df.columns if col.lower() not in ['date', 'amount', 'type', 'description']]
        
        category_col = possible_category_cols[0] if possible_category_cols else None



        # Validate we found required columns
        if not date_col or not amount_col or not category_col:
            missing = []
            if not date_col: missing.append("date")
            if not amount_col: missing.append("amount")
            if not category_col: missing.append("category/type")
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        # Preprocess data
        df = df.rename(columns={
            date_col: 'date',
            amount_col: 'amount',
            category_col: 'category'
        })
        
        # Convert date with month/day/year format
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
        
        # Handle amount conversion (remove currency symbols, commas)
        df['amount'] = (
            df['amount']
            .astype(str)
            .str.replace(r'[^\d\.-]', '', regex=True)
            .astype(float)
        )
        
        df['category'] = df['category'].astype(str).str.strip()

        # Drop rows with missing essential data
        df = df.dropna(subset=['date', 'amount'])
        
        return df
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")
# Convert DataFrame to text summary for RAG
def df_to_text_summary(df: pd.DataFrame) -> str:
    summary = []
    # Basic summary statistics
    expense_df = df[df.get('type', '').str.lower() == 'expense']
    total_expenses = expense_df['amount'].sum()
    category_sums = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
    expense_df['month'] = expense_df['date'].dt.to_period('M')
    monthly_sums = expense_df.groupby('month')['amount'].sum()


    summary.append(f"Total expenses: Rs.{total_expenses:,.2f}")

    # Category-wise summary
    summary.append("\nCategory-wise expenses:")
    for category, amount in category_sums.items():
        summary.append(f"- {category}: ${amount:,.2f}")

    # Monthly trends
    df['month'] = df['date'].dt.to_period('M')
    summary.append("\nMonthly expenses:")
    for month, amount in monthly_sums.items():
        summary.append(f"- {month}: ${amount:,.2f}")

    return "\n".join(summary)

# Generate LLM response using HuggingFace Inference API
def query_llm(summary: str, prompt: str) -> str:
    try:
        system_prompt = "You are a financial assistant. Use the data summary provided to answer user questions accurately."
        user_prompt = f"""Here is the summary of financial data:\n\n{summary}\n\nNow, answer this question:\n{prompt}"""

        response = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logger.error(f"LLM query error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")

# API Endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file extension
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["csv", "xlsx", "xls"]:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Read and process file
        content = await file.read()
        df = validate_and_process_file(content, "csv" if file_extension == "csv" else "excel")

        # Generate session ID and cache data
        session_id = str(uuid.uuid4())
        cache[session_id] = df

        return {"session_id": session_id, "message": "File uploaded successfully"}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
async def query_data(request: QueryRequest):
    logger.info(f"Prompt: {request.prompt}")
    

    try:
        # Retrieve data from cache
        if request.session_id not in cache:
            raise HTTPException(status_code=404, detail="Session not found")

        df = cache[request.session_id]
        summary = df_to_text_summary(df)
        logger.info(f"Summary:\n{summary}")
        answer = query_llm(summary, request.prompt)

        # Format response
        response = {
            "session_id": request.session_id,
            "prompt": request.prompt,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }

        return response
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Example prompt templates
PROMPT_TEMPLATES = {
    "summary": "Summarize total expenses for each category.",
    "top_categories": "Identify the top 3 spending categories in the last 3 months.",
    "highest_month": "Which month had the highest spending?"
}

@app.get("/prompt_templates")
async def get_prompt_templates():
    return {"prompt_templates": PROMPT_TEMPLATES}