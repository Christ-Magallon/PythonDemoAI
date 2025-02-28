import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from main import fetch_data, preprocess_data, train_model, recommend_products

# Global Variables for Model & Data
df_pivot = None
model = None
df = None

# Background Task to Update every 5 minutes
async def update_model():
    """Background task to update the model every 5 minutes."""
    global df_pivot, model, df
    while True:
        print("🔄 Updating model with latest data...")
        df = fetch_data()  
        df_pivot = preprocess_data(df)
        model = train_model(df_pivot)  
        print("✅ Model updated successfully!")
        await asyncio.sleep(300)  

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event manager (Runs background task on startup)."""
    task = asyncio.create_task(update_model())  
    yield
    task.cancel()  

app = FastAPI(lifespan=lifespan)  

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# API Endpoint to Get Recommendations
@app.get("/recommend/{user_id}")
def recommend(user_id: str):
    if model is None or df_pivot is None:
        return {"error": "Model is still loading, please try again later."}
    
    recommended_products = recommend_products(user_id, model, df_pivot, df)
    return recommended_products