# import os
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates

# app = FastAPI()

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Mount static directory inside app/static
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# # Setup Jinja2 templates inside app/templates
# templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# # Import your model logic here (adjust the import path accordingly)
# from app.model.model_utils import enhance_rfid_signal  # ensure path is correct

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/predict", response_class=HTMLResponse)
# async def predict(request: Request, signal: str = Form(...)):
#     result = enhance_rfid_signal(signal)
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "result": result,
#         "input_signal": signal
#     })
## second code
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from app.model.model_utils import enhance_rfid_signal
# import joblib
# import tensorflow as tf

# app = FastAPI()

# # Load your scaler and model - change these paths as needed
# scaler = joblib.load("C:\\Users\\pc\\Desktop\\rfid-enhancer-project\\rfid_ai_project\\app\\model\\scaler.save")
# import tensorflow as tf

# model = tf.keras.models.load_model(
#     "C:\\Users\\pc\\Desktop\\rfid-enhancer-project\\rfid_ai_project\\app\\model\\lstm_model.h5",
#     custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
# )
# seq_length = 10  # Your sequence length (must match training)

# @app.post("/predict")
# async def predict(request: Request):
#     try:
#         data = await request.json()
#         signal = data.get("signal")
#         if not signal:
#             return JSONResponse(status_code=400, content={"error": "No signal provided"})

#         result = enhance_rfid_signal(signal, scaler, model, seq_length)
#         return {"enhanced_signal": result}

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
## third code
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.model.model_utils import enhance_rfid_signal

app = FastAPI()

# Mount static and template directories
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# For Swagger /docs testing
class SignalInput(BaseModel):
    signal: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def get_prediction(request: Request, signal: str = Form(...)):
    try:
        result = enhance_rfid_signal(signal)
        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {e}"})

# Optional: Swagger UI testing
@app.post("/predict")
async def predict_signal(data: SignalInput):
    try:
        result = enhance_rfid_signal(data.signal)
        return {"enhanced_signal": result}
    except Exception as e:
        return {"error": str(e)}

