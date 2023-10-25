from fastapi import FastAPI,UploadFile, File
import torch
from PIL import Image
import io

app = FastAPI()

gen_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/general_model.pt', force_reload=True)
gen_model.eval()
custom_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/custom_model.pt', force_reload=True)
custom_model.eval()
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    gen_results = gen_model(image)
    gen_xyxy = gen_results.pandas().xyxy[0].to_dict(orient="records")
    gen_xywhn = gen_results.pandas().xywhn[0].to_dict(orient="records")

    cus_results = custom_model(image)
    cus_xyxy = cus_results.pandas().xyxy[0].to_dict(orient="records")
    cus_xywhn = cus_results.pandas().xywhn[0].to_dict(orient="records")
    return {"gen_xyxy" : gen_xyxy, "gen_xywhn" : gen_xywhn, "cus_xyxy" : cus_xyxy, "cus_xywhn" : cus_xywhn}
