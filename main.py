from fastapi import FastAPI,UploadFile, File
import torch, uvicorn, io
from PIL import Image

app = FastAPI()

gen_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/general_model.pt', force_reload=True)
gen_model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    results = gen_model(image)
    xyxy = results.pandas().xyxy[0].to_dict(orient="records")
    xywhn = results.pandas().xywhn[0].to_dict(orient="records")

    return {"xyxy" : xyxy, "xywhn" : xywhn}

if __name__ == '__main__':
    
    uvicorn.run("main:app", host="127.0.0.1", port=8080,
                reload=True, debug=True)
