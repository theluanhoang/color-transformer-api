from fastapi import Body, FastAPI, File, UploadFile
import cv2
import numpy as np
import shutil
import os
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def apply_color_transfer(source_path, target_path, output_path):
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    if source is None or target is None:
        raise ValueError("Không thể đọc một trong hai ảnh")

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    mean_src, std_src = cv2.meanStdDev(source_lab)
    mean_tar, std_tar = cv2.meanStdDev(target_lab)

    adjusted_lab = ((source_lab - mean_src.reshape(1, 1, 3)) * (std_tar.reshape(1, 1, 3) / (std_src.reshape(1, 1, 3) + 1e-6))) + mean_tar.reshape(1, 1, 3)
    adjusted_lab = np.clip(adjusted_lab, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, result)

@app.get("/")
async def hello():
    return {"message": "Hello"}

@app.post("/")
async def test(
    name: str       
):
    return {"message": name}


@app.post("/upload/")
async def upload_images(
    source_file: UploadFile = File(...), target_file: UploadFile = File(...)
):
    print(f"Received source file: {source_file.filename}")
    print(f"Received target file: {target_file.filename}")
    source_path = os.path.join(UPLOAD_DIR, "source.jpg")
    target_path = os.path.join(UPLOAD_DIR, "target.jpg")
    output_path = os.path.join(UPLOAD_DIR, "output.jpg")

    with open(source_path, "wb") as buffer:
        shutil.copyfileobj(source_file.file, buffer)
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(target_file.file, buffer)

    try:
        apply_color_transfer(source_path, target_path, output_path)
    except ValueError as e:
        return {"error": str(e)}

    return FileResponse(output_path, media_type="image/jpeg", filename="output.jpg")

# Thêm API POST để kiểm tra
@app.post("/test/")
async def test_api(data: dict = Body(...)):
    return JSONResponse(content={"message": "Dữ liệu nhận được", "data": data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
