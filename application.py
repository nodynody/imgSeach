from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import chromadb
from chromadb.config import Settings
import torch
import os
from glob import glob
from tqdm import tqdm

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요한 도메인만 설정 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 경로 설정
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# ViT 모델 및 Feature Extractor 로드
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16').to("cuda")

# ChromaDB 초기화
client = chromadb.Client(Settings(
    persist_directory="database",
    anonymized_telemetry=False
))
collection = client.get_or_create_collection("images")

# 이미지 특징 벡터 생성 함수
def extract_embedding(image: Image.Image):
    img_tensor = feature_extractor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**img_tensor)
    embedding = outputs.pooler_output.detach().cpu().numpy().squeeze()
    return embedding

# 이미지 등록 함수
def register_images(image_dir: str):
    # img_list = sorted(glob(os.path.join(image_dir, "*.jpg")))
    
    valid_extensions = (".jpg", ".jpeg", ".png")
    img_list = sorted(
    [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    )

    if not img_list:
        print("No images found in the specified directory.")
        return

    embeddings = []
    metadatas = []
    ids = []

    for i, img_path in enumerate(tqdm(img_list)):
        try:
            img = Image.open(img_path).convert("RGB")
            cls = os.path.basename(img_path).split("_")[0]

            embedding = extract_embedding(img).tolist()

            embeddings.append(embedding)
            metadatas.append({"uri": img_path, "name": cls})
            ids.append(str(i))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if embeddings:
        collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
        print(f"{len(img_list)} images registered successfully.")
    else:
        print("No embeddings to add.")

# 이미지 폴더 경로 설정
image_dir = "data/images"
register_images(image_dir)

# 등록 API
@app.post("/register/")
async def register_images(image_dir: str):
    img_list = sorted(glob(os.path.join(image_dir, "*.jpg")))

    embeddings = []
    metadatas = []
    ids = []

    for i, img_path in enumerate(tqdm(img_list)):
        img = Image.open(img_path).convert("RGB")
        cls = os.path.basename(img_path).split("_")[0]

        embedding = extract_embedding(img).tolist()

        embeddings.append(embedding)
        metadatas.append({"uri": img_path, "name": cls})
        ids.append(str(i))

    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
    return {"message": f"{len(img_list)} images registered successfully."}

# 검색 API
@app.post("/query/")
async def search_image(file: UploadFile = File(...), n_results: int = 1):
    print("Received file:", file.filename)  # 로그 추가
    try:
        image = Image.open(file.file).convert("RGB")
        query_embedding = extract_embedding(image).tolist()

        
        # ChromaDB에서 유사 이미지 검색
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        query_results = [
            {
                "uri": meta["uri"],
                "name": os.path.splitext(os.path.basename(meta["uri"]))[0],  # 파일 이름 추출
                "distance": dist
            }
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]
        print(query_results)
        return {"query_results": query_results}
    except Exception as e:
        return {"error": str(e)}

# @app.get("/")
# async def main():
#     content = """
#     <form action="/upload/" enctype="multipart/form-data" method="post">
#     <input name="file" type="file">
#     <input type="submit">
#     </form>
#     <form action="/query/" enctype="multipart/form-data" method="post"> 
#     <input name="file" type="file"> 
#     <input type="submit"> 
#     </form>
#     """
#     return HTMLResponse(content=content)

@app.get("/")
def apps():
        return FileResponse('static/index.html'); 