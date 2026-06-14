import os

PROJECT_ROOT = r"C:\my_work\AI_MultiModal_Search_Engine"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
DATABASE_ROOT = os.path.join(PROJECT_ROOT, "database")

FAISS_PATH = os.path.join(DATABASE_ROOT, "image_index.faiss")
METADATA_PATH = os.path.join(DATABASE_ROOT, "metadata.json")
EMBEDDINGS_PATH = os.path.join(DATABASE_ROOT, "embeddings.npy")

MODEL_PATH = r"C:\my_work\models\clip-ViT-L-14-ST"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
BATCH_SIZE = 32

os.makedirs(DATABASE_ROOT, exist_ok=True)