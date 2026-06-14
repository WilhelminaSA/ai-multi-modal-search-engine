from sentence_transformers import SentenceTransformer

MODEL_PATH = r"C:\my_work\models\clip-ViT-L-14-ST"

print("\n================================================")
print("LOADING CLIP MODEL (ViT-L/14)")
print("================================================")

model = SentenceTransformer(MODEL_PATH)

print("Model Loaded Successfully")
print("Model Path:", MODEL_PATH)