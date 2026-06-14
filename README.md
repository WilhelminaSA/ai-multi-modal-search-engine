# AI Multi-Modal Search Engine

A semantic image retrieval system built using CLIP ViT-L/14 embeddings and FAISS vector search.

This project supports both text-to-image and image-to-image retrieval by mapping text and images into a shared embedding space and performing similarity search using FAISS.

---

## Features

### Text → Image Search

Search images using natural language queries.

Example:

computer monitor

The system retrieves semantically relevant images instead of relying on filenames or keywords.

---

### Image → Image Search

Provide an input image and retrieve visually and semantically similar images from the indexed dataset.

Example:

Query Image → Similar Monitors, Laptops, Screens, Electronics

---

### CLIP ViT-L/14 Embeddings

Uses CLIP ViT-L/14 through Sentence Transformers to generate high-quality image and text embeddings.

---

### FAISS Vector Search

Uses Facebook AI Similarity Search (FAISS) for efficient nearest-neighbor retrieval.

---

### Incremental Index Updates

New images can be added without rebuilding the entire index.

The system:

* Detects newly added images
* Generates embeddings only for new files
* Updates the FAISS index
* Updates metadata automatically

---

### Embedding Extraction Utility

Embeddings stored inside the FAISS index can be extracted and saved for analysis and experimentation.

---

## Project Architecture

### Text → Image Search

Text Query

↓

CLIP Text Encoder

↓

Shared Embedding Space

↓

FAISS Search

↓

Relevant Images

---

### Image → Image Search

Query Image

↓

CLIP Image Encoder

↓

Shared Embedding Space

↓

FAISS Search

↓

Similar Images

---

## Project Structure

ai-multi-modal-search-engine/

├── src/

│   ├── build_image_index.py

│   ├── config.py

│   ├── download_model.py

│   ├── extract_embeddings.py

│   ├── image_search.py

│   ├── text_search.py

│   ├── update_index.py

│   └── utils_index_tracker.py

│

├── requirements.txt

├── README.md

└── .gitignore

---

## Installation

Clone the repository:

git clone <repository-url>

Install dependencies:

pip install -r requirements.txt

---

## Build the Initial Index

Run:

python src/build_image_index.py

This process:

* Scans the dataset
* Generates CLIP embeddings
* Creates metadata
* Builds a FAISS index

---

## Text → Image Search

Run:

python src/text_search.py

Example query:

computer monitor

---

## Image → Image Search

Run:

python src/image_search.py

Example:

Enter image path:

C:\images\monitor.jpg

---

## Update Existing Index

After adding new images to the dataset:

python src/update_index.py

Only newly added images will be processed and indexed.

---

## Technologies Used

* Python
* CLIP ViT-L/14
* Sentence Transformers
* FAISS
* NumPy
* Pillow
* Matplotlib
* tqdm

---

## Future Improvements

* Web Interface
* REST API Support
* Larger Datasets
* Category Filtering
* Hybrid Search
* Thumbnail Database
* Faster Retrieval Pipelines

---

## Author

Sneha Tripathi

AI / Machine Learning Project
