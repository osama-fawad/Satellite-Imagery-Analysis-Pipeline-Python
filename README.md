# 🌍 Satellite Imaging Project

## 📌 Overview
This project leverages **Sentinel-2 satellite imagery** and **machine learning** to detect vegetation, water bodies, and urban structures. The goal is to extract actionable insights for **environmental monitoring, agriculture, and urban planning** using geospatial tools, image-processing algorithms, and deep learning techniques.

---

## ✨ Features
✅ **Satellite Data Processing** (Sentinel-2 L2A with 13 spectral bands)  
✅ **Preprocessing Techniques** (Contrast Stretching, Histogram Equalization, Gamma Correction)  
✅ **Vegetation & Water Detection** using **NDVI** and **NDWI**  
✅ **Building Detection** with **YOLOv8 Object Detection**  
✅ **Interactive Map** using **Leaflet.js** for user-defined bounding boxes  
✅ **FastAPI Deployment** for real-time image processing  

---

## 📡 Data Acquisition

The project retrieves **Sentinel-2 L2A imagery** using the **Copernicus Sentinel Hub API**.

### 🛰 **Satellite Bands Used**
| Band  | Wavelength (nm) | Resolution | Use Case |
|--------|----------------|-------------|------------------------------------------------|
| **B2 (Blue)** | 490 | 10m | Water bodies, land cover classification |
| **B3 (Green)** | 560 | 10m | Vegetation health monitoring |
| **B4 (Red)** | 665 | 10m | Vegetation detection (NDVI) |
| **B8 (NIR)** | 842 | 10m | Vegetation & crop health |
| **B11 (SWIR)** | 1610 | 20m | Soil & moisture detection |

📍 **Region Used:** *Le Creusot, France*  
[4.397836, 46.7946465, 4.4203335, 46.80950475]


---

## 🎨 Preprocessing Techniques

### 🔹 Contrast Stretching  
Enhances the difference between dark and light areas.

### 🔹 Histogram Equalization  
Redistributes pixel intensities for better contrast.

### 🔹 Gamma Correction  
- **Gamma < 1**: Enhances darker tones  
- **Gamma > 1**: Reduces brightness  
- **Gamma = 1**: No change  

📌 **Best Result:** **Gamma Correction** provided the most visually appealing enhancement.

---

## 🌿 Feature Detection & Analysis

### 🏡 **Vegetation Detection (NDVI)**
\[
NDVI = \frac{(NIR - Red)}{(NIR + Red)}
\]
- **Healthy vegetation** → High NDVI values  
- **Bare soil/water** → Low NDVI values  

### 🌊 **Water Detection (NDWI)**
\[
NDWI = \frac{(Green - NIR)}{(Green + NIR)}
\]
- Highlights water bodies (Green band vs. Near-Infrared).

### 🔥 **Burned Area Detection (NBR)**
\[
NBR = \frac{(NIR - SWIR)}{(NIR + SWIR)}
\]
- Detects burned land areas after wildfires.

---

## 🏙 Building Detection (YOLOv8)

- **Model:** CSPDarkNet-based **YOLOv8**
- **Dataset Used:** Sentinel-2 Land Use Land Cover (LULC)
- **Classes:**
  - River
  - Built Area (Buildings)
  - Trees/Crops
  - Bare Ground
  - Snow
  - Rangeland

### 🚀 **Challenges & Solutions**
🔴 **Problem:** Low-resolution Sentinel-2 images caused false positives.  
✅ **Solution:** **Grid-based Image Splitting & Stitching**
1. **Divided images into 16 smaller tiles** for zoomed detection.
2. **Detected buildings in each sub-image separately**.
3. **Merged** detections for **high-resolution output**.

---

## 🚀 Deployment

### 🔧 **Backend - FastAPI**
- API Endpoints:
  - `POST /save_bbox/` → Saves bounding box coordinates.
  - `GET /get_images/` → Retrieves processed images.

### 🗺 **Frontend - Interactive Map**
- Uses **Leaflet.js** for user **bounding box selection**.
- API fetches & processes satellite images dynamically.

---

## ✅ Testing & Validation

### 🔍 **1. Detection Accuracy**
- **Intersection over Union (IoU)** for bounding box accuracy.
- Analyzed **False Positives (FP)** & **False Negatives (FN)**.

### ⏳ **2. Performance Metrics**
- **Inference Time**: Model processing speed.
- **API Latency**: Image request-response time.

### 🖥 **3. User Interaction Testing**
- Validated **FastAPI** responses for errors.
- Ensured **intuitive UI usability** with interactive maps.

---

## ⚠️ Limitations & Future Improvements

### **⚠️ Current Limitations**
- **Low Satellite Resolution** → Sentinel-2 is **too low-resolution** for fine object detection.
- **Pretrained Model Issues** → YOLOv8 model was not originally trained on Sentinel-2.
- **Local Deployment** → Currently **hosted locally**, needs **cloud hosting**.

### **🚀 Future Enhancements**
✅ Use **hyperspectral satellite imagery** for better classification.  
✅ Integrate **AI-based anomaly detection** for disaster monitoring.  
✅ Deploy a **cloud-based web app** for real-time access.

---

## 📺 Demo

📽 **Watch the demo video here**:  
[![Demo Video](https://img.shields.io/badge/Watch-Video-red?style=for-the-badge&logo=youtube)](https://docs.google.com/file/d/1Gy1vpioK7bMfWgAd-zFmzuYRSgzb8GRv/preview)

---

## 🔧 Installation & Setup

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/osama-fawad/satellite-imaging-project.git
cd satellite-imaging-project

### 2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
### 3️⃣ Run FastAPI Backend
bash
Copy
Edit
uvicorn app:main --reload
API available at: http://127.0.0.1:8000/docs
### 4️⃣ Launch Interactive Map
Open index.html in a browser.
