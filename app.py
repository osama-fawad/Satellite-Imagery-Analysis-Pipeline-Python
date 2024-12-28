# app.py
# Run this on terminal --> uvicorn app:app --reload

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import json
from sentinel_api_and_detections import run
import time  # For adding a timestamp to prevent caching
import shutil

# Create an instance of FastAPI
app = FastAPI()

# Directory for saving the bounding box coordinates and processed images
OUTPUT_DIR = "output"
IMAGE_FOLDER = "saved_results"
# Clear the saved_results folder before processing new images
folder = IMAGE_FOLDER
if os.path.exists(folder):
    shutil.rmtree(folder)  # Delete the folder and all its contents

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Mount the static folder to serve images
app.mount(f"/{IMAGE_FOLDER}", StaticFiles(directory=IMAGE_FOLDER), name="images")

# Route to display the interactive map by serving the map.html file directly
@app.get("/", response_class=HTMLResponse)
async def show_map():
    with open("map.html", "r", encoding="utf-8") as f:
        map_html = f.read()
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Satellite Image Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f9;
                color: #333;
            }}
            h1 {{
                text-align: center;
                padding: 20px;
                background-color: #4CAF50;
                color: white;
                margin: 0;
            }}
            .map-container {{
                width: 80%;
                margin: 20px auto;
                border: 2px solid #ccc;
                border-radius: 10px;
                overflow: hidden;
            }}
            .waiting-message {{
                text-align: center;
                font-size: 18px;
                color: #FF5733;
                margin: 20px 0;
            }}
            .gallery {{
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-top: 20px;
            }}
            .gallery div {{
                width: 70%;
                margin: 20px 0;
                text-align: center;
            }}
            .gallery img {{
                width: 100%;
                height: auto;
                object-fit: cover;
                border: 2px solid #ddd;
                border-radius: 10px;
            }}
            .image-heading {{
                font-size: 20px;
                color: #4CAF50;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Satellite Image Analysis</h1>
        
        <div class="map-container">
            {map_html}
        </div>

        <div class="waiting-message" id="waiting-message" style="display:none;">
            Processing your request. Please wait for the images to be displayed after processing...
        </div>

        <h2 style="text-align:center; margin-top: 20px;">Processed Images</h2>
        <div id="image-gallery" class="gallery"></div>

        <script>
            function fetchImages() {{
                fetch('/get_images')
                    .then(response => response.json())
                    .then(data => {{
                        const gallery = document.getElementById('image-gallery');
                        gallery.innerHTML = '';
                        if (data.images.length > 0) {{
                            document.getElementById('waiting-message').style.display = 'none';
                        }}
                        data.images.forEach(image => {{
                            const timestamp = new Date().getTime();  // Cache-busting timestamp
                            const imgDiv = document.createElement('div');
                            imgDiv.innerHTML = `
                                <h3 class='image-heading'>${{image.split('.')[0]}}</h3>
                                <img src='/{IMAGE_FOLDER}/${{image}}?t=${{timestamp}}' alt='Processed Image'>
                            `;
                            gallery.appendChild(imgDiv);
                        }});
                    }})
                    .catch(error => console.error('Error fetching images:', error));
            }}

            function showWaitingMessage() {{
                document.getElementById('waiting-message').style.display = 'block';
            }}

            // Show waiting message immediately when page loads
            showWaitingMessage();

            // Poll for new images every 5 seconds
            setInterval(fetchImages, 5000);
            fetchImages();  // Initial call to load images immediately
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Route to receive and save the bounding box coordinates and trigger the run function
@app.post("/save_bbox/")
async def save_bbox(longitude_min: float = Form(...), longitude_max: float = Form(...),
                    latitude_min: float = Form(...), latitude_max: float = Form(...)):
    bbox_coordinates = {
        "longitude_min": longitude_min,
        "longitude_max": longitude_max,
        "latitude_min": latitude_min,
        "latitude_max": latitude_max
    }

    # Save the coordinates to a JSON file
    bbox_path = os.path.join(OUTPUT_DIR, "bbox_coordinates.json")
    with open(bbox_path, "w") as f:
        json.dump(bbox_coordinates, f, indent=4)

    # Call the run function from sentinel_api_and_detections.py
    run(bbox_coordinates)

    return JSONResponse(content={"message": "Bounding box saved and processing started", "bbox": bbox_coordinates})

# Route to fetch the list of images in the saved_results folder
@app.get("/get_images")
async def get_images():
    # Get a list of all image files with supported extensions
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".png", ".jpg", ".jpeg", ".tiff"))]
    
    # Sort image files based on their creation time (earlier files first)
    image_files.sort(key=lambda x: os.path.getctime(os.path.join(IMAGE_FOLDER, x)))

    return {"images": image_files}
