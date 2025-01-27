import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox
import os
import cv2
from ultralytics import YOLO
from ipyleaflet import Map, DrawControl, basemaps  # For creating interactive maps and adding drawing controls
import json
from IPython.display import display  # For displaying the map in a Jupyter Notebook
import time


def run(bbox_coordinates):
    
    # Save the coordinates in Sentinel Hub's BBox format
    le_creusot_bbox = BBox(
        bbox=[
            bbox_coordinates['longitude_min'],
            bbox_coordinates['latitude_min'],
            bbox_coordinates['longitude_max'],
            bbox_coordinates['latitude_max']
        ],
        crs=CRS.WGS84  # Specify that the coordinates are in WGS84 format
    )

    # Print the BBox variable for confirmation
    print(f"BBox for Sentinel Hub API: {le_creusot_bbox}")

    # ----------------------------
    # Initialize Sentinel Hub Configuration
    # ----------------------------
    # SentinelHub requires authentication to access its API. We need:
    # - instance_id: ID associated with SentinelHub instance configuration -> after creating a new configuration
    # - sh_client_id and sh_client_secret: These are generated when we create OAuth credentials on SentinelHub.
    config = SHConfig()
    config.instance_id = ''  # instance_id
    config.sh_client_id = ''   # client_id
    config.sh_client_secret = ''  # client_secret

    # ----------------------------
    # Evalscript: Specify which bands we want to retrieve
    # ----------------------------
    # Sentinel-2 has multiple bands (representing different parts of the electromagnetic spectrum).
    # For vegetation analysis, we are interested in:
    # - Visible bands (Red, Green, Blue) -> B04, B03, B02
    # - NIR (Near-Infrared) -> B08
    # - SCL (Scene Classification Layer) -> (optional, but can be used to mask clouds, etc.)
    # These bands are given as input in the 'evalscript'.
    evalscript_all_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08", "B11"],  // Blue, Green, Red, NIR, and SWIR bands
                units: "REFLECTANCE"
            }],
            output: {
                bands: 5,  // Expecting 5 output bands now (Blue, Green, Red, NIR, SWIR)
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11];  // Return Blue, Green, Red, NIR, SWIR
    }
    """

    # ----------------------------
    # Create a SentinelHubRequest to fetch data
    # ----------------------------
    # - DataCollection.SENTINEL2_L2A: We are using Sentinel-2 Level 2A data, which provides surface reflectance (already atmospherically corrected).
    # - time_interval: We want the latest available image from 2024.
    # - mosaicking_order: This tells SentinelHub how to handle overlapping images (if there are multiple scenes).
    request = SentinelHubRequest(
        evalscript=evalscript_all_bands,  # Our evalscript to select the bands
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,  # Using Level-2A data
                time_interval=('2024-08-24', '2024-08-25'),  # Specify the date range for data
                other_args={
                    "maxcc": 0.2,  # Maximum cloud coverage set to 20%
                    "mosaickingOrder": "mostRecent"  # Use the most recent images within the time interval
                }
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)  # Output in TIFF format
        ],
        bbox=le_creusot_bbox,  # The bounding box for Le Creusot
        size=(512, 512),  # Size of the image (pixels). You can adjust this based on your needs.
        data_folder='satellite_images',  # Specify the folder to save data
        config=config  # The SentinelHub config (authentication)
    )

    # ----------------------------
    # Explanation of the Input Parameters:
    # ----------------------------
    # - DataCollection.SENTINEL2_L2A: Sentinel-2 L2A provides surface reflectance data, corrected for atmospheric conditions.
    # - time_interval: Specifies the date range for the data (2024-01-01 to 2024-12-31). You can adjust it based on your needs.
    # - mosaicking_order='mostRecent': Chooses the most recent image if multiple images overlap for the given area and date range.
    # - bbox (Bounding Box): Defines the area of interest by providing 4 coordinates (longitude, latitude). The format is:
    #   [lower-left longitude, lower-left latitude, upper-right longitude, upper-right latitude].
    # - size=(512, 512): Specifies the resolution (width x height) of the image to be fetched in pixels.

    # ----------------------------
    # Output Information:
    # ----------------------------
    # - Image size: 512x512 pixels
    # - Number of channels (bands): 4 (Blue, Green, Red, NIR)
    # - Format: TIFF (stored as Float32 reflectance values)

    # Get the data (list of images containing Blue, Green, Red, and NIR bands)
    images = request.get_data(save_data=True)
    image = images[0]
    # Only proceed if images are present (retreived)
    if len(images) > 0:
        print('Images Retreived - Can Proceed - Number of images =',len(images))
        # The first image contains all bands (Blue, Green, Red, NIR)
        image = images[0]   
    else:
        'No Image Present'

    # Extract RGB (Red, Green, Blue) and NIR (Near Infrared)
    blue_band = image[:, :, 0]
    green_band = image[:, :, 1]
    red_band = image[:, :, 2]
    nir_band = image[:, :, 3]
    swir_band = image[:, :, 4]
    # each of these arrays are reflectance values from the satellite normalised between 0 to 1

    ## Convert sentinel band into opencv image format ##

    # The values in the band arrays (e.g., red_band, green_band, blue_band) are reflectance values in the range [0, 1] 
    # because Sentinel data often represents reflectance as normalized values.
    # To process them for OpenCV (which typically expects images in the range [0, 255] of type uint8), we should scale these values and convert the data type.

    # red_band is [512,512] sized array with values between 0-1 and float32 data type, same for other bands.

    # Step 1 - ensure reflectance range is [0,1]
    # Step 2 - scale to [0,255] by multiplying the normalised values to 255
    # Step 3 - convert to unit8 (8 bit integers)
    # Step 4 - stack channels to create rgb image using np.dstack


    # Verify the properties of the red band
    print('Red band')
    print(f"Data type: {red_band.dtype}")
    print(f"Shape: {red_band.shape}")
    print(f"Min value: {red_band.min()}")
    print(f"Max value: {red_band.max()}")
    # already between 0 to 1 as expected


    # Scale the bands to [0, 255] and convert to uint8
    red_band_255 = (red_band * 255).astype('uint8')
    green_band_255 = (green_band * 255).astype('uint8')
    blue_band_255 = (blue_band * 255).astype('uint8')

    print('\nRed Band Normalised between 0-255')
    print(f"Data type: {red_band_255.dtype}")
    print(f"Shape: {red_band_255.shape}")
    print(f"Min value: {red_band_255.min()}")
    print(f"Max value: {red_band_255.max()}")

    # Stack the bands into an RGB image
    opencv_rgb_image = np.dstack((red_band_255, green_band_255, blue_band_255))

    # Verify the properties of the OpenCV-ready image
    print('\nRGB Opencv format image')
    print(f"Data type: {opencv_rgb_image.dtype}")
    print(f"Shape: {opencv_rgb_image.shape}")
    print(f"Min value: {opencv_rgb_image.min()}")
    print(f"Max value: {opencv_rgb_image.max()}")

    # ------------------------------
    # Check and Create 'saved_results' Folder
    # ------------------------------
    output_folder = "saved_results"

    # Check if the folder exists, if not, create it
    os.makedirs(output_folder, exist_ok=True)

    # ------------------------------
    # Display and Save the RGB Satellite Image
    # ------------------------------
    plt.figure(figsize=(10, 10))
    plt.imshow(opencv_rgb_image)  # Assuming 'opencv_rgb_image' is defined
    plt.axis('off')  # Turn off axes for better visualization
    plt.title("RGB Satellite Image")

    # Save the plot in the 'saved_results' folder
    save_path = os.path.join(output_folder, "Original Unprocessed RGB Satellite Image.png")
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure with no extra whitespace

    # Display the plot
    # plt.show()

    print(f"Image saved to {save_path}")

    ## Enhance visibility

    # option 1: Contrast Stretching (Linear Normalization) - Stretch the minimum and maximum values of your data to [0, 255]

    # Find the actual min and max values of the bands
    min_val = np.min([red_band.min(), green_band.min(), blue_band.min()])
    max_val = np.max([red_band.max(), green_band.max(), blue_band.max()])

    # Apply linear normalization
    red_band_stretched = ((red_band - min_val) / (max_val - min_val) * 255).astype('uint8')
    green_band_stretched = ((green_band - min_val) / (max_val - min_val) * 255).astype('uint8')
    blue_band_stretched = ((blue_band - min_val) / (max_val - min_val) * 255).astype('uint8')

    # Combine into an RGB image
    rgb_image_contrast_stretching = np.dstack((red_band_stretched, green_band_stretched, blue_band_stretched))

    # Verify the new range
    print(f"Min value: {rgb_image_contrast_stretching.min()}, Max value: {rgb_image_contrast_stretching.max()}")

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image_contrast_stretching)  # Matplotlib expects RGB order
    plt.axis('off')  # Turn off axes for better visualization
    plt.title("Enhanced RGB Image - Contrast Stretching")

    # Save the plot in the 'saved_results' folder
    save_path = os.path.join(output_folder, "Enhanced RGB Image - Contrast Stretching.png")
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure with no extra whitespace

    # plt.show()


    ## Contrast enhancing

    # option 2 - Adaptive Histogram Equalization

    # Combine the scaled bands
    opencv_rgb_image = np.dstack((red_band_255, green_band_255, blue_band_255))

    # Convert to LAB color space for better contrast enhancement
    lab_image = cv2.cvtColor(opencv_rgb_image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])  # Apply to L-channel

    # Convert back to RGB
    rgb_image_histogram_equalization = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    # Verify the new range
    print(f"Min value: {rgb_image_histogram_equalization.min()}, Max value: {rgb_image_histogram_equalization.max()}")

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image_histogram_equalization)  # Matplotlib expects RGB order
    plt.axis('off')  # Turn off axes for better visualization
    plt.title("Enhanced RGB Image - Histogram Equilisation")

    # Save the plot in the 'saved_results' folder
    save_path = os.path.join(output_folder, "Enhanced RGB Image - adaptive Histogram Equilisation.png")
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure with no extra whitespace

    # plt.show()


    ## Contrast enhancing

    # option 3 - Gamma Correction

    # Scale to [0, 255] as before
    red_band_255 = (red_band * 255).astype('uint8')
    green_band_255 = (green_band * 255).astype('uint8')
    blue_band_255 = (blue_band * 255).astype('uint8')

    # Define a gamma value (e.g., 1.5 for brightening)
    gamma = 1.7
    gamma_correction = lambda band: ((band / 255) ** (1 / gamma) * 255).astype('uint8')

    # Apply gamma correction
    red_band_gamma = gamma_correction(red_band_255)
    green_band_gamma = gamma_correction(green_band_255)
    blue_band_gamma = gamma_correction(blue_band_255)

    # Combine into an RGB image
    rgb_image_gamma_correction = np.dstack((red_band_gamma, green_band_gamma, blue_band_gamma))

    # Verify the new range
    print(f"Min value: {rgb_image_gamma_correction.min()}, Max value: {rgb_image_gamma_correction.max()}")

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image_gamma_correction)  # Matplotlib expects RGB order
    plt.axis('off')  # Turn off axes for better visualization
    plt.title("Enhanced RGB Image - Gamma Correction")

    # Save the plot in the 'saved_results' folder
    save_path = os.path.join(output_folder, "Enhanced RGB Image - Gamma Correction.png")
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure with no extra whitespace

    # plt.show()

    # ----------------------------
    # Compute NDVI (Vegetation Detection)
    # ----------------------------
    # NDVI (Normalized Difference Vegetation Index) measures the health and density of vegetation.
    # Formula: NDVI = (NIR - Red) / (NIR + Red)
    # NIR reflects strongly from vegetation, whereas Red is absorbed by plants, so their difference indicates vegetation health.
    ndvi = (nir_band - red_band) / (nir_band + red_band)

    # ----------------------------
    # NDVI Intuition:
    # ----------------------------
    # NDVI measures the difference between the NIR (Near-Infrared) and Red bands, both of which are affected by vegetation.
    # Healthy vegetation reflects a lot of NIR and absorbs Red light, resulting in high NDVI values.
    # - NDVI > 0.5: Dense vegetation
    # - NDVI between 0.2 and 0.5: Moderate vegetation
    # - NDVI between 0 and 0.2: Bare soil or sparse vegetation
    # - NDVI < 0: Non-vegetated surfaces (water, urban areas)

    # Plot NDVI
    plt.figure(figsize=(10, 10))
    plt.imshow(ndvi, cmap='RdYlGn')  # Use a colormap where green represents vegetation
    plt.colorbar(label='NDVI')
    plt.title('NDVI (Vegetation Detection)')
    plt.axis('off')

    # Save the plot in the 'saved_results' folder
    save_path = os.path.join(output_folder, "NDVI - Normalised Difference Vegetation  Index.png")
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure with no extra whitespace

    # plt.show()

    # ----------------------------
    # Compute NDWI (Water Detection)
    # ----------------------------
    # NDWI (Normalized Difference Water Index) highlights water bodies by comparing the reflectance
    # in the Green band (which reflects more from water) to the Near-Infrared (NIR) band (which water absorbs strongly).
    # Formula: NDWI = (Green - NIR) / (Green + NIR)
    # High NDWI values (close to 1) indicate water bodies, while lower or negative values indicate other features such as vegetation or soil.
    ndwi = (green_band - nir_band) / (green_band + nir_band)

    # ----------------------------
    # NDWI Intuition:
    # ----------------------------
    # NDWI leverages the difference between the Green and NIR bands to identify water bodies:
    # - High NDWI values (close to 1): Indicate the presence of water bodies.
    # - NDWI between 0 and 0.2: Bare soil or non-water surfaces.
    # - NDWI < 0: Vegetated surfaces or urban areas.
    # It is particularly useful in remote sensing for detecting rivers, lakes, and other water bodies while distinguishing them from surrounding land.

    # Plot NDWI
    plt.figure(figsize=(10, 10))
    plt.imshow(ndwi,cmap='RdBu' )  # Use a colormap where green represents vegetation
    plt.colorbar(label='NDWI')
    plt.title('NDWI (Water Detection)')
    plt.axis('off')

    # Save the plot in the 'saved_results' folder
    save_path = os.path.join(output_folder, "NDWI - Normalised Difference Water Index.png")
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure with no extra whitespace

    # plt.show()


    # ----------------------------
    # Compute NBR (Burned Area Detection)
    # ----------------------------
    # NBR (Normalized Burn Ratio) is used to identify burned areas and assess burn severity.
    # Formula: NBR = (NIR - SWIR) / (NIR + SWIR)
    # NIR reflects strongly from healthy vegetation, while SWIR reflects strongly from burned areas or bare soil.

    nbr = (nir_band - swir_band) / (nir_band + swir_band)

    # ----------------------------
    # NBR Intuition:
    # ----------------------------
    # NBR measures the difference between the Near-Infrared (NIR) and Short-Wave Infrared (SWIR) bands.
    # - Healthy vegetation reflects NIR and absorbs SWIR, leading to high NBR values.
    # - Burned areas and bare soil reflect more SWIR and less NIR, resulting in low or negative NBR values.
    #
    # Typical NBR Ranges:
    # - NBR > 0.1: Healthy vegetation
    # - NBR between 0 and 0.1: Sparse vegetation or bare soil
    # - NBR < 0: Burned areas or recently disturbed land

    # ----------------------------
    # Plot NBR
    # ----------------------------
    plt.figure(figsize=(10, 10))
    plt.imshow(nbr, cmap='RdYlGn')  # Use Red-Yellow-Green colormap for burn severity
    plt.colorbar(label='NBR')
    plt.title('NBR (Burned Area Detection)')
    plt.axis('off')

    # Save the plot in the 'saved_results' folder
    save_path = os.path.join(output_folder, "NBR - Normalised Burn Ratio.png")
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure with no extra whitespace

    # plt.show()

    model = YOLO("yolov8_building_detection_saved_weights.pt")  # YOLOv8 model weights file

    def yolo_function(model, rgb_image, conf_threshold):
        # Perform Inference on the RGB Image
        results = model.predict(rgb_image, save=False, save_txt=False, save_crop=False, conf=conf_threshold, iou=0.9)

        # Extract Predictions
        predictions = results[0]

        # Filter Predictions for the 'building' Class
        target_class = "building"
        filtered_indices = [
            i for i, cls_idx in enumerate(predictions.boxes.cls.cpu().numpy())
            if model.names[int(cls_idx)] == target_class
        ]

        # Extract bounding boxes and confidence scores
        filtered_boxes = predictions.boxes[filtered_indices]
        filtered_scores = predictions.boxes.conf[filtered_indices].cpu().numpy()
        filtered_classes = predictions.boxes.cls[filtered_indices].cpu().numpy()

        # Copy the RGB image for visualization
        visualized_image = rgb_image.copy()

        # Check if masks are available
        if predictions.masks is not None:
            filtered_masks = predictions.masks[filtered_indices]

            # Loop through filtered predictions and draw on the image
            for box, mask, score, cls_idx in zip(filtered_boxes, filtered_masks.data, filtered_scores, filtered_classes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Process the mask
                mask = mask.cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)
                mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]))

                # Overlay the mask with a blue color
                visualized_image[mask == 1] = [0, 0, 150]

                # Draw the bounding box
                cv2.rectangle(visualized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add the label
                label = f"{model.names[int(cls_idx)]} {score:.2f}"
                cv2.putText(visualized_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        else:
            # If no masks are available, just draw the bounding boxes
            for box, score, cls_idx in zip(filtered_boxes, filtered_scores, filtered_classes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw the bounding box
                cv2.rectangle(visualized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add the label
                label = f"{model.names[int(cls_idx)]} {score:.2f}"
                cv2.putText(visualized_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return visualized_image 

    # ------------------------------
    # Split the Bounding Box into Smaller Grids
    # ------------------------------
    def split_bbox(bbox, rows, cols):
        """
        Splits a bounding box into a grid of smaller bounding boxes.

        Args:
        - bbox (BBox): The original bounding box to split.
        - rows (int): Number of rows in the grid.
        - cols (int): Number of columns in the grid.

        Returns:
        - List of smaller BBox objects.
        """
        # Unpack the original bounding box
        min_lon, min_lat, max_lon, max_lat = bbox
        lon_step = (max_lon - min_lon) / cols
        lat_step = (max_lat - min_lat) / rows

        # Create smaller boxes by iterating through rows and columns
        sub_boxes = []
        for i in range(rows):
            for j in range(cols):
                sub_min_lon = min_lon + j * lon_step
                sub_max_lon = sub_min_lon + lon_step
                sub_min_lat = min_lat + i * lat_step
                sub_max_lat = sub_min_lat + lat_step

                sub_boxes.append(BBox([sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat], crs=CRS.WGS84))

        return sub_boxes

    # Split the bounding box into a 4x4 grid (16 smaller boxes)
    rows, cols = 4, 4
    sub_boxes = split_bbox(le_creusot_bbox, rows, cols)

    # ------------------------------
    # SentinelHub Evalscript for Band Retrieval
    # ------------------------------
    evalscript_all_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{ bands: ["B02", "B03", "B04", "B08"], units: "REFLECTANCE" }],
            output: { bands: 4, sampleType: "FLOAT32" }
        };
    }
    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B08];
    }
    """

    # ------------------------------
    # Function to Perform Gamma Correction
    # ------------------------------
    def gamma_correction(band, gamma=1.7):
        return ((band / 255) ** (1 / gamma) * 255).astype('uint8')

    # ------------------------------
    # Function for YOLOv8 Inference and Visualization
    # ------------------------------
    def yolo_function(model, rgb_image, conf_threshold):
        results = model.predict(rgb_image, save=False, save_txt=False, save_crop=False, conf=conf_threshold, iou=0.9)
        predictions = results[0]

        # Copy the image for visualization
        visualized_image = rgb_image.copy()

        if predictions.masks is not None:
            for box, mask, score, cls_idx in zip(predictions.boxes, predictions.masks.data, predictions.boxes.conf, predictions.boxes.cls):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                mask = mask.cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)
                mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]))

                visualized_image[mask == 1] = [0, 0, 150]
                cv2.rectangle(visualized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[int(cls_idx)]} {score:.2f}"
                cv2.putText(visualized_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return visualized_image

    # ------------------------------
    # Retrieve, Plot, and Stitch Images
    # ------------------------------

    model = YOLO("yolov8_building_detection_saved_weights.pt")

    gamma_corrected_images = []
    visualized_images = []

    # Loop through each sub-box and process the images
    for idx, bbox in enumerate(sub_boxes):
        request = SentinelHubRequest(
            evalscript=evalscript_all_bands,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=('2024-08-24', '2024-08-25'),
                other_args={"maxcc": 0.2}
            )],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=(512, 512),
            data_folder=output_folder,
            config=config
        )

        images = request.get_data(save_data=False)
        if not images:
            print(f"No image returned for sub-box {idx}")
            continue

        image = images[0]
        blue_band = (image[:, :, 0] * 255).astype('uint8')
        green_band = (image[:, :, 1] * 255).astype('uint8')
        red_band = (image[:, :, 2] * 255).astype('uint8')

        red_band_gamma = gamma_correction(red_band)
        green_band_gamma = gamma_correction(green_band)
        blue_band_gamma = gamma_correction(blue_band)

        gamma_corrected_image = np.dstack((red_band_gamma, green_band_gamma, blue_band_gamma))
        gamma_corrected_images.append(gamma_corrected_image)

        # Visualize the gamma-corrected image
        plt.figure(figsize=(5, 3))
        plt.imshow(gamma_corrected_image)
        plt.axis('off')
        plt.title(f"Gamma Corrected - Sub-box {idx}")
        # plt.show()

        # Perform YOLO inference
        visualized_image = yolo_function(model, gamma_corrected_image, 0.5)
        visualized_images.append(visualized_image)

        # Visualize the YOLO inference result
        plt.figure(figsize=(5, 3))
        plt.imshow(visualized_image)
        plt.axis('off')
        plt.title(f"YOLO Detection - Sub-box {idx}")
        # plt.show()

    # ------------------------------
    # Corrected Function to Stitch Images in the Correct Order
    # ------------------------------
    def stitch_images(images, rows, cols):
        """
        Stitches a list of images into a single combined image in the correct top-to-bottom and left-to-right order.

        Args:
        - images (list of np.array): List of images to stitch (all in RGB format).
        - rows (int): Number of rows in the grid.
        - cols (int): Number of columns in the grid.

        Returns:
        - np.array: Combined stitched image.
        """
        # Get the height and width of the first image (assuming all images are the same size)
        img_height, img_width = images[0].shape[:2]

        # Create a blank canvas for the stitched image (in RGB format)
        stitched_image = np.zeros((rows * img_height, cols * img_width, 3), dtype=np.uint8)

        # Place each image in the correct position within the stitched image
        for idx, img in enumerate(images):
            # Calculate the correct row and column index
            row_idx = rows - 1 - (idx // cols)  # Invert the row index for correct top-to-bottom order
            col_idx = idx % cols
            stitched_image[row_idx * img_height:(row_idx + 1) * img_height,
                        col_idx * img_width:(col_idx + 1) * img_width] = img

        return stitched_image

    # Stitch gamma-corrected images
    stitched_gamma_corrected = stitch_images(gamma_corrected_images, rows, cols)

    # Stitch YOLO visualized images
    stitched_visualized = stitch_images(visualized_images, rows, cols)

    # Display the stitched gamma-corrected image
    plt.figure(figsize=(10, 10))
    plt.imshow(stitched_gamma_corrected)
    plt.axis('off')
    plt.title("Stitched Gamma-Corrected Image")
    # plt.show()

    # Display the stitched YOLO visualized image
    plt.figure(figsize=(10, 10))
    plt.imshow(stitched_visualized)
    plt.axis('off')
    plt.title("Stitched YOLO Detection Image")

    # Save the plot in the 'saved_results' folder
    save_path = os.path.join(output_folder, "YOLOv8 Building Detection on Stitched multiple sub-boxes of original Satellite Image.png")
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure with no extra whitespace

    # plt.show()
