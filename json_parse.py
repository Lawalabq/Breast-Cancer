import json
import pydicom
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import os


def get_coordinates(annotation):
    coordinates = []
    x= annotation.get("x")
    y= annotation.get("y")
    coordinates.append(x)
    coordinates.append(y)
    return coordinates


def get_dicom(dicom_id,dicom_view,dicom_laterilty):
    #Change directory to dicom dataset path
    dataset_path = "C:\\Users\\alqud\\Desktop\\2025\\Breast Cancer\\dataset-uta7-dicom\\dataset"
    os.chdir(dataset_path)
    patient_dict= {"6244f5c5-faf5-4e52-83e6-74fee6476dec":"Anonymized1",
                   "829de0c1-5e7a-4b9a-978a-537922a46b35":"Anonymized2",
                   "fb8c2ffc-5f8c-4ac7-a8d9-74b3c56b69a4":"Anonymized3",
                   "b191c30e-db18-4d25-b156-6b6bc2dda995":"Anonymized4"
                   }
    for filename in glob.glob(patient_dict[dicom_id]+"\\*.dcm"): # Loop through all DICOM files in the directory

        
        ds = pydicom.dcmread(filename)
        if ds.get("ViewPosition") == dicom_view and ds.get("ImageLaterality") == dicom_laterilty:
            print(f"Matched DICOM file: {filename}")
            return ds



def create_mask(dicom_image, annotation_data):
    """
    Reads a DICOM and an annotation object, returns the Image and the Binary Mask.
    """
    
    # A. Read the DICOM file
    # If you don't have the real file yet, we can create a dummy image for testing

    ds = dicom_image
    image_data = ds.pixel_array
    height, width = image_data.shape
    # B. Create an empty black mask (same size as image)
    mask = np.zeros((height, width), dtype=np.uint8)

    # C. Extract coordinates
    # We look inside the first item of 'freehand' -> 'handles'
    roi_points = []
    for point in annotation_data:
         x = int(point[0]) 
         y = int(point[1])
         roi_points.append([x, y])
    if roi_points:
        # Convert to numpy array of shape (N, 1, 2)
        roi_array = np.array([roi_points], dtype=np.int32)

        # D. Draw the filled polygon onto the mask
        # 255 represents white (the tumor), 0 is black (background)
        cv2.fillPoly(mask, roi_array, 255)

    return image_data, mask

def overlay_mask_on_image(image, mask, alpha=0.4, ax=None, show=True):
    """
    Overlay a binary mask on a grayscale image using a light green color.

    Parameters:
    - image: 2D numpy array (grayscale) or 3D RGB image
    - mask: 2D numpy array with mask pixels >0 indicating the region
    - alpha: float in [0,1], transparency of the green overlay
    - ax: optional Matplotlib Axes to draw on
    - show: if True, call plt.show()

    Returns:
    - ax: Matplotlib Axes with the overlay plotted
    """

    # Ensure mask is boolean
    mask_bool = mask is not None and (mask > 0)

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Display base image
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    # Create RGBA overlay: light green (0,1,0) with given alpha where mask==True
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=float)
    overlay[mask_bool] = [0.0, 1.0, 0.0, alpha]

    ax.imshow(overlay)
    ax.set_axis_off()

    if show:
        plt.show()

    return ax



def get_dataset():

            dataset_path = "C:\\Users\\alqud\\Desktop\\2025\\Breast Cancer\\dataset-uta7-annotations\\dataset" 
            os.chdir(dataset_path)
            output_dict= {}
            for json_file in glob.glob("*.json"): # Loop through all JSON files in the directory
                dataset_path = "C:\\Users\\alqud\\Desktop\\2025\\Breast Cancer\\dataset-uta7-annotations\\dataset" 
                os.chdir(dataset_path)
                if json_file not in ["b191c30e-db18-4d25-b156-6b6bc2dda995.json","6244f5c5-faf5-4e52-83e6-74fee6476dec.json","829de0c1-5e7a-4b9a-978a-537922a46b35.json"]:
                    continue
                print(f"Loading annotations from: {json_file}")
                
                with open(json_file, 'r') as f:
                    json_data = json.load(f)

                    json_stacks= json_data.get("rawData", {}).get("stacks", [])
                    for stack in json_stacks:
                        if  "freehand" in stack.keys():
                            annotations = "freehand" if "freehand" in stack.keys() else "probe"
                            if stack.get("seriesDescription") != "MG":
                                continue
                            view_position= stack.get("viewPosition")
                            laterality= stack.get("imageSide")
                            handles =stack.get(annotations, {})[0].get("handles")
                            coordinates= []
                            for handle in handles:
                                coordinates.append(get_coordinates(handle))
                            image_string= json_data.get("rawData", {}).get("studyId") + "_" + view_position + "_" + laterality
                            
                            dicom_ds= get_dicom(json_data.get("rawData", {}).get("studyId"),view_position,laterality)
                            image_data, mask=create_mask(dicom_ds,coordinates)
                            output_dict[image_string+"_image"]= image_data
                            output_dict[image_string+"_mask"]= mask
                            

                        elif "probe" in stack.keys():
                            annotations = "probe" 
                            if stack.get("seriesDescription") != "MG":
                                continue
                            view_position= stack.get("viewPosition")
                            laterality= stack.get("imageSide")
                            coordinates= []
                            for handle in handles:
                                coordinates.append(get_coordinates(handle))
                            dicom_ds= get_dicom(json_data.get("rawData", {}).get("studyId"),view_position,laterality)
                            image_data, mask=create_mask(dicom_ds,coordinates)
                            image_string= json_data.get("rawData", {}).get("studyId") + "_" + view_position + "_" + laterality
                            output_dict[image_string+"_image"]= image_data
                            output_dict[image_string+"_mask"]= mask
            
            return output_dict
                            
                        #     print(f"Coordinates: {coords}")
                    
dict=get_dataset()
   
