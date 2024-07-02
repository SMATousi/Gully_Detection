import zipfile
import os
import xml.etree.ElementTree as ET

def extract_kml_from_kmz(kmz_path, extraction_root):
    # Ensure the extraction directory exists
    extract_path = os.path.join(extraction_root, 'extracted_kml')
    os.makedirs(extract_path, exist_ok=True)
    
    # Unzip the KMZ file
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kmz.extractall(extract_path)
    
    # Find the KML file in the extracted files
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.kml'):
                return os.path.join(root, file)
    
    raise FileNotFoundError("No KML file found in the KMZ archive")

def get_bounding_box_from_kml(kml_path):
    # Parse the KML file
    tree = ET.parse(kml_path)
    root = tree.getroot()
    
    # Define the namespaces
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Find all coordinates in the KML file
    coordinates = []
    for placemark in root.findall('.//kml:Placemark', namespaces):
        for polygon in placemark.findall('.//kml:Polygon', namespaces):
            for coord in polygon.findall('.//kml:coordinates', namespaces):
                coords_text = coord.text.strip()
                for coord_pair in coords_text.split():
                    lon, lat, _ = map(float, coord_pair.split(','))
                    coordinates.append((lat, lon))
    
    # Calculate the bounding box
    lats, lons = zip(*coordinates)
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Return the four coordinates of the bounding box
    return (min_lon, min_lat), (max_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat)

def main_extracter(kmz_path, extraction_root):
    kml_path = extract_kml_from_kmz(kmz_path, extraction_root)
    bounding_box = get_bounding_box_from_kml(kml_path)
    return bounding_box


