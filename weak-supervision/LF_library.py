from LF_utils import *




def detect_repetitive_lines(images, 
                            threshold1=100, 
                            threshold2=200, 
                            threshold=50, 
                            min_line_length=10, 
                            max_line_gap=5, 
                            repetition_threshold=1,
                            tolerance=5):

    # Convert RGB images to grayscale
    grayscale_images = convert_to_grayscale(images)
    # Perform Canny edge detection on grayscale images
    edge_images_canny = edge_detection_canny(grayscale_images, threshold1, threshold2)
    # Perform line detection on edge-detected images
    line_images, line_coords = line_detection(edge_images_canny, threshold, min_line_length, max_line_gap)
    # Check for repetitive lines
    result = check_repetitive_lines(line_coords, repetition_threshold, tolerance)
    
    return result