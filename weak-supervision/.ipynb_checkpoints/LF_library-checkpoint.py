from LF_utils import *




class LineDetector:
    def __init__(self, 
                 threshold1=100, 
                 threshold2=200, 
                 threshold=50, 
                 min_line_length=10, 
                 max_line_gap=5, 
                 repetition_threshold=1,
                 tolerance=5):
        # Initialize the parameters
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.repetition_threshold = repetition_threshold
        self.tolerance = tolerance

    def detect_repetitive_lines(self, images):
        # Convert RGB images to grayscale
        grayscale_images = convert_to_grayscale(images)
        # Perform Canny edge detection on grayscale images
        edge_images_canny = edge_detection_canny(grayscale_images, self.threshold1, self.threshold2)
        # Perform line detection on edge-detected images
        line_images, line_coords = line_detection(edge_images_canny, self.threshold, self.min_line_length, self.max_line_gap)
        # Check for repetitive lines
        result = check_repetitive_lines(line_coords, self.repetition_threshold, self.tolerance)
        
        return result


class SuperPixelClassifier:
    def __init__(self,
                 num_segments,
                 circularity_threshold,
                 aspect_ratio_threshold,
                 detection_threshold
                 ):
    
        self.num_segments = num_segments
        self.circularity_threshold = circularity_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.detection_threshold = detection_threshold

    
    def run_classification(self, images):

        total_round_ratios = []
        total_elingated_ratios = []

        for image in images:

            img = image.numpy().squeeze().transpose(1,2,0)

            segmented_image, segments = generate_superpixels(img, self.num_segments)

            # Classify superpixels as round or elongated
            round_superpixels, elongated_superpixels = classify_superpixel_shape(segments, 
                                                                                 circularity_thresh=self.circularity_threshold,
                                                                                 aspect_ratio_thresh=self.aspect_ratio_threshold)

            round_ratio = len(round_superpixels)/self.num_segments
            elongated_ratio = len(elongated_superpixels)/self.num_segments

            total_round_ratios.append(round_ratio)
            total_elingated_ratios.append(elongated_ratio)

        if np.sum(np.array(total_round_ratios)>self.detection_threshold) >= 4:

            return -1

        else:

            return 0
