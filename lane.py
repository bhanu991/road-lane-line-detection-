import cv2
import numpy as np

# Load the original image
img = cv2.imread('image1.png')


# Define region of interest
roi = np.array([[(0, 720), (0, 300), (1280, 300), (1280, 720)]], dtype=np.int32)

# Define color range for detecting white and yellow colors
white_lower = np.array([200, 200, 200], dtype=np.uint8)
white_upper = np.array([255, 255, 255], dtype=np.uint8)
yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
yellow_upper = np.array([30, 255, 255], dtype=np.uint8)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny algorithm
edges = cv2.Canny(blur, 50, 150)

# Mask the edges to only include the region of interest
mask = np.zeros_like(edges)
cv2.fillPoly(mask, roi, 255)
masked_edges = cv2.bitwise_and(edges, mask)

# Detect white and yellow colors using color thresholding
white_mask = cv2.inRange(img, white_lower, white_upper)
yellow_mask = cv2.inRange(img, yellow_lower, yellow_upper)
color_mask = cv2.bitwise_or(white_mask, yellow_mask)

# Combine the masked edges and color mask
combined_mask = cv2.bitwise_or(masked_edges, color_mask)

# Apply Hough transform to detect lines
lines = cv2.HoughLinesP(combined_mask, rho=1, theta=np.pi/180, threshold=40, minLineLength=100, maxLineGap=50)

# Filter out the unwanted lines
if lines is not None:
    left_lane_points = []
    right_lane_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue  # skip vertical lines
        slope = dy / dx
        if abs(slope) < 0.5:
            continue  # skip lines that are too horizontal
        if slope < 0:
            left_lane_points.extend([(x1, y1), (x2, y2)])
        else:
            right_lane_points.extend([(x1, y1), (x2, y2)])

    # Fit a polynomial to the points for each lane
    if len(left_lane_points) > 0:
        left_x, left_y = zip(*left_lane_points)
        left_fit = np.polyfit(left_y, left_x, 2)
        left_fitx = left_fit[0] * np.square(np.array(left_y)) + left_fit[1] * np.array(left_y) + left_fit[2]

        left_line = np.array([np.vstack((left_fitx, left_y)).astype(np.int32).T])
        cv2.polylines(img, left_line, False, (0, 255, 0), thickness=2)

    if len(right_lane_points) > 0:
     right_x, right_y = zip(*right_lane_points)
     right_fit = np.polyfit(right_y, right_x, 2)
     right_fitx = right_fit[0] * np.square(np.array(right_y)) + right_fit[1] * np.array(right_y) + right_fit[2]
     right_fitx = right_fit[0] * np.square(np.array(right_y)) + right_fit[1] * np.array(right_y) + right_fit[2]

     right_line = np.array([np.vstack((right_fitx, right_y)).astype(np.int32).T])
     cv2.polylines(img, right_line, False, (0, 255, 0), thickness=2)

    cv2.imshow('Lane Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

       
