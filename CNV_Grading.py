import cv2
import numpy as np

# Load the OCT image
#image = cv2.imread('train\cnv\CNV-103044-2.jpeg', 1)
image = cv2.imread('image5.jfif', 1)


# Perform image segmentation to isolate the CNV region
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
th = cv2.medianBlur(th, 3)
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)
mask = np.zeros_like(gray)
cv2.drawContours(mask, [cnt], -1, 255, -1)

# Show the segmented CNV region
cv2.imshow("Segmented CNV region", cv2.bitwise_and(image, image, mask=mask))
cv2.waitKey(0)

# Extract area feature
area = cv2.countNonZero(mask)
cv2.imshow("Area", mask)
cv2.waitKey(0)

# Extract perimeter feature
perimeter = cv2.arcLength(cnt, True)
cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)
cv2.imshow("Perimeter", image)
cv2.waitKey(0)

# Extract compactness feature
compactness = 4 * np.pi * area / (perimeter ** 2)
#cv2.putText(image, "Compactness: {:.2f}".format(compactness), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#cv2.imshow("Compactness", image)
#cv2.waitKey(0)

# Extract mean intensity feature
mean_intensity = cv2.mean(gray, mask=mask)[0]
#cv2.putText(image, "Mean intensity: {:.2f}".format(mean_intensity), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#cv2.imshow("Mean intensity", image)
#cv2.waitKey(0)

# Extract standard deviation intensity feature
std_dev_intensity = cv2.meanStdDev(gray, mask=mask)[1][0]
#cv2.imshow("std_dev_intensity", image)
#cv2.waitKey(0)


# Define the grading criteria
thresholds = {
    "area_high": 1000,
    "compactness_high": 0.3,
    "mean_intensity_high": 80,
    "std_dev_intensity_high": 25,
    "area_medium": 1000,
    "compactness_medium": 0.1,
    "mean_intensity_medium": 70,
    "std_dev_intensity_medium": 24,
}

# Assign a grade based on the grading criteria
if area > thresholds["area_high"] and compactness < thresholds["compactness_high"] and mean_intensity > thresholds["mean_intensity_high"] and std_dev_intensity > thresholds["std_dev_intensity_high"]:
    cnv_grade = 3
    grade_text = "High"
    grade_color = (0, 0, 255) # red
elif (area > thresholds["area_medium"] and compactness < thresholds["compactness_medium"]) or (mean_intensity > thresholds["mean_intensity_medium"] and std_dev_intensity > thresholds["std_dev_intensity_medium"]):    
    cnv_grade = 2    
    grade_text = "Medium"
    grade_color = (0, 255, 255) # yellow
else:
    cnv_grade = 1
    grade_text = "Low"
    grade_color = (0, 255, 0) # green

# Draw the grade on the image and display it
cv2.putText(image, f"CNV grade: {grade_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, grade_color, 2)
cv2.imshow("CNV grading", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
