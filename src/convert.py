import cv2

image = cv2.imread("img/source_images/photo1.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Could not load file")

print(image.shape)  # (height, width)
print(image)  # Array of pixel values
