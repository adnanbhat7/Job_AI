# Import necessary libraries
import easyocr
import cv2
from matplotlib import pyplot as plt
import os
from pdf2image import convert_from_path  # Import pdf2image

# Path to PDF and output folder
PDF_PATH = 'txt.pdf'
OUTPUT_DIR = 'output_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convert PDF pages to images using pdf2image
image_paths = []
pages = convert_from_path(PDF_PATH, dpi=300)  # Convert PDF to images at 300 DPI

for i, page in enumerate(pages):
    image_path = os.path.join(OUTPUT_DIR, f'page_{i + 1}.jpeg')
    page.save(image_path, 'JPEG')  # Save each page as JPEG
    image_paths.append(image_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Process each image and extract text
all_detected_text = []  # Store text from all pages
for image_path in image_paths:
    # Read the image
    img = cv2.imread(image_path)

    # Preprocess the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Binarization
    denoised = cv2.medianBlur(binary, 3)  # Noise removal using median blur

    # Perform OCR on the preprocessed image
    result = reader.readtext(denoised)

    detected_text = []  # Text for the current page
    font = cv2.FONT_HERSHEY_SIMPLEX
    spacer = 100

    # Draw boxes and extract text
    for detection in result:
        # Convert coordinates to integers
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        text = detection[1]  # Extract detected text

        # Append detected text to the list
        detected_text.append(text)

        # Draw rectangle and put text
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
        img = cv2.putText(img, text, (20, spacer), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        spacer += 15

    # Save detected text for the current page
    all_detected_text.append("\n".join(detected_text))

    # Display the image with detected text
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Page {image_paths.index(image_path) + 1}")
    plt.show()

# Combine all detected text into a single string
final_detected_text = "\n\n".join(all_detected_text)

# Print the final detected text
print("Final Detected Text:")
print(final_detected_text)
