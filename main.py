import cv2
import numpy as np
from IPython.display import display
from google.colab import files
from PIL import Image
import io
def detect_defects():
   
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded.")
        return
    file_name = list(uploaded.keys())[0]
    image_data = uploaded[file_name]
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    image_np = np.array(image)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (600, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defect_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            defect_count += 1

    print(f"Detected defects: {defect_count}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(img_rgb))
detect_defects()
