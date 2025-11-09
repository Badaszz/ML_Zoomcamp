import requests
import base64
import numpy as np
import cv2

url = "http://127.0.0.1:5000/predict"
file_path = "download.png"

with open(file_path, "rb") as f:
    response = requests.post(url, files={"image": f})

result = response.json()

# Decode the mask from base64
mask_data = base64.b64decode(result['min'])
mask_array = np.frombuffer(mask_data, np.uint8)
mask_img = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

# Display the mask
cv2.imshow("Predicted Mask", mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

