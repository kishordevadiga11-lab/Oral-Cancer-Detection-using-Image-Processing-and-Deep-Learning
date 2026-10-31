import cv2

def preprocess(image_path, size1, size2):
    img = cv2.imread(image_path)

    if img is None:
        return None, None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img1 = cv2.resize(img_rgb, (size1, size1)) / 255.0
    img2 = cv2.resize(img_rgb, (size2, size2)) / 255.0

    return img_rgb, img1, img2