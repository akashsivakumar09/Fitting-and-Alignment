
import cv2
def measure_earrings(image_path):
    focal_length_mm = 8.0
    distance_mm = 720.0
    pixel_size_mm = 2.2 * 1e-3 

    mm_per_pixel = pixel_size_mm * (distance_mm / focal_length_mm)
    print(f"Camera Scale: 1 pixel = {mm_per_pixel:.3f} mm\n")

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        width_mm = w * mm_per_pixel
        height_mm = h * mm_per_pixel
        
        print(f"Earring {i+1}:")
        print(f"  Pixel dimensions: {w}px width x {h}px height")
        print(f"  Physical size:    {width_mm:.2f}mm width x {height_mm:.2f}mm height")
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Measured Earrings', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

measure_earrings('earrings.jpg')