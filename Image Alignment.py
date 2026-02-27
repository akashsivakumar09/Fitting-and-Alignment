import cv2
import numpy as np
import matplotlib.pyplot as plt
points = []
img_display = None

def mouse_callback(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 Points", img_display)
            
        if len(points) == 4:
            print("\nFour points selected. Press any key to continue to the transformation...")

def main():
    global img_display, points

    turf_img = cv2.imread("turf.jpg")
    flag_img = cv2.imread("flag.png") 
    
    if turf_img is None or flag_img is None:
        raise FileNotFoundError("Could not find turf.jpg or flag.jpg. Check your file paths.")

    img_display = turf_img.copy()
    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", mouse_callback)
    
    print("Click 4 corners on the turf in this exact order:")
    print("1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left")
    
    cv2.imshow("Select 4 Points", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) != 4:
        print("You didn't select 4 points. Exiting.")
        return

    pts_dst = np.array(points, dtype=np.float32)

    h, w = flag_img.shape[:2]
    pts_src = np.array([
        [0, 0],       
        [w, 0],       
        [w, h],       
        [0, h]        
    ], dtype=np.float32)

    matrix, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)

    turf_h, turf_w = turf_img.shape[:2]
    warped_flag = cv2.warpPerspective(flag_img, matrix, (turf_w, turf_h))

    gray_warped = cv2.cvtColor(warped_flag, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)
    
    turf_bg = cv2.bitwise_and(turf_img, turf_img, mask=mask_inv)
    
    final_result = cv2.add(turf_bg, warped_flag)
    final_result_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(final_result_rgb)
    plt.title("Homography Result")
    plt.axis('off') 
    plt.show()

if __name__ == "__main__":
    main()

