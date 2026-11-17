
import cv2
import sys
import time
import numpy
import os

# Open camera
def open_camera(cam_id=1):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_AVFOUNDATION)
    return cap

# Get frame from camera
def get_frame(device):
    ret, img = device.read()
    if not ret:
        return None
    return img

# Close camera and windows
def cleanup(device):
    device.release()
    cv2.destroyAllWindows()

# Convert image to grayscale
def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    return gray

# Image thresholding
def threshold_image(image):
    _, threshold = cv2.threshold(
        image, 170, 255, cv2.THRESH_BINARY
    )
    return threshold

# Find contours
def find_contours(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

# Draw contours on black background
def draw_contours(image, contours):
    result = image.copy()
    result[:] = 0
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
    return result

# Find circular contours
def find_ring(contours):
    rings = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0 or area < 100 or area > 5000:
            continue

        # Calculate circularity
        circularity = 4 * numpy.pi * area / (perimeter ** 2)

        if circularity > 0.8:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            rings.append((int(cx), int(cy), int(radius)))

    return rings

# Find centers of contours
def find_centers(contours):
    centers = []

    for contour in contours:
        moments = cv2.moments(contour)

        # Prevent division by zero
        if moments["m00"] == 0:
            continue

        center = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"]
        )

        center = tuple(int(round(x)) for x in center)
        centers.append(center)

    return centers

# Draw centers on image
def draw_centers(centers, image):
    for center in centers:
        cv2.circle(image, center, 20, (0, 255, 255), 2)

# Draw circles and centers on black background
def draw_rings(image, rings):
    result = image.copy()
    result[:] = 0

    for cx, cy, radius in rings:
        cv2.circle(result, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(result, (cx, cy), 4, (0, 255, 255), -1)

    return result


# Resize image
def resize_image(image, scale=0.5):
    return cv2.resize(image, None, fx=scale, fy=scale)


 # Global variable containing the 4 points selected by the user in the corners of the board
corner_point_list = []

##
# This function is called by OpenCV when the user clicks
# anywhere in a window displaying an image.
##
def mouse_click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Click at (%d,%d)" % (x, y))
        corner_point_list.append((x, y))
 
def get_transform_matrix(dev, board_size, dpi, calib_file = None):
    corner_point_list.clear()
    # Read a frame from the video device
    img = get_frame(dev)
 
    # Displace image to user
    cv2.imshow("Calibrate", img)
 
    # Register the mouse callback on this window. When 
    # the user clicks anywhere in the "Calibrate" window,
    # the function mouse_click_callback() is called (defined above)
    cv2.setMouseCallback("Calibrate", mouse_click_callback)
 
    # Wait until the user has selected 4 points
    while True:
        # If the user has selected all 4 points, exit loop.
        if (len(corner_point_list) >= 4):
            print("Got 4 points: "+str(corner_point_list))
            break
 
        # If the user hits a key, exit loop, otherwise remain.
        if (cv2.waitKey(10) >= 0):
            break;
 
    # Close the calibration window:
    cv2.destroyWindow("Calibrate")
 
    # If the user selected 4 points
    if (len(corner_point_list) >= 4):
        # Do calibration
 
        # src is a list of 4 points on the original image selected by the user
        # in the order [TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT]
        src = numpy.array(corner_point_list, numpy.float32)
 
        # dest is a list of where these 4 points should be located on the
        # rectangular board (in the same order):
        dest = numpy.array( [ (0, 0), (0, board_size[1]*dpi), (board_size[0]*dpi, 0), (board_size[0]*dpi, board_size[1]*dpi) ], numpy.float32)
 
        # Calculate the perspective transform matrix
        trans = cv2.getPerspectiveTransform(src, dest)
 
        # If we were given a calibration filename, save this matrix to a file
        if calib_file:
            numpy.savetxt(calib_file, trans)
        return trans
    else:
        return None

# Apply perspective transformation
def transform_image(image, matrix, board_size, dpi):
    width = int(board_size[0] * dpi)
    height = int(board_size[1] * dpi)

    return cv2.warpPerspective(image, matrix, (width, height))


# Calculate ring position relative to paper [cm]
def ring_position(rings, dpi):
    positions = []

    for cx, cy, radius in rings:
        x = cx / dpi
        y = cy / dpi

        positions.append((x, y))

    return positions


# Main program
if __name__ == "__main__":

    camera_id = 1
    dev = open_camera(camera_id)

    if not dev.isOpened():
        print("Cannot open camera")
        exit()

    # A4 paper dimensions [cm]
    board_size = (29.7, 21.0)
    dpi = 20

    # Calibrate perspective transformation
    matrix = get_transform_matrix(dev, board_size, dpi)

    paused = False

    try:
        while True:

            if not paused:
                img = get_frame(dev)

                if img is None:
                    print("Cannot read image")
                    break

                # Image processing - Original
                gray = grayscale(img)
                threshold = threshold_image(gray)
                contours = find_contours(threshold)

                # Find contours and rings - Original
                rings = find_ring(contours)
                centers = find_centers(contours)

                # Draw results - Original
                contour_img = draw_contours(img, contours)
                ring_img = draw_rings(img, rings)

                # Perspective transformation
                if matrix is not None:
                    warped = transform_image(img, matrix, board_size, dpi)
                else:
                    warped = img

                # Image processing - Warped
                warped_gray = grayscale(warped)
                warped_threshold = threshold_image(warped_gray)
                warped_contours = find_contours(warped_threshold)

                # Find contours and rings - Warped
                warped_rings = find_ring(warped_contours)
                warped_centers = find_centers(warped_contours)

                # Draw results - Warped
                warped_contour_img = draw_contours(warped, warped_contours)
                warped_ring_img = draw_rings(warped, warped_rings)

                # Calculate ring position on paper
                if matrix is not None:
                    positions = ring_position(warped_rings, dpi)

                    for x, y in positions:
                        
                        # Print position in terminal
                        print(f"Ring center: x = {x:.2f} cm, y = {y:.2f} cm")
                        
                        # Display position on image
                        cv2.putText(
                            warped_ring_img,
                            f"({x:.1f}, {y:.1f}) cm",
                            (int(x * dpi) + 10, int(y * dpi) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )

                # Display images - Original
                # cv2.imshow("Original", resize_image(img))
                # cv2.imshow("Threshold", resize_image(threshold))
                # cv2.imshow("Contours", resize_image(contour_img))
                # cv2.imshow("Circles and Centers", resize_image(ring_img))

                # Display images - Warped
                cv2.imshow("Warped", warped)
                cv2.imshow("Warped Threshold", warped_threshold)
                cv2.imshow("Warped Contours", warped_contour_img)
                cv2.imshow("Warped Circles and Centers", warped_ring_img)

                # Arrange windows in a 2x2 grid
                cv2.moveWindow("Warped", 0, 30)
                cv2.moveWindow("Warped Threshold", 800, 30)
                cv2.moveWindow("Warped Contours", 0, 400)
                cv2.moveWindow("Warped Circles and Centers", 800, 400)

            # Keyboard controls
            key = cv2.waitKey(30) & 0xFF

            if key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")

            elif key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program stopped")

    finally:
        cleanup(dev)
