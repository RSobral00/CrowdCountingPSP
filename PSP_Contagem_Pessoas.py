import cv2
import platform
import time
import os
import numpy as np
from screeninfo import get_monitors
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import datetime

def euclidean_distance_loss(y_true, y_pred):
    return K.square(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))

def maximize_window(window_name):
    if platform.system() == 'Windows':
        import ctypes
        user32 = ctypes.windll.user32
        user32.ShowWindow(user32.GetForegroundWindow(), 3)
    else:
        monitor = get_monitors()[0]
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(window_name, monitor.width, monitor.height)

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def save_density_map(density_map, original_frame, total_people, folder_path):
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Get current date and time
    current_time = datetime.datetime.now()
    date_time_str = current_time.strftime("%H-%M-%S-%d_%Y-%m")

    # Construct file name
    file_name = f"DM_{date_time_str}.jpg"
    file_path = os.path.join(folder_path, file_name)

    # Save the overlayed frame with density map
    cv2.imwrite(file_path, original_frame)

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the model
    model_filename = "PSP_DC_Soft_CSRNETP_FT16"  
    model_path = os.path.join(script_dir, model_filename)
    loaded_model = tf.keras.models.load_model(model_path, compile=True, custom_objects={"euclidean_distance_loss": euclidean_distance_loss})

    available_cameras = list_available_cameras()
    if not available_cameras:
        print("No cameras found.")
        return
    else:
        print(f"Available cameras: {available_cameras}")

    current_camera_index = 0
    cap = cv2.VideoCapture(available_cameras[current_camera_index], cv2.CAP_DSHOW)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    window_name = 'PSP Contagem de Pessoas'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    time.sleep(1)

    # Maximize the window
    maximize_window(window_name)

    prediction_mode = False
    density_map = None
    total_people = 0

    while True:
        # Capture frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break
        print(f"Original camera feed resolution: {frame.shape[1]}x{frame.shape[0]}")
        # Resize frame to 1080x1920
        frame = cv2.resize(frame, (1920, 1080))

        if not prediction_mode:
            # Show the frame
            cv2.imshow(window_name, frame)
        else:
            if density_map is None:
                # Preprocess frame for model prediction
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float16) / 255
                img = np.expand_dims(img, axis=0)

                # Predict
                density_map = loaded_model.predict(img)[0]

                # Apply threshold

                # Calculate the prediction sum
                density_map *= 0.042
                total_people = int(np.sum(density_map))

                threshold = 2e-4
                density_map[density_map < threshold] = 0

                # Overlay the density map on the original frame
                density_map_normalized = cv2.normalize(density_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                density_map_colored = cv2.applyColorMap(density_map_normalized, cv2.COLORMAP_JET)
                density_map_resized = cv2.resize(density_map_colored, (frame.shape[1], frame.shape[0]))
                overlayed_frame = cv2.addWeighted(frame, 1, density_map_resized, 0.5, 0)

                # Display the overlayed frame with total people count
                cv2.putText(overlayed_frame, f"Total People: {total_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, overlayed_frame)

                # Save the overlayed frame with density map
                save_density_map(density_map, overlayed_frame, total_people, "Images")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == 9:  # Tab key
            # Change to the next camera
            current_camera_index = (current_camera_index + 1) % len(available_cameras)
            cap.release()
            cap = cv2.VideoCapture(available_cameras[current_camera_index], cv2.CAP_DSHOW)
            width = 1920
            height = 1080
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


            if not cap.isOpened():
                print(f"Error: Could not open camera {available_cameras[current_camera_index]}.")
        elif key == 13:  # Enter key
            prediction_mode = not prediction_mode
            if not prediction_mode:
                density_map = None  # Reset density map when exiting prediction mode
        elif key == ord('g'): 
            # Save the overlayed frame with density map
            save_density_map(density_map, overlayed_frame, total_people, "Images")

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
