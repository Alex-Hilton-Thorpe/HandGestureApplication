import cv2

def startCamera():
    return cv2.VideoCapture(0)

if __name__ == "__main__":
    capture = startCamera();
    while(1):
        # Capture frame-by-frame
        # ret is a boolean indicating successful frame capture
        # frame is the captured image
        ret, frame = capture.read();
        if not ret:
            print("Failed to capture frames")
            break
        else:
            #display the frames
            cv2.imshow('Hand Gesture Frame', frame);
            # Need way to shut it off - "q" key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    # When everything done, release the capture and destroy the window
    capture.release();
    cv2.destroyAllWindows();


