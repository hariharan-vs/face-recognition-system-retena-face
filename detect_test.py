"""Utility script to test face detection independently of Streamlit.

Usage:
    python detect_test.py --image path/to/photo.jpg
    python detect_test.py            # runs webcam

This helps verify that `face_utils.detect_and_align_faces` is operating correctly.
"""
import argparse
import cv2
from face_utils import detect_and_align_faces


def draw_boxes(img, faces):
    for f in faces:
        x1, y1, x2, y2 = f['bbox']
        score = f.get('score', 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image file (optional).')
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Unable to read image: {args.image}")
            return
        faces = detect_and_align_faces(img)
        print("Detected faces:", faces)
        out = draw_boxes(img.copy(), faces)
        cv2.imshow('detection', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not available")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = detect_and_align_faces(frame)
            out = draw_boxes(frame.copy(), faces)
            cv2.imshow('webcam detection', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
