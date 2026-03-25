import cv2
import numpy as np
import streamlit as st
try:
    from retinaface import RetinaFace
    _HAVE_RETINAFACE = True
except Exception:
    _HAVE_RETINAFACE = False
try:
    from keras_facenet import FaceNet
    _HAVE_FACENET = True
except Exception:
    _HAVE_FACENET = False


class _SimpleEmbedder:
    """Lightweight fallback embedder when `keras_facenet` is unavailable.

    This generates a normalized flattened image vector (not a true face embedding),
    but lets the app run and test flows without the external dependency.
    """
    def embeddings(self, imgs):
        out = []
        for img in imgs:
            try:
                im = cv2.resize(img, (64, 64))
            except Exception:
                im = cv2.resize(img.astype(np.uint8), (64, 64))
            vec = im.astype(np.float32).flatten()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out.append(vec)
        return np.array(out)


@st.cache_resource(show_spinner=False)
def load_embedder():
    # Cache the FaceNet model to avoid reloading on every Streamlit render
    if _HAVE_FACENET:
        return FaceNet()
    # return a lightweight fallback that still exposes `embeddings()`
    return _SimpleEmbedder()

def get_cosine_similarity(emb1, emb2):
    """
    Cosine similarity mimic for InsightFace/ArcFace matching logic.
    """
    emb1_norm = np.linalg.norm(emb1)
    emb2_norm = np.linalg.norm(emb2)
    if emb1_norm == 0 or emb2_norm == 0:
        return 0.0
    return np.dot(emb1, emb2) / (emb1_norm * emb2_norm)

def detect_and_align_faces(img_bgr):
    """
    img_bgr: BGR numpy image array.
    Returns: list of dicts with 'bbox', 'landmarks', 'face_img' (cropped), 'score'
    """
    results = []

    if _HAVE_RETINAFACE:
        try:
            # RetinaFace expects RGB input; OpenCV provides BGR
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = RetinaFace.detect_faces(img_rgb)
        except Exception:
            faces = {}

        if isinstance(faces, dict):
            for key, face_info in faces.items():
                score = face_info.get('score', 0)
                # use a more permissive threshold to catch faces that are partially occluded
                if score < 0.5:
                    continue
                facial_area = face_info.get('facial_area', None)
                landmarks = face_info.get('landmarks', None)
                if facial_area is None:
                    continue

                x1, y1, x2, y2 = facial_area
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img_bgr.shape[1], int(x2))
                y2 = min(img_bgr.shape[0], int(y2))

                face_img = img_bgr[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                results.append({
                    'bbox': (x1, y1, x2, y2),
                    'landmarks': landmarks,
                    'face_img': face_img,
                    'score': score
                })
        # if retinaface gave us any faces, return them immediately
        if results:
            return results
        # otherwise, try secondary detector (face_recognition) if available
        try:
            import face_recognition
        except ImportError:
            return results
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img_rgb, model='hog')
        for (top, right, bottom, left) in boxes:
            x1, y1, x2, y2 = left, top, right, bottom
            face_img = img_bgr[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            results.append({
                'bbox': (x1, y1, x2, y2),
                'landmarks': None,
                'face_img': face_img,
                'score': 0.5
            })
        return results

    # Fallback: use OpenCV Haar Cascade detector (works without retinaface)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in detected:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        face_img = img_bgr[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        results.append({
            'bbox': (x1, y1, x2, y2),
            'landmarks': None,
            'face_img': face_img,
            'score': 1.0
        })

    return results

def extract_embedding(face_img_bgr, embedder):
    """
    Extracts embedding using FaceNet or fallback.
    Applies an elliptical mask to remove the background from the square crop.
    """
    # 1. Resize face_img to 160x160 as expected by FaceNet
    face_img_resized = cv2.resize(face_img_bgr, (160, 160))
    
    # 2. Create an elliptical mask to remove background corners
    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.ellipse(mask, (80, 80), (70, 80), 0, 0, 360, 255, -1) # Ellipse centered at (80,80) with axes 70x80
    
    # 3. Apply the mask (background becomes black)
    masked_face = cv2.bitwise_and(face_img_resized, face_img_resized, mask=mask)
    
    # 4. Convert BGR to RGB for FaceNet (keras-facenet expects RGB generally)
    face_img_rgb = cv2.cvtColor(masked_face, cv2.COLOR_BGR2RGB)
    
    # FaceNet expects a list/array of images [batch, h, w, c]
    embeddings = embedder.embeddings([face_img_rgb])
    if len(embeddings) > 0:
        emb = embeddings[0]
            
        # Normalize the embedding as requested
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    return None

def match_face(target_embedding, database_students, threshold=0.6):
    """
    target_embedding: query embedding
    database_students: list of dicts from get_all_students()
    threshold: InsightFace logic threshold (configurable, >= 0.6 standard)
    Returns: best_match_student_dict, max_confidence
    """
    best_match = None
    max_confidence = 0.0
    
    for student in database_students:
        db_emb = student['face_embedding']
        if db_emb is None:
            continue
            
        similarity = get_cosine_similarity(target_embedding, db_emb)
        if similarity > max_confidence:
            max_confidence = similarity
            best_match = student
            
    if max_confidence >= threshold:
        return best_match, max_confidence
    else:
        return None, max_confidence
