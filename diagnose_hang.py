import os
import time
import sys

def test_import(module_name):
    print(f"Testing import of {module_name}...", flush=True)
    start = time.time()
    try:
        __import__(module_name)
        print(f"  SUCCESS: {module_name} imported in {time.time() - start:.2f}s", flush=True)
    except Exception as e:
        print(f"  FAILED: {module_name} failed: {e}", flush=True)

# Force settings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Starting diagnostics...", flush=True)
test_import("cv2")
test_import("numpy")
test_import("streamlit")
test_import("tensorflow")
test_import("keras_facenet")
test_import("retinaface")
print("Diagnostics complete.", flush=True)
