import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("DefectVisionAI"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from app import app
    print("FastAPI app imported successfully!")
    print(f"Routes defined: {len(app.routes)}")
    for r in app.routes:
        if hasattr(r, 'path'):
            print(f"  {r.methods if hasattr(r, 'methods') else 'WS'} {r.path}")
except Exception as e:
    import traceback
    print(f"Failed to import app: {e}")
    traceback.print_exc()
