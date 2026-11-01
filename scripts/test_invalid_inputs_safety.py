"""
test_invalid_inputs_safety.py
Verifies error handling and safe failures on invalid inputs using urllib.
"""

import urllib.request
import urllib.error
import json
import uuid

BASE_URL = "http://127.0.0.1:8000"

def post_multipart(url, fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()
    
    for k, v in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{v}\r\n".encode("utf-8"))
        
    for k, (filename, content, mime) in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{k}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {mime}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")
        
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    
    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    return req

def test_invalid_category():
    print("\n--- Test 1: Invalid Category ---")
    req = post_multipart(
        f"{BASE_URL}/predict",
        fields={"category": "non_existent_category_xyz123"},
        files={"file": ("test.png", b"\x89PNG\r\n\x1a\nfakeimagecontent", "image/png")}
    )
    try:
        urllib.request.urlopen(req)
        print("FAIL: Expected error for invalid category!")
        return False
    except urllib.error.HTTPError as e:
        print(f"Status Code: {e.code}")
        err_msg = e.read().decode("utf-8")
        print(f"Response: {err_msg[:200]}")
        assert e.code in [400, 404, 422, 500], f"Unexpected code {e.code}"
        print("Test 1 Passed: Safe error returned for invalid category.")
        return True

def test_non_image_file():
    print("\n--- Test 2: Non-Image Text File ---")
    req = post_multipart(
        f"{BASE_URL}/predict",
        fields={"category": "bottle"},
        files={"file": ("document.txt", b"Hello World this is plain text not an image", "text/plain")}
    )
    try:
        urllib.request.urlopen(req)
        print("FAIL: Expected error for non-image file!")
        return False
    except urllib.error.HTTPError as e:
        print(f"Status Code: {e.code}")
        err_msg = e.read().decode("utf-8")
        print(f"Response: {err_msg[:200]}")
        assert e.code in [400, 422, 500], f"Unexpected code {e.code}"
        print("Test 2 Passed: Safe error returned for non-image file.")
        return True

def test_empty_file():
    print("\n--- Test 3: Zero-Byte Empty File ---")
    req = post_multipart(
        f"{BASE_URL}/predict",
        fields={"category": "bottle"},
        files={"file": ("empty.png", b"", "image/png")}
    )
    try:
        urllib.request.urlopen(req)
        print("FAIL: Expected error for empty file!")
        return False
    except urllib.error.HTTPError as e:
        print(f"Status Code: {e.code}")
        err_msg = e.read().decode("utf-8")
        print(f"Response: {err_msg[:200]}")
        assert e.code in [400, 422, 500], f"Unexpected code {e.code}"
        print("Test 3 Passed: Safe error returned for empty file.")
        return True

def test_invalid_batch_id():
    print("\n--- Test 4: Invalid Batch ID ---")
    try:
        urllib.request.urlopen(f"{BASE_URL}/batch/non_existent_batch_99999/status")
        print("FAIL: Expected 404 for invalid batch ID!")
        return False
    except urllib.error.HTTPError as e:
        print(f"Status Code: {e.code}")
        err_msg = e.read().decode("utf-8")
        print(f"Response: {err_msg[:200]}")
        assert e.code == 404, f"Expected 404, got {e.code}"
        print("Test 4 Passed: 404 returned for non-existent batch.")
        return True

if __name__ == "__main__":
    t1 = test_invalid_category()
    t2 = test_non_image_file()
    t3 = test_empty_file()
    t4 = test_invalid_batch_id()
    assert all([t1, t2, t3, t4]), "One or more tests failed"
    print("\n=== ALL INVALID INPUT SAFETY TESTS PASSED ===")
