import urllib.request
import json

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
with open('DefectVisionAI/outputs/similarity_search_samples/bottle_similarity_demo.png', 'rb') as f:
    img = f.read()

body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="b.png"\r\n'
    f'Content-Type: image/png\r\n\r\n'
).encode('latin-1') + img + (
    f'\r\n--{boundary}\r\n'
    f'Content-Disposition: form-data; name="category"\r\n\r\n'
    f'bottle\r\n'
    f'--{boundary}--\r\n'
).encode('latin-1')

req = urllib.request.Request(
    'http://127.0.0.1:5173/predict',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

res = json.loads(urllib.request.urlopen(req).read())
print('PREDICT THROUGH VITE PROXY SUCCESS:')
print('  Predicted label:', res['predicted_label'])
print('  Anomaly score:  ', res['anomaly_score'])
print('  Threshold:      ', res['optimal_threshold'])
print('  Severity:       ', res['severity_level'], res['severity_score'])
print('  Heatmap length: ', len(res['heatmap']))
