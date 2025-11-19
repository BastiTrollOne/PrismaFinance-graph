import requests

url = "http://127.0.0.1:8000/upload?ocr=auto&emit_multi=false"
file_path = r"C:\Users\basti\OneDrive\Desktop\clusternodo.pdf"

with open(file_path, "rb") as f:
    files = {"file": (file_path.split("\\")[-1], f, "application/pdf")}
    resp = requests.post(url, files=files)

print("Status:", resp.status_code)
print("Body:", resp.text[:500])
