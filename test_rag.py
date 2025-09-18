import requests
import time

# Testar health
print("Testando API...")
response = requests.get("http://localhost:8000/api/v1/health")
print(f"Health: {response.json()}")

# Upload documento teste
print("\nUpload de documento teste...")
with open("data/raw/exemplo.pdf", "rb") as f:
    files = {"file": ("exemplo.pdf", f, "application/pdf")}
    response = requests.post("http://localhost:8000/api/v1/documents", files=files)
    print(f"Upload: {response.json()}")

time.sleep(2)

# Fazer query
print("\nFazendo query...")
query = {"question": "Qual é a política de privacidade?", "top_k": 3}
response = requests.post("http://localhost:8000/api/v1/query", json=query)
print(f"Resposta: {response.json()}")