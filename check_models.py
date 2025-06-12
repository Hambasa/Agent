import ollama

try:
    client = ollama.Client()
    models = client.list()
    print("Available models:")
    for m in models.get('models', []):
        name = m.get('name', 'unknown')
        size = m.get('size', 'unknown')
        print(f"- {name} (size: {size})")
except Exception as e:
    print(f"Error: {e}")
