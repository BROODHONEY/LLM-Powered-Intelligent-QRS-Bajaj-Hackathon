import os
import shutil
import subprocess
import platform

def get_home():
    return os.path.expanduser("~")

def delete_folder(path):
    if os.path.exists(path):
        print(f"Deleting: {path}")
        shutil.rmtree(path)
    else:
        print(f"Already clean: {path}")

def clear_huggingface_cache():
    path = os.path.join(get_home(), ".cache", "huggingface")
    delete_folder(path)

def clear_pytorch_cache():
    path = os.path.join(get_home(), ".cache", "torch")
    delete_folder(path)

def clear_tensorflow_cache():
    tf_paths = [
        os.path.join(get_home(), ".keras"),
        os.path.join(get_home(), ".cache", "tensorflow")
    ]
    for path in tf_paths:
        delete_folder(path)

def clear_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Ollama not found or not responding.")
            return
        lines = result.stdout.strip().splitlines()
        if len(lines) <= 1:
            print("No Ollama models installed.")
            return

        print("\nInstalled Ollama Models:")
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        for idx, model in enumerate(models):
            print(f"[{idx}] {model}")

        indices = input("\nEnter indices (comma-separated) of models to delete, or 'all': ")
        if indices.lower() == "all":
            for model in models:
                subprocess.run(["ollama", "remove", model])
        else:
            try:
                indices = list(map(int, indices.split(",")))
                for i in indices:
                    subprocess.run(["ollama", "remove", models[i]])
            except Exception as e:
                print("Invalid input:", e)
    except Exception as e:
        print("Error clearing Ollama models:", e)

def main():
    print("\n🧹 Model Cache Cleanup Utility 🧹\n")

    # Ollama
    choice = input("Do you want to remove Ollama models? (y/n): ").strip().lower()
    if choice == 'y':
        clear_ollama_models()

    # HuggingFace
    choice = input("Clear Hugging Face model cache? (y/n): ").strip().lower()
    if choice == 'y':
        clear_huggingface_cache()

    # PyTorch
    choice = input("Clear PyTorch model cache? (y/n): ").strip().lower()
    if choice == 'y':
        clear_pytorch_cache()

    # TensorFlow
    choice = input("Clear TensorFlow model cache? (y/n): ").strip().lower()
    if choice == 'y':
        clear_tensorflow_cache()

    print("\n✅ Cleanup complete. Check your disk space!\n")

if __name__ == "__main__":
    main()
