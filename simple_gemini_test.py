import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found in environment")
    exit(1)

print(f"🔑 Key found: {api_key[:5]}...****")
print(f"📦 Library Version: {genai.__version__}")
genai.configure(api_key=api_key)

print("\n🚀 Testing Gemini Connectivity...")

models_to_test = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

for model_name in models_to_test:
    print(f"\nTrying model: {model_name}...")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, can you hear me?")
        print(f"   ✅ SUCCESS! Response: {response.text.strip()}")
        print("   (Your API key and connection are working perfectly)")
        break
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

print("\n------------------------------------------------")
print("Diagnosis:")
print("If you saw at least one ✅ SUCCESS, the problem is in the QRail code.")
print("If you only saw ❌ FAILED, the problem is your API Key, Network, or Region.")
