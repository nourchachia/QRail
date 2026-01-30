
import sys
import time
import numpy as np
from fastembed import TextEmbedding

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_fastembed_generation():
    print("=" * 60)
    print("⚡ FastEmbed Standalone Verification")
    print("=" * 60)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"1. Initializing model: {model_name}...")
    
    try:
        start_time = time.time()
        embedding_model = TextEmbedding(model_name=model_name)
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.4f} seconds")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False

    documents = [
        "Signal failure at Central Station causing delays.",
        "The train is running on time.",
        "Passenger emergency at platform 4."
    ]
    
    print(f"\n2. Generatng embeddings for {len(documents)} documents...")
    try:
        start_time = time.time()
        # TextEmbedding.embed returns a generator
        embeddings_generator = embedding_model.embed(documents)
        embeddings = list(embeddings_generator)
        gen_time = time.time() - start_time
        
        print(f"   ✅ Embeddings generated in {gen_time:.4f} seconds")
        
        # Verify shape
        expected_dim = 384
        print(f"\n3. Verifying output dimensions (Expected: {expected_dim})...")
        
        for i, emb in enumerate(embeddings):
            # emb is a numpy array
            shape = emb.shape
            print(f"   Document {i+1}: Shape {shape}")
            
            if shape[0] != expected_dim:
                print(f"   ❌ Dimension mismatch! Expected {expected_dim}, got {shape[0]}")
                return False
                
        print("\n4. Sanity Check (Cosine Similarity)...")
        # Simple dot product check (normalized vectors)
        vec1 = embeddings[0] # Signal failure
        vec2 = embeddings[1] # On time
        vec3 = embeddings[2] # Passenger emergency
        
        # Calculate similarity
        sim_1_2 = np.dot(vec1, vec2)
        sim_1_3 = np.dot(vec1, vec3)
        
        print(f"   Similarity (Signal vs OnTime): {sim_1_2:.4f}")
        print(f"   Similarity (Signal vs Passenger): {sim_1_3:.4f}")
        
        print("\n✅ FastEmbed is working correctly!")
        return True

    except Exception as e:
        print(f"   ❌ specific Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fastembed_generation()
    sys.exit(0 if success else 1)
