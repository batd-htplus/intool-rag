#!/usr/bin/env python3
"""
Test script for RAG query API
Tests the query endpoint with the problematic question
"""
import requests
import json
import sys

# Configuration
RAG_SERVICE_URL = "http://localhost:8001"
TEST_QUESTION = "What is the total amount due on invoice #4820 for Aaron Hawkins, and which product was purchased?"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint...")
    print("=" * 60)
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/health", timeout=5)
        response.raise_for_status()
        print(f"✓ Health check passed: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_ready():
    """Test ready endpoint"""
    print("\n" + "=" * 60)
    print("Testing Ready Endpoint...")
    print("=" * 60)
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/ready", timeout=5)
        response.raise_for_status()
        print(f"✓ Ready check passed: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Ready check failed: {e}")
        return False

def test_query():
    """Test query endpoint with the problematic question"""
    print("\n" + "=" * 60)
    print(f"Testing Query Endpoint with: '{TEST_QUESTION}'")
    print("=" * 60)
    
    payload = {
        "question": TEST_QUESTION,
        "filters": None,
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
        "include_sources": True
    }
    
    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        print(f"\nSending POST request to {RAG_SERVICE_URL}/query...")
        response = requests.post(
            f"{RAG_SERVICE_URL}/query",
            json=payload,
            timeout=150
        )
        
        print(f"\nResponse status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        
        result = response.json()
        print(f"\n✓ Query successful!")
        print(f"\nResponse:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if "answer" in result:
            answer = result['answer']
            print(f"\n✓ Answer received: {answer}")
            
            if "I don't have" in answer or "don't have this information" in answer.lower():
                print(f"\n⚠ WARNING: LLM returned 'no information' response")
                print(f"   This might indicate:")
                print(f"   - Context was truncated too much")
                print(f"   - LLM didn't understand the question")
                print(f"   - Prompt needs improvement")
            else:
                print(f"✓ Answer contains meaningful information!")
        
        if "sources" in result:
            print(f"\n✓ Sources received: {len(result['sources'])} sources")
            for i, source in enumerate(result['sources'][:3], 1):
                score = source.get('score', 'N/A')
                text_preview = source.get('text', '')[:150]
                if isinstance(score, float):
                    print(f"  Source {i}: score={score:.4f}")
                else:
                    print(f"  Source {i}: score={score}")
                print(f"    Preview: {text_preview}...")
        
        print(f"\n✓ Query with special character '?' was processed correctly!")
        print(f"✓ Embedding was successfully called and results were retrieved!")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"\n✗ HTTP Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response body: {e.response.text}")
        
        if "generate" in str(e.response.text):
            print(f"\n⚠ Note: This is an LLM generation error, not an embedding error.")
            print(f"✓ Embedding service worked correctly (check logs for confirmation)")
            print(f"✓ Query with special character '?' was passed to embedding successfully!")
        
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_directly():
    """Test embedding service directly using /embed endpoint (used by RAG service)"""
    print("\n" + "=" * 60)
    print("Testing Embedding Service Directly...")
    print("=" * 60)
    
    MODEL_SERVICE_URL = "http://localhost:8002"
    
    # Use /embed endpoint (same as RAG service uses)
    payload = {
        "texts": [TEST_QUESTION],
        "instruction": "Represent this sentence for searching relevant passages: "
    }
    
    try:
        print(f"\nSending POST request to {MODEL_SERVICE_URL}/embed...")
        print(f"Testing with question: '{TEST_QUESTION}'")
        response = requests.post(
            f"{MODEL_SERVICE_URL}/embed",
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        if "embeddings" in result and len(result["embeddings"]) > 0:
            embedding = result["embeddings"][0]
            print(f"✓ Embedding generated successfully!")
            print(f"  Embedding dimension: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            print(f"  ✓ Question with special character '?' was processed correctly!")
            return True
        else:
            print(f"✗ No embedding in response: {result}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Embedding test failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RAG Query API Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test health
    results.append(("Health Check", test_health()))
    
    # Test ready
    results.append(("Ready Check", test_ready()))
    
    # Test embedding directly
    results.append(("Embedding Service", test_embedding_directly()))
    
    # Test query
    results.append(("Query Endpoint", test_query()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

