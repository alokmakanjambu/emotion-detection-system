"""
API Test Script for Emotion Detection API.
Run this script to test all endpoints.
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def print_response(title, response):
    """Pretty print response."""
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)

def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print_response("Health Check", response)
    return response.status_code == 200

def test_emotions():
    """Test emotions list endpoint."""
    response = requests.get(f"{BASE_URL}/api/v1/emotions")
    print_response("Supported Emotions", response)
    return response.status_code == 200

def test_predict_single(text):
    """Test single prediction."""
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json={"text": text}
    )
    print_response(f"Predict: '{text[:50]}...'", response)
    return response

def test_predict_batch(texts):
    """Test batch prediction."""
    response = requests.post(
        f"{BASE_URL}/api/v1/predict/batch",
        json={"texts": texts}
    )
    print_response(f"Batch Predict ({len(texts)} texts)", response)
    return response

def test_validation_error():
    """Test validation error with empty text."""
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json={"text": ""}
    )
    print_response("Validation Error (empty text)", response)
    return response.status_code == 422

def main():
    print("\n" + "="*60)
    print("🎭 EMOTION DETECTION API TEST SUITE")
    print("="*60)
    
    # Test health
    print("\n🔥 Testing Health Endpoint...")
    test_health()
    
    # Test emotions list
    print("\n🔥 Testing Emotions List...")
    test_emotions()
    
    # Test predictions for all emotions
    print("\n🔥 Testing Single Predictions...")
    test_cases = [
        ("JOY", "I'm so happy and excited about this wonderful news!"),
        ("SADNESS", "I feel so sad and lonely, nobody understands me"),
        ("ANGER", "This is absolutely terrible, I hate everything about this"),
        ("FEAR", "I'm really scared about what might happen next"),
        ("SURPRISE", "Oh my God! I can't believe this just happened!"),
        ("LOVE", "I love my family so much, they mean everything to me"),
    ]
    
    results = []
    for expected, text in test_cases:
        response = test_predict_single(text)
        if response.status_code == 200:
            result = response.json()
            actual = result['emotion'].upper()
            confidence = result['confidence']
            status = "✅" if actual == expected else "⚠️"
            results.append((expected, actual, confidence, status))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 PREDICTION SUMMARY")
    print("="*60)
    print(f"{'Expected':<12} {'Actual':<12} {'Confidence':<12} {'Status'}")
    print("-"*48)
    for expected, actual, confidence, status in results:
        print(f"{expected:<12} {actual:<12} {confidence:.2%}        {status}")
    
    # Test batch
    print("\n🔥 Testing Batch Prediction...")
    test_predict_batch([
        "I am happy",
        "I am sad", 
        "I am angry",
        "I am scared",
        "I am surprised",
        "I love you"
    ])
    
    # Test validation
    print("\n🔥 Testing Validation Error...")
    test_validation_error()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
