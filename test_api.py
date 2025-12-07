# %%
from dotenv import load_dotenv
load_dotenv()

import requests
import json
import etl.extract as extract

# Remplacez par votre username HuggingFace
USERNAME = "synaxio"
API_URL = f"https://{USERNAME}-api-prediction.hf.space"

def test_health():
    """Test du endpoint de santé."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"✓ Health check: {response.status_code}")
        print(f"  Response: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_prediction():
    """Test d'une prédiction."""
    # Ajustez les données selon votre modèle de fraud detection
    test_data = {
        "transaction_amount": 150.50,
        "merchant_category": "online_retail",
        "time_of_day": 14,
        "customer_age": 35,
        # Ajoutez d'autres features selon votre modèle
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n✓ Prediction request: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Prediction: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

def test_batch_predictions():
    """Test de prédictions en batch."""
    batch_data = {
        "transactions": [
            {"transaction_amount": 50.0, "merchant_category": "grocery"},
            {"transaction_amount": 1500.0, "merchant_category": "electronics"},
            {"transaction_amount": 25.0, "merchant_category": "coffee_shop"}
        ]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n✓ Batch prediction: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"  Results: {json.dumps(results, indent=2)}")
            return True
        else:
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        return False
# %%

if __name__ == "__main__":
    print(f"Testing API at: {API_URL}\n")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n[1] Testing health endpoint...")
    test_health()
    
    # Test 2: Single prediction
    print("\n[2] Testing single prediction...")
    test_prediction()
    
    # Test 3: Batch predictions (si disponible)
    print("\n[3] Testing batch predictions...")
    test_batch_predictions()
    
    print("\n" + "=" * 50)
    print("Tests completed!")