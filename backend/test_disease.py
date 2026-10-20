import asyncio
import os
import sys

# Add current directory to path so services can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from services.disease_service import predict_disease

async def test_real_images():
    test_dir = r"D:\ALIP-TOTAL\ALIP\ml_models\real_field_images"
    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}")
        return

    print("="*70)
    print(" DISEASE MODULE - REAL FIELD IMAGE EVALUATION")
    print("="*70 + "\n")

    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    correct = 0
    total = len(files)

    for filename in sorted(files):
        filepath = os.path.join(test_dir, filename)
        with open(filepath, "rb") as f:
            image_bytes = f.read()
        
        # Test without location first to just get model and local DB response
        # Passing location="Pune" would trigger weather API which might slow it down or hit rate limits
        try:
            result = await predict_disease(image_bytes, location=None)
            
            if result.get("success"):
                pred = result.get("disease", "")
                conf = result.get("confidence", 0.0)
                sev = result.get("severity", "")
                treat = result.get("treatment", "")
                
                print(f"Image: {filename}")
                print(f"Prediction:   {pred} (Confidence: {conf}%)")
                print(f"Severity:     {sev}")
                print(f"Action:       {treat}")
                print("-" * 60)
            else:
                print(f"Image: {filename} - FAILED: {result.get('error')}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    asyncio.run(test_real_images())
