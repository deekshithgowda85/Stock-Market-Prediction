import requests
import json

# Test 7 days
r1 = requests.post('http://localhost:8000/api/v1/predict-lightgbm', 
                   json={'symbol': 'RELIANCE', 'days': 7})
data1 = r1.json()

# Test 60 days
r2 = requests.post('http://localhost:8000/api/v1/predict-lightgbm', 
                   json={'symbol': 'RELIANCE', 'days': 60})
data2 = r2.json()

print("=" * 70)
print("7 DAYS PREDICTION:")
print("=" * 70)
print(f"Current Price: ₹{data1['current_price']:.2f}")
print(f"First 3 predictions: {[round(p, 2) for p in data1['predictions'][:3]]}")
print(f"Last 3 predictions: {[round(p, 2) for p in data1['predictions'][-3:]]}")
print(f"Final Price: ₹{data1['predictions'][-1]:.2f}")
print(f"Predicted Change: {data1['predicted_change']:.2f}%")

print("\n" + "=" * 70)
print("60 DAYS PREDICTION:")
print("=" * 70)
print(f"Current Price: ₹{data2['current_price']:.2f}")
print(f"First 3 predictions: {[round(p, 2) for p in data2['predictions'][:3]]}")
print(f"Last 3 predictions: {[round(p, 2) for p in data2['predictions'][-3:]]}")
print(f"Final Price: ₹{data2['predictions'][-1]:.2f}")
print(f"Predicted Change: {data2['predicted_change']:.2f}%")

print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)
if data1['predictions'][-1] == data2['predictions'][-1]:
    print("❌ SAME PREDICTIONS - Model is NOT working properly!")
else:
    diff = abs(data1['predictions'][-1] - data2['predictions'][-1])
    print(f"✅ DIFFERENT PREDICTIONS - Model is working!")
    print(f"   Difference: ₹{diff:.2f}")
    print(f"   7-day final: ₹{data1['predictions'][-1]:.2f}")
    print(f"   60-day final: ₹{data2['predictions'][-1]:.2f}")
