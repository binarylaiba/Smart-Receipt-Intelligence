"""Sample script demonstrating how to make an API request to the AI Receipt Analyser API."""

import sys
import requests

API_URL = "http://127.0.0.1:8000/api/analyze-receipt"


def send_receipt_for_analysis(image_path: str):
    """Send an image file to the API endpoint and print the JSON response."""
    print(f"Sending '{image_path}' to {API_URL}...")

    with open(image_path, "rb") as img_file:
        files = {"file": (image_path, img_file, "image/jpeg")}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        data = response.json()
        print("\n--- Success! Receipt Analysis Received ---")
        print(f"Merchant: {data.get('merchant')}")
        print(f"Date:     {data.get('date')}")
        print(f"Total:    ${data.get('total'):.2f}")
        print(f"Items Count: {len(data.get('items', []))}")
        print("\nFull JSON Response:")
        import json
        print(json.dumps(data, indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        send_receipt_for_analysis(sys.argv[1])
    else:
        print("Usage: py test_client.py <path_to_receipt_image.jpg>")
