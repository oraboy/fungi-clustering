from google.cloud import vision

def test_vision_api():
    try:
        # Create a client
        client = vision.ImageAnnotatorClient()
        print("Successfully created Vision API client!")
        print("Credentials are working correctly.")
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_vision_api()
