import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    api_key = input("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Save to .env file for future use
    with open(".env", "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}")

# Run the application
if __name__ == "__main__":
    print("Starting Legal AI Assistant for SMEs...")
    print("Access the web interface at http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)