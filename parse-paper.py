# This script is used to parse and extract information from research papers.

## Input: CSV file containing paper metadata
## Output: CSV file with labeled papers 

from email.mime import base
import pandas as pd
from rich.console import Console
from src.openai_api import get_openai_client, get_openai_response

console = Console()

file_path = console.input("Enter the path to the CSV file containing paper metadata: ").strip()
if not file_path:
    console.print("❌ Invalid file path. Please provide a valid CSV file path.")
    exit()

paper_df = pd.read_csv(file_path)
if paper_df.empty:
    console.print("❌ The provided CSV file is empty or not formatted correctly.")
    exit()

console.print(f"📄 Loaded {len(paper_df)} papers from {file_path}.")

api_key = console.input("Enter your OpenAI API key: ").strip()
if not api_key:
    console.print("❌ API key cannot be empty. Please provide a valid OpenAI API key.")
    exit()

base_url = console.input("Enter the OpenAI API base URL (default: https://api.openai.com/v1): ").strip() or "https://api.openai.com/v1"
console.print(f"🌐 Using OpenAI API base URL: {base_url}")

openai_client = get_openai_client(api_key, base_url)
console.print("🔍 Starting analysis of paper abstracts...")

# Currently, we only test first paper and print its result
paper = paper_df.iloc[0]
abstract = paper.get('Abstract Note', '')
if not abstract:
    console.print("❌ No abstract found for the first paper. Please ensure the CSV file contains an 'Abstract Note' column.")
    exit()

analyze_result = get_openai_response(openai_client, abstract)
if not analyze_result:
    console.print("❌ Failed to get a response from OpenAI API. Please check your API key and network connection.")
    exit()

console.print("✅ Analysis completed successfully!")

console.print("📊 Analysis Result:")
console.print(analyze_result)
    