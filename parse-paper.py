# This script is used to parse and extract information from research papers.

## Input: CSV file containing paper metadata
## Output: CSV file with labeled papers 

import json
import pandas as pd
from rich.console import Console
from src.openai_api import get_openai_client, get_openai_response
import os

console = Console()

local_config_file = "config.json"

file_path = console.input("Enter the path to the CSV file containing paper metadata: ").strip()
if not file_path:
    console.print("❌ Invalid file path. Please provide a valid CSV file path.")
    exit()

paper_df = pd.read_csv(file_path)
if paper_df.empty:
    console.print("❌ The provided CSV file is empty or not formatted correctly.")
    exit()

console.print(f"📄 Loaded {len(paper_df)} papers from {file_path}.")

api_key = ''
base_url = ''

try:
    with open(local_config_file, 'r') as f:
        config = json.load(f)
        api_key = config.get('openai_api_key', '')
        base_url = config.get('openai_base_url', '')
except FileNotFoundError:
    console.print(f"⚠️ Configuration file '{local_config_file}' not found. Please provide your OpenAI API key and base URL.")

if not api_key:
    api_key = console.input("Enter your OpenAI API key: ").strip()
    
if not base_url:
  base_url = console.input("Enter the OpenAI API base URL (default: https://api.openai.com/v1): ").strip() or "https://api.openai.com/v1"

console.print(f"🌐 Using OpenAI API base URL: {base_url}")

# Store the API key and base URL to config file if they were entered manually
if api_key or base_url:
  config_dir = os.path.expanduser("~/.paper-labeler")
  config_file_path = os.path.expanduser(local_config_file)
  
  # Create directory if it doesn't exist
  os.makedirs(config_dir, exist_ok=True)
  
  # Load existing config or create new one
  config = {}
  if os.path.exists(config_file_path):
    try:
      with open(config_file_path, 'r') as f:
        config = json.load(f)
    except json.JSONDecodeError:
      config = {}
  
  # Update config with new values
  config['openai_api_key'] = api_key
  config['openai_base_url'] = base_url
  
  # Save config
  with open(config_file_path, 'w') as f:
    json.dump(config, f, indent=2)
  
  console.print(f"💾 Configuration saved to {config_file_path}")

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
    