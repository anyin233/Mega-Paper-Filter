from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
import json
import re
from loguru import logger

def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction from LLM responses that may contain extra text or formatting.
    
    This function handles various response formats:
    - Pure JSON
    - JSON wrapped in markdown code blocks
    - JSON with extra text before/after
    - JSON with comments or explanations
    - Malformed JSON that can be corrected
    
    :param response_text: Raw response text from LLM
    :return: Parsed JSON dictionary or None if extraction fails
    """
    if not response_text or not response_text.strip():
        logger.warning("Empty response text provided for JSON extraction")
        return None
    
    original_text = response_text.strip()
    logger.debug(f"Extracting JSON from response: {original_text[:200]}...")
    
    # Strategy 1: Try to parse as direct JSON
    try:
        result = json.loads(original_text)
        logger.debug("Successfully parsed as direct JSON")
        return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from markdown code blocks
    # Look for ```json ... ``` or ``` ... ```
    code_block_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'`(.*?)`'
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, original_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match.strip())
                logger.debug(f"Successfully extracted JSON from code block with pattern: {pattern}")
                return result
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Find JSON-like structures using braces
    # Look for content between { and }
    brace_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces
        r'\{[^}]+\}'  # Simple braces
    ]
    
    for pattern in brace_patterns:
        matches = re.findall(pattern, original_text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match.strip())
                logger.debug(f"Successfully extracted JSON using brace pattern: {pattern}")
                return result
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Try to clean and fix common JSON issues
    cleaned_attempts = []
    
    # Remove common prefixes/suffixes
    for prefix in ['Here is the JSON:', 'The JSON response is:', 'Response:', 'Result:']:
        if original_text.lower().startswith(prefix.lower()):
            cleaned_attempts.append(original_text[len(prefix):].strip())
    
    # Try to isolate JSON by finding first { and last }
    first_brace = original_text.find('{')
    last_brace = original_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        json_candidate = original_text[first_brace:last_brace + 1]
        cleaned_attempts.append(json_candidate)
    
    # Try each cleaned version
    for attempt in cleaned_attempts:
        try:
            result = json.loads(attempt.strip())
            logger.debug("Successfully parsed cleaned JSON")
            return result
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Try to fix common JSON formatting issues
    fix_attempts = [original_text]
    
    # Add attempts with common fixes
    for text in [original_text] + cleaned_attempts:
        if not text:
            continue
            
        # Fix single quotes to double quotes
        fixed_quotes = text.replace("'", '"')
        fix_attempts.append(fixed_quotes)
        
        # Fix trailing commas
        fixed_commas = re.sub(r',\s*}', '}', text)
        fixed_commas = re.sub(r',\s*]', ']', fixed_commas)
        fix_attempts.append(fixed_commas)
        
        # Fix missing quotes around keys (common LLM mistake)
        fixed_keys = re.sub(r'(\w+):', r'"\1":', text)
        fix_attempts.append(fixed_keys)
        
        # Combine fixes
        combined_fix = fixed_keys
        combined_fix = re.sub(r',\s*}', '}', combined_fix)
        combined_fix = re.sub(r',\s*]', ']', combined_fix)
        combined_fix = combined_fix.replace("'", '"')
        fix_attempts.append(combined_fix)
    
    # Try each fix attempt
    for attempt in fix_attempts:
        try:
            result = json.loads(attempt.strip())
            logger.debug("Successfully parsed JSON with formatting fixes")
            return result
        except json.JSONDecodeError:
            continue
    
    # Strategy 6: Manual key-value extraction as last resort
    # Look for common patterns like "key": "value" or key: value
    try:
        manual_extract = {}
        
        # Look for name field
        name_patterns = [
            r'"name"\s*:\s*"([^"]*)"',
            r"'name'\s*:\s*'([^']*)'",
            r'name\s*:\s*"([^"]*)"',
            r'name\s*:\s*([^,}\n]*)',
            r'cluster name is\s*"([^"]*)"',  # "The cluster name is X"
            r'name.*?[:\-]\s*"([^"]*)"',      # "name: X" or "name - X"
            r'called\s*"([^"]*)"'            # "called X"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, original_text, re.IGNORECASE)
            if match:
                manual_extract['name'] = match.group(1).strip()
                break
        
        # Look for description field
        desc_patterns = [
            r'"description"\s*:\s*"([^"]*)"',
            r"'description'\s*:\s*'([^']*)'",
            r'description\s*:\s*"([^"]*)"',
            r'description\s*:\s*([^,}\n]*)',
            r'description would be\s*"([^"]*)"',  # "description would be X"
            r'description.*?[:\-]\s*"([^"]*)"',   # "description: X"
            r'represents?\s*"([^"]*)"'           # "represents X"
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, original_text, re.IGNORECASE | re.DOTALL)
            if match:
                manual_extract['description'] = match.group(1).strip()
                break
        
        # Look for summary field (alternative to description)
        if 'description' not in manual_extract:
            summary_patterns = [
                r'"summary"\s*:\s*"([^"]*)"',
                r"'summary'\s*:\s*'([^']*)'",
                r'summary\s*:\s*"([^"]*)"'
            ]
            
            for pattern in summary_patterns:
                match = re.search(pattern, original_text, re.IGNORECASE | re.DOTALL)
                if match:
                    manual_extract['description'] = match.group(1).strip()
                    break
        
        # Look for keywords field (for summary responses)
        keywords_patterns = [
            r'"keywords"\s*:\s*\[(.*?)\]',
            r"'keywords'\s*:\s*\[(.*?)\]",
            r'keywords\s*:\s*\[(.*?)\]'
        ]
        
        for pattern in keywords_patterns:
            match = re.search(pattern, original_text, re.IGNORECASE | re.DOTALL)
            if match:
                keywords_str = match.group(1)
                # Try to parse keywords
                try:
                    keywords = json.loads(f'[{keywords_str}]')
                    manual_extract['keywords'] = keywords
                except:
                    # Fallback: split by comma and clean
                    keywords = [kw.strip(' "\'') for kw in keywords_str.split(',')]
                    manual_extract['keywords'] = [kw for kw in keywords if kw]
                break
        
        if manual_extract:
            logger.debug(f"Successfully extracted data manually: {manual_extract}")
            return manual_extract
            
    except Exception as e:
        logger.error(f"Manual extraction failed: {e}")
    
    logger.error(f"Failed to extract JSON from response: {original_text[:500]}...")
    return None


def safe_json_parse(response_text: str, fallback_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safe JSON parsing with fallback for critical operations.
    
    :param response_text: Raw response text from LLM
    :param fallback_dict: Fallback dictionary to use if parsing fails
    :return: Parsed dictionary or fallback
    """
    result = extract_json_from_response(response_text)
    
    if result is not None:
        return result
    
    if fallback_dict is not None:
        logger.warning(f"Using fallback dictionary due to JSON parsing failure")
        return fallback_dict
    
    # Default fallback
    logger.warning("Using default fallback dictionary")
    return {
        "name": "Unknown Cluster",
        "description": "Failed to parse cluster description from LLM response"
    }


SYSTEM_PROMPT = """# Role: Academic Paper Abstract Analyst

## Background: 
Academic research requires efficient processing and categorization of vast literature volumes. Researchers, students, and database managers need streamlined methods to quickly comprehend research essence and organize papers effectively. This analyst addresses the critical need for consistent, accurate abstract interpretation and keyword extraction that facilitates literature discovery, research gap identification, and cross-disciplinary knowledge synthesis.

## Attention: 
Your expertise in academic analysis is crucial for advancing research efficiency and knowledge organization. Accurate abstract interpretation and keyword extraction directly impacts research discovery, literature reviews, and academic database functionality. Your analytical precision enables researchers to quickly identify relevant work, understand research landscapes, and build upon existing knowledge foundations.

## Profile:
- Author: pp
- Version: 0.1
- Language: English
- Description: Specialized AI analyst with deep expertise in academic literature interpretation, research methodology recognition, and scientific communication synthesis across multiple disciplines

### Skills:
- Advanced abstract decomposition identifying core research problems, methodological approaches, and theoretical contributions with scholarly precision
- Sophisticated keyword extraction balancing specificity and generality for optimal research discoverability and categorization
- Cross-disciplinary research pattern recognition enabling accurate context understanding and methodological classification
- Scientific writing analysis interpreting complex research structures, experimental designs, and theoretical frameworks
- Academic terminology standardization ensuring consistent vocabulary usage across diverse research domains

## Goals:
- Systematically analyze academic abstracts to extract fundamental research problems and innovative solution approaches
- Generate comprehensive yet concise summaries capturing essential research contributions and methodological innovations
- Identify 7-10 strategically selected keywords representing core concepts, methodologies, and thematic elements
- Produce structured JSON outputs enabling seamless integration with research databases and literature management systems
- Maintain analytical objectivity while ensuring accessibility for diverse academic audiences

## Constraints:
- Maintain absolute fidelity to original abstract content without introducing interpretive bias or external assumptions
- Provide exactly 7-10 keywords balancing technical specificity with broader conceptual accessibility
- Structure all outputs as valid JSON format compatible with programmatic parsing and database integration
- Ensure summaries address both research problems and methodological solutions within concise, logical frameworks
- Apply consistent academic terminology standards while accommodating interdisciplinary vocabulary variations

## Workflow:
1. Conduct comprehensive abstract reading, identifying research context, objectives, methodological frameworks, and theoretical contributions
2. Extract and articulate the fundamental research problem, gap, or question driving the investigation
3. Analyze and synthesize the methodological approach, experimental design, or theoretical solution employed
4. Perform strategic keyword selection representing core concepts, methods, and thematic elements with optimal research discoverability
5. Validate JSON output structure ensuring proper formatting, field completeness, and programmatic compatibility

## OutputFormat:
- Valid JSON structure containing "summary" string field and "keywords" array field
- Summary addressing both research problem identification and methodological solution description
- Keywords array containing exactly 7-10 terms balancing specificity and generality
- Proper JSON syntax with UTF-8 encoding and appropriate character escaping
- Clean formatting without code blocks or extraneous explanatory content

## Suggestions:
- Consider interdisciplinary terminology when extracting keywords to enhance cross-field research discoverability
- Balance technical precision with accessibility ensuring summaries serve diverse academic audiences effectively
- Prioritize methodological keywords alongside conceptual terms for comprehensive research categorization
- Validate keyword diversity ensuring both specific techniques and broader research themes representation
- Maintain consistent analytical depth across different research domains and methodological approaches

## Initialization
As Academic Paper Abstract Analyst, you must follow Constraints and communicate with users using default Language. Analyze provided abstracts systematically according to Workflow, extracting research problems and solutions while identifying optimal keywords for JSON-formatted output.
"""

class AbstractAnalysisSchema(BaseModel):
    """
    A Pydantic model representing the structure of the abstract analysis output.
    """
    summary: str = Field(..., description="A concise summary of the research problem and solution approach.")
    keywords: list[str] = Field(..., description="A list of 7-10 relevant keywords representing the core concepts and themes of the research.")

class ClusterNamingSchema(BaseModel):
    """
    A Pydantic model representing the structure of the cluster naming output.
    """
    name: str = Field(..., description="A concise descriptive name for the research cluster (2-4 words).")
    description: str = Field(..., description="A brief description explaining what the cluster represents.")
    
class PaperInfo(BaseModel):
    """
    A Pydantic model representing a paper's metadata.
    """
    title: str = Field(..., description="The title of the paper.")
    abstract: str = Field(..., description="The abstract of the paper.")
    
class ClusterInfo(BaseModel):
    """
    A Pydantic model representing the information about a research cluster.
    """
    size: int = Field(..., description="The number of papers in the cluster.")
    papers: list[PaperInfo] = Field(..., description="A list of papers in the cluster.")
    name: str = Field(..., description="The name of the cluster.")

class ClusteringRequest(BaseModel):
    """
    A Pydantic model representing a request for clustering analysis.
    """
    papers: list[PaperInfo] = Field(..., description="A list of papers to be clustered.")
    max_num_clusters: int = Field(..., description="The number of clusters to create from the provided papers.")

class ClusteringResponse(BaseModel):
    """
    A Pydantic model representing the response from a clustering analysis.
    """
    clusters: list[ClusterInfo] = Field(..., description="A list of clusters created from the provided papers.")

    
def get_openai_client(api_key: str, base_url: str = "https://api.openai.com/v1") -> OpenAI:
    """
    Initialize and return an OpenAI client with the provided API key.
    
    :param api_key: The OpenAI API key.
    :return: An OpenAI client instance.
    """
    if not api_key:
        raise ValueError("API key must be provided")
    return OpenAI(api_key=api_key, base_url=base_url)

def get_async_openai_client(api_key: str, base_url: str = "https://api.openai.com/v1") -> AsyncOpenAI:
    """
    Initialize and return an async OpenAI client with the provided API key.
    
    :param api_key: The OpenAI API key.
    :param base_url: The OpenAI API base URL.
    :return: An async OpenAI client instance.
    """
    if not api_key:
        raise ValueError("API key must be provided")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

def get_openai_response(client: OpenAI, prompt: str, model: str = "gpt-4o-mini") -> str:
  """
  Get a response from the OpenAI API using the provided prompt with system prompt and structured output.
  
  :param client: An OpenAI client instance.
  :param prompt: The prompt to send to the OpenAI API.
  :param model: The model to use for generating the response.
  :return: The response text from the OpenAI API.
  """
  response = client.chat.completions.parse(
    model=model,
    messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": prompt}
    ],
    response_format=AbstractAnalysisSchema,
  )
  return response.choices[0].message.content.strip()

async def get_async_openai_response(client: AsyncOpenAI, prompt: str, model: str = "gpt-4o-mini") -> str:
  """
  Get a response from the OpenAI API asynchronously using the provided prompt with system prompt and structured output.
  
  :param client: An async OpenAI client instance.
  :param prompt: The prompt to send to the OpenAI API.
  :param model: The model to use for generating the response.
  :return: The response text from the OpenAI API.
  """
  response = await client.chat.completions.parse(
    model=model,
    messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": prompt}
    ],
    response_format=AbstractAnalysisSchema,
  )
  return response.choices[0].message.content.strip()

CLUSTER_NAMING_SYSTEM_PROMPT = """# Role: Research Cluster Naming Specialist

## Profile
- language: English
- description: A specialized AI expert in analyzing research clusters and creating descriptive names and explanations
- expertise: Academic research categorization, cluster analysis, taxonomic classification, research domain understanding
- goal: Generate concise, meaningful names for research paper clusters based on their content

## Task
You will be provided with information about a research cluster including:
- Sample paper titles from the cluster
- Sample paper abstracts from the cluster
- Common keywords from papers in the cluster
- Number of papers in the cluster

Based on this information, create:
1. A concise name (2-4 words) that captures the main research theme
2. A brief description explaining what the cluster represents

## Guidelines
- Names should be clear and descriptive
- Use academic terminology appropriate to the field
- Focus on the core research theme or methodology
- Avoid overly generic terms
- Consider both technical specificity and broad comprehensibility
- Analyze the actual research content from titles and abstracts rather than statistical features

## Output Format
Provide your response in JSON format with "name" and "description" fields.

Example:
{
  "name": "Neural Network Optimization",
  "description": "Research focused on improving neural network training algorithms and optimization techniques for better performance and efficiency."
}"""

async def get_cluster_name(client: AsyncOpenAI, cluster_info: dict, model: str = "gpt-4o-mini") -> str:
    """
    Generate a name for a research cluster using OpenAI API.
    
    :param client: An async OpenAI client instance.
    :param cluster_info: Dictionary containing cluster analysis information.
    :param model: The model to use for generating the response.
    :return: JSON string with cluster name and description.
    """
    # Extract relevant information from cluster analysis
    sample_titles = cluster_info.get('sample_titles', [])[:5]
    sample_abstracts = cluster_info.get('sample_abstracts', [])[:3]  # New: abstracts instead of TF-IDF
    common_keywords = cluster_info.get('common_keywords', {})
    cluster_size = cluster_info.get('size', 0)
    
    # Create prompt with cluster information (no TF-IDF features)
    prompt = f"""Analyze the following research cluster and provide a concise name and description:

Cluster Size: {cluster_size} papers

Sample Paper Titles:
{chr(10).join([f"- {title}" for title in sample_titles])}
"""

    # Add sample abstracts if available
    if sample_abstracts:
        prompt += f"""
Sample Paper Abstracts:
{chr(10).join([f"- {abstract[:300]}{'...' if len(abstract) > 300 else ''}" for abstract in sample_abstracts])}
"""

    # Add common keywords
    if common_keywords:
        prompt += f"""
Common Keywords from Papers:
{chr(10).join([f"- {keyword}: {count} papers" for keyword, count in list(common_keywords.items())[:10]])}
"""

    prompt += """
Please analyze the actual research content from the titles and abstracts to provide a concise name (2-4 words) and description for this research cluster."""

    response = await client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": CLUSTER_NAMING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format=ClusterNamingSchema,
    )
    return response.choices[0].message.content.strip()

LLM_CLUSTERING_SYSTEM_PROMPT = """# Role: Academic Paper Clustering Specialist

## Profile
- language: English
- description: A specialized AI expert in semantic clustering of academic papers based on research themes and methodologies
- expertise: Academic research categorization, thematic analysis, semantic similarity, research domain understanding
- goal: Group academic papers into coherent clusters based on their titles and abstracts, then provide meaningful names for each cluster

## Task
You will receive a collection of academic papers with their titles and abstracts. Your task is to:
1. Analyze the semantic content and research themes of each paper
2. Group papers into coherent clusters based on thematic similarity
3. Assign each paper to exactly one cluster
4. Provide a descriptive name for each cluster that captures the main research theme

## Clustering Guidelines
- Create clusters that are thematically coherent and meaningful
- Consider research methodology, domain, and problem focus when grouping papers
- Aim for balanced cluster sizes when possible, but prioritize thematic coherence
- Each paper should belong to exactly one cluster
- Cluster names should be concise (2-4 words) and descriptive
- Avoid overly generic cluster names

## Considerations
- Papers from the same research domain may belong to different clusters if they address different problems or use different approaches
- Papers from different domains may belong to the same cluster if they share similar methodologies or conceptual approaches
- Consider both explicit keywords and implicit thematic connections
- Balance specificity with broader applicability in cluster naming

## Output Format
Provide your response in JSON format matching the ClusteringResponse schema:
- clusters: Array of cluster objects
- Each cluster contains: name, size, and papers array
- Papers in each cluster should include title and abstract

## Quality Criteria
- Clusters should be internally coherent (papers within a cluster are thematically similar)
- Clusters should be externally distinct (clear differences between clusters)
- Cluster names should be immediately understandable and descriptive
- All input papers must be assigned to exactly one cluster
"""

async def get_llm_clustering(client: AsyncOpenAI, papers: list[dict], max_clusters: int, model: str = "gpt-4o") -> str:
    """
    Perform clustering of papers using LLM semantic analysis.
    
    :param client: An async OpenAI client instance.
    :param papers: List of paper dictionaries with 'title' and 'abstract' fields.
    :param max_clusters: Maximum number of clusters to create.
    :param model: The model to use for clustering (recommend gpt-4o for better reasoning).
    :return: JSON string with clustering results.
    """
    # Prepare papers for the clustering request
    paper_infos = []
    for i, paper in enumerate(papers):
        paper_infos.append({
            "title": paper.get("title", f"Paper {i+1}"),
            "abstract": paper.get("abstract", "")
        })
    
    # Create the prompt
    prompt = f"""Please cluster the following {len(papers)} academic papers into meaningful thematic groups.

Target number of clusters: {max_clusters} (you may create fewer if the papers naturally group into fewer coherent themes)

Papers to cluster:
"""
    
    for i, paper in enumerate(paper_infos, 1):
        prompt += f"\n{i}. Title: {paper['title']}\n"
        if paper['abstract']:
            # Truncate abstract if too long to stay within token limits
            abstract = paper['abstract'][:500] + "..." if len(paper['abstract']) > 500 else paper['abstract']
            prompt += f"   Abstract: {abstract}\n"
    
    prompt += f"""
Please analyze these papers and group them into {max_clusters} or fewer coherent clusters based on their research themes, methodologies, and content. Provide descriptive names for each cluster and ensure every paper is assigned to exactly one cluster."""

    response = await client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": LLM_CLUSTERING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format=ClusteringResponse,
    )
    return response.choices[0].message.content.strip()

def get_embedding_client(api_key: str, base_url: str = "https://api.openai.com/v1") -> OpenAI:
    """
    Initialize and return an OpenAI client for embeddings.
    
    :param api_key: The OpenAI API key.
    :param base_url: The OpenAI API base URL.
    :return: An OpenAI client instance for embeddings.
    """
    if not api_key:
        raise ValueError("API key must be provided")
    return OpenAI(api_key=api_key, base_url=base_url)

async def get_async_embedding_client(api_key: str, base_url: str = "https://api.openai.com/v1") -> AsyncOpenAI:
    """
    Initialize and return an async OpenAI client for embeddings.
    
    :param api_key: The OpenAI API key.
    :param base_url: The OpenAI API base URL.
    :return: An async OpenAI client instance for embeddings.
    """
    if not api_key:
        raise ValueError("API key must be provided")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

def get_text_embedding(client: OpenAI, text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Get text embedding using OpenAI API.
    
    :param client: An OpenAI client instance.
    :param text: The text to get embedding for.
    :param model: The embedding model to use.
    :return: The embedding vector as a list of floats.
    """
    if not text.strip():
        # Return zero vector for empty text
        response = client.embeddings.create(
            model=model,
            input="placeholder"
        )
        return [0.0] * len(response.data[0].embedding)
    
    response = client.embeddings.create(
        model=model,
        input=text.strip()[:8000]  # Truncate to avoid token limits
    )
    return response.data[0].embedding

async def get_async_text_embedding(client: AsyncOpenAI, text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Get text embedding using async OpenAI API.
    
    :param client: An async OpenAI client instance.
    :param text: The text to get embedding for.
    :param model: The embedding model to use.
    :return: The embedding vector as a list of floats.
    """
    if not text.strip():
        # Return zero vector for empty text
        response = await client.embeddings.create(
            model=model,
            input="placeholder"
        )
        return [0.0] * len(response.data[0].embedding)
    
    response = await client.embeddings.create(
        model=model,
        input=text.strip()[:8000]  # Truncate to avoid token limits
    )
    return response.data[0].embedding

def get_batch_text_embeddings(client: OpenAI, texts: List[str], model: str = "text-embedding-ada-002", 
                             batch_size: int = 100) -> List[List[float]]:
    """
    Get embeddings for multiple texts in batches.
    
    :param client: An OpenAI client instance.
    :param texts: List of texts to get embeddings for.
    :param model: The embedding model to use.
    :param batch_size: Number of texts to process in each batch.
    :return: List of embedding vectors.
    """
    all_embeddings = []
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Prepare texts (handle empty texts)
        processed_texts = []
        for text in batch_texts:
            if not text.strip():
                processed_texts.append("placeholder")
            else:
                processed_texts.append(text.strip()[:8000])  # Truncate to avoid token limits
        
        try:
            response = client.embeddings.create(
                model=model,
                input=processed_texts
            )
            
            batch_embeddings = []
            for j, data in enumerate(response.data):
                if batch_texts[j].strip():  # Original text was not empty
                    batch_embeddings.append(data.embedding)
                else:  # Original text was empty, return zero vector
                    batch_embeddings.append([0.0] * len(data.embedding))
            
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Return zero vectors for failed batch
            embedding_dim = 1536  # Default dimension for Ada-002
            for _ in batch_texts:
                all_embeddings.append([0.0] * embedding_dim)
    
    return all_embeddings

async def get_async_batch_text_embeddings(client: AsyncOpenAI, texts: List[str], model: str = "text-embedding-ada-002", 
                                        batch_size: int = 100, max_concurrent: int = 5) -> List[List[float]]:
    """
    Get embeddings for multiple texts in batches asynchronously.
    
    :param client: An async OpenAI client instance.
    :param texts: List of texts to get embeddings for.
    :param model: The embedding model to use.
    :param batch_size: Number of texts to process in each batch.
    :param max_concurrent: Maximum number of concurrent requests.
    :return: List of embedding vectors.
    """
    async def process_batch(batch_texts: List[str], batch_index: int) -> List[List[float]]:
        """Process a single batch of texts."""
        # Prepare texts (handle empty texts)
        processed_texts = []
        for text in batch_texts:
            if not text.strip():
                processed_texts.append("placeholder")
            else:
                processed_texts.append(text.strip()[:8000])  # Truncate to avoid token limits
        
        try:
            response = await client.embeddings.create(
                model=model,
                input=processed_texts
            )
            
            batch_embeddings = []
            for j, data in enumerate(response.data):
                if batch_texts[j].strip():  # Original text was not empty
                    batch_embeddings.append(data.embedding)
                else:  # Original text was empty, return zero vector
                    batch_embeddings.append([0.0] * len(data.embedding))
            
            return batch_embeddings
            
        except Exception as e:
            print(f"Error processing batch {batch_index}: {e}")
            # Return zero vectors for failed batch
            embedding_dim = 1536  # Default dimension for Ada-002
            return [[0.0] * embedding_dim for _ in batch_texts]
    
    # Create batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batches.append((batch_texts, i // batch_size))
    
    # Process batches with concurrency limit
    all_embeddings = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch_with_semaphore(batch_data):
        async with semaphore:
            return await process_batch(batch_data[0], batch_data[1])
    
    # Process all batches
    batch_results = await asyncio.gather(
        *[process_batch_with_semaphore(batch_data) for batch_data in batches],
        return_exceptions=True
    )
    
    # Flatten results
    for result in batch_results:
        if isinstance(result, Exception):
            print(f"Batch processing error: {result}")
            # Add zero vectors for failed batch
            embedding_dim = 1536
            all_embeddings.extend([[0.0] * embedding_dim for _ in range(batch_size)])
        else:
            all_embeddings.extend(result)
    
    return all_embeddings[:len(texts)]  # Ensure we return exactly the right number