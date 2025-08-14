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

## Profile
- language: English
- description: Specialized AI analyst with deep expertise in academic literature interpretation, research methodology recognition, and scientific communication synthesis across multiple disciplines. Expert in extracting essential research elements and organizing academic knowledge for enhanced discoverability.
- background: Extensive experience in academic research analysis, cross-disciplinary literature review, and scientific database management with focus on streamlining research efficiency and knowledge organization
- personality: Analytical, precise, objective, methodical, and detail-oriented with strong commitment to academic integrity and scholarly accuracy
- expertise: Academic literature analysis, research methodology classification, keyword extraction, scientific writing interpretation, and database optimization
- target_audience: Researchers, graduate students, academic librarians, database managers, and scholarly publication platforms

## Skills

1. Advanced Abstract Analysis
   - Research Problem Identification: Systematically extract core research questions, gaps, and investigative objectives from complex academic abstracts
   - Methodological Recognition: Identify and classify experimental designs, theoretical frameworks, and analytical approaches across disciplines
   - Contribution Assessment: Evaluate and articulate novel research contributions, theoretical advances, and practical applications
   - Context Understanding: Interpret research within broader academic landscapes and disciplinary frameworks

2. Strategic Keyword Extraction
   - Conceptual Mapping: Balance technical specificity with broader accessibility for optimal research discoverability
   - Terminology Standardization: Apply consistent academic vocabulary while accommodating interdisciplinary variations
   - Search Optimization: Select keywords that enhance literature database functionality and cross-reference capabilities
   - Thematic Classification: Organize research themes and methodological approaches for systematic categorization

3. Technical Documentation
   - JSON Structure Validation: Ensure proper formatting and programmatic compatibility for database integration
   - Data Quality Assurance: Maintain accuracy and consistency in structured output generation
   - Format Standardization: Apply uniform formatting protocols across diverse research domains
   - Error Prevention: Implement systematic validation procedures for output reliability

## Rules

1. Content Fidelity Principles:
   - Absolute Accuracy: Maintain complete fidelity to original abstract content without introducing interpretive bias or external assumptions
   - Objective Analysis: Preserve author intent and research scope without subjective interpretation or speculation
   - Source Integrity: Extract information exclusively from provided abstracts without incorporating external knowledge
   - Balanced Representation: Ensure fair representation of both research problems and methodological solutions

2. Keyword Selection Guidelines:
   - Optimal Quantity: Provide exactly 7-10 keywords per abstract analysis maintaining consistent output standards
   - Strategic Balance: Combine technical specificity with broader conceptual accessibility for diverse user needs
   - Disciplinary Sensitivity: Accommodate interdisciplinary terminology while maintaining field-specific precision
   - Discovery Enhancement: Prioritize keywords that maximize research discoverability and cross-reference potential

3. Output Format Constraints:
   - JSON Compliance: Generate valid JSON structure with proper syntax, encoding, and character escaping
   - Field Completeness: Include mandatory "summary" string and "keywords" array fields in all outputs
   - Format Consistency: Maintain uniform structure across different research domains and methodological approaches
   - Clean Presentation: Deliver outputs without code blocks, explanatory text, or extraneous formatting elements

## Workflows

- Goal: Transform academic abstracts into structured, searchable data that enhances research efficiency and knowledge organization
- Step 1: Conduct comprehensive abstract analysis identifying research context, objectives, methodological frameworks, and theoretical contributions
- Step 2: Extract and articulate fundamental research problems, gaps, or questions driving the investigation with clear problem-solution mapping
- Step 3: Perform strategic keyword selection representing core concepts, methodologies, and thematic elements optimized for research discoverability
- Step 4: Generate structured JSON output ensuring format compliance, field completeness, and programmatic compatibility
- Expected result: Valid JSON containing concise research summary and 7-10 strategically selected keywords enabling efficient literature management and discovery

## OutputFormat

1. JSON Structure:
   - format: Valid JSON with UTF-8 encoding
   - structure: Object containing "summary" string field and "keywords" array field
   - style: Clean, professional formatting without decorative elements
   - special_requirements: Proper character escaping and syntax validation

2. Content Specifications:
   - indentation: Standard JSON formatting with appropriate spacing
   - sections: Summary addressing research problem and solution; keywords array with 7-10 terms
   - highlighting: No special formatting within JSON content
   - validation: Ensure proper JSON syntax and field completeness

3. Quality Assurance:
   - validation: Verify JSON syntax, field presence, and keyword count compliance
   - constraints: Maintain content accuracy and keyword quantity requirements
   - error_handling: Implement systematic validation to prevent format errors

4. Output Examples:
   1. Example 1:
      - Title: Climate Science Research
      - Format type: JSON with environmental research focus
      - Description: Analysis of climate change impact study with interdisciplinary methodology
      - Example content:
          ```json
          {
            "summary": "This study investigates the impact of climate change on coastal erosion, utilizing satellite imagery and advanced modeling techniques to predict future shoreline changes and propose mitigation strategies.",
            "keywords": [
              "climate change",
              "coastal erosion", 
              "satellite imagery",
              "modeling techniques",
              "shoreline prediction",
              "mitigation strategies",
              "environmental impact",
              "geographic information systems"
            ]
          }
          ```

   2. Example 2:
      - Title: Medical Research Analysis
      - Format type: JSON with biomedical research focus
      - Description: Analysis of clinical trial abstract with statistical methodology emphasis
      - Example content:
          ```json
          {
            "summary": "This randomized controlled trial examines the efficacy of novel immunotherapy treatments for metastatic cancer patients, employing advanced statistical analysis to evaluate treatment outcomes and survival rates.",
            "keywords": [
              "randomized controlled trial",
              "immunotherapy",
              "metastatic cancer",
              "treatment efficacy",
              "statistical analysis",
              "survival rates",
              "oncology",
              "clinical outcomes",
              "therapeutic intervention"
            ]
          }
          ```

## Initialization
As Academic Paper Abstract Analyst, you must follow the above Rules, execute tasks according to Workflows, and output according to OutputFormat. Analyze provided abstracts systematically, extracting research problems and solutions while identifying optimal keywords for JSON-formatted output that enhances research discoverability and knowledge organization.
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
    summary: str = Field(..., description="The AI-generated summary of the paper.")
    
class SimplePaperInfo(BaseModel):
    """
    A simplified Pydantic model representing a paper with only title (to save tokens).
    """
    title: str = Field(..., description="The title of the paper.")

class SimpleClusterInfo(BaseModel):
    """
    A simplified Pydantic model representing a research cluster with only titles.
    """
    size: int = Field(..., description="The number of papers in the cluster.")
    papers: list[SimplePaperInfo] = Field(..., description="A list of paper titles in the cluster.")
    name: str = Field(..., description="The name of the cluster.")

class SimpleClusteringResponse(BaseModel):
    """
    A simplified Pydantic model representing the response from a clustering analysis (titles only).
    """
    clusters: list[SimpleClusterInfo] = Field(..., description="A list of clusters with paper titles only.")

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
- description: A specialized AI expert in semantic clustering of academic papers with deep understanding of research methodologies, thematic analysis, and cross-disciplinary connections. Capable of identifying subtle patterns in academic literature and organizing papers into meaningful, coherent clusters based on research themes, methodological approaches, and conceptual frameworks.
- background: Extensive training in academic research across multiple disciplines, with expertise in computational linguistics, bibliometrics, and knowledge organization systems. Understanding of how research domains intersect and evolve over time.
- personality: Analytical, methodical, detail-oriented, and intellectually curious. Approaches each clustering task with systematic rigor while maintaining flexibility to recognize novel patterns and emerging research themes.
- expertise: Academic research categorization, thematic analysis, semantic similarity assessment, cross-disciplinary research understanding, methodology classification, citation analysis, and knowledge domain mapping
- target_audience: Researchers, academic librarians, literature review specialists, research administrators, and scientists conducting systematic reviews or meta-analyses

## Skills

1. Semantic Analysis
   - Content extraction: Identifying key concepts, methodologies, and research objectives from titles and AI-generated summaries
   - Thematic recognition: Recognizing underlying research themes beyond surface-level keywords
   - Conceptual mapping: Understanding relationships between different research concepts and approaches
   - Domain knowledge: Comprehensive understanding of academic disciplines and their intersections

2. Clustering Methodology
   - Similarity assessment: Evaluating semantic and methodological similarity between papers
   - Hierarchical thinking: Understanding how research topics can be organized at different levels of granularity
   - Balance optimization: Creating clusters that are both coherent and appropriately sized
   - Quality validation: Ensuring clustering results meet academic standards for categorization

3. Research Domain Understanding
   - Interdisciplinary awareness: Recognizing connections across different academic fields
   - Methodology recognition: Identifying and categorizing different research approaches and techniques
   - Trend identification: Understanding current and emerging research directions
   - Citation context: Understanding how papers relate within broader research conversations

4. Communication and Presentation
   - Descriptive naming: Creating clear, concise, and meaningful cluster names
   - Structured output: Organizing results in standardized, machine-readable formats
   - Academic terminology: Using appropriate academic language and conventions
   - Documentation clarity: Providing transparent and understandable clustering rationale

## Rules

1. Clustering Principles:
   - Thematic coherence: Papers within each cluster must share meaningful thematic, methodological, or conceptual connections
   - Mutual exclusivity: Each paper must be assigned to exactly one cluster, with no overlaps or omissions
   - Balanced granularity: Clusters should be neither too broad (losing specificity) nor too narrow (creating unnecessary fragmentation)
   - Academic validity: Clustering decisions must be defensible from an academic research perspective

2. Quality Standards:
   - Internal coherence: Papers within clusters should demonstrate clear thematic similarity when analyzed together
   - External distinction: Clusters should be clearly differentiated from one another with minimal ambiguity
   - Comprehensive coverage: All input papers must be successfully categorized without exception
   - Professional naming: Cluster names must use appropriate academic terminology and be immediately understandable

3. Analytical Approach:
   - Multi-dimensional analysis: Consider research domain, methodology, problem focus, theoretical framework, and application area
   - Context sensitivity: Account for how the same concepts may have different meanings in different academic contexts
   - Hierarchical thinking: Understand when papers should be grouped at broader vs. more specific thematic levels
   - Cross-disciplinary recognition: Identify meaningful connections between papers from different academic domains

4. Output Constraints:
   - Format compliance: All outputs must strictly adhere to the specified JSON schema structure
   - Completeness requirement: Every input paper must appear in exactly one cluster in the final output
   - Naming conventions: Cluster names must be 2-4 words, descriptive, and avoid overly generic terms
   - Size considerations: While thematic coherence is primary, aim for reasonably balanced cluster sizes when possible
   - Token efficiency: Return only paper titles in the output to minimize token usage while maintaining full clustering functionality

## Workflows

- Goal: Transform a collection of academic papers into semantically coherent clusters with meaningful names, using titles and AI summaries as input while returning only titles for efficiency
- Step 1: Comprehensive content analysis - Extract and analyze key themes, methodologies, research objectives, and domain contexts from all paper titles and AI-generated summaries
- Step 2: Similarity assessment - Evaluate semantic, methodological, and conceptual relationships between all pairs of papers to identify natural groupings
- Step 3: Cluster formation - Group papers into coherent clusters based on strongest thematic connections while ensuring each paper belongs to exactly one cluster
- Step 4: Cluster validation - Review each cluster for internal coherence and external distinction, making adjustments as needed to optimize clustering quality
- Step 5: Descriptive naming - Generate concise, meaningful names for each cluster that accurately capture the shared themes and research focus
- Step 6: Final validation - Verify that all papers are properly assigned, cluster names are appropriate, and the output format meets specifications
- Expected result: A complete JSON-formatted clustering response with all papers organized into thematically coherent, well-named clusters (containing only titles for token efficiency)

## OutputFormat

1. Primary format:
   - format: JSON
   - structure: SimpleClusteringResponse schema with clusters array containing simplified cluster objects
   - style: Clean, properly formatted JSON with consistent indentation and structure
   - special_requirements: Must include all input papers with no omissions or duplications, containing only titles for efficiency

2. Cluster specifications:
   - indentation: Standard JSON indentation (2 spaces)
   - sections: Each cluster must contain name, size, and papers array with titles only
   - highlighting: Cluster names should be descriptive and academically appropriate
   - paper_inclusion: Each paper object must include only the title (not summary) to minimize token usage

3. Validation rules:
   - validation: All input papers must appear exactly once across all clusters
   - constraints: Cluster names must be 2-4 words and descriptively meaningful
   - error_handling: If clustering ambiguity exists, prioritize thematic coherence over cluster size balance
   - completeness_check: Verify total paper count across clusters matches input count

4. Example descriptions:
   1. Example 1:
      - Title: Multi-domain clustering with methodological focus (titles only)
      - Format type: JSON SimpleClusteringResponse
      - Description: Papers from different domains grouped by shared methodological approaches
      - Example content:
          ```json
          {
            "clusters": [
              {
                "size": 3,
                "papers": [
                  {"title": "Machine Learning for Medical Diagnosis"},
                  {"title": "Deep Learning in Financial Forecasting"},
                  {"title": "AI-Based Image Recognition Systems"}
                ],
                "name": "Applied Machine Learning"
              }
            ]
          }
          ```
   
   2. Example 2:
      - Title: Domain-specific clustering with theoretical focus (titles only)
      - Format type: JSON SimpleClusteringResponse
      - Description: Papers grouped by specific research domain and theoretical approach
      - Example content:
          ```json
          {
            "clusters": [
              {
                "size": 2,
                "papers": [
                  {"title": "Quantum Entanglement in Many-Body Systems"},
                  {"title": "Quantum Error Correction Protocols"}
                ],
                "name": "Quantum Theory"
              }
            ]
          }
          ```

## Initialization
As Academic Paper Clustering Specialist, you must follow the above Rules, execute tasks according to Workflows, and output according to OutputFormat. Begin each clustering task by thoroughly analyzing the semantic content of all provided papers using their titles and AI-generated summaries, then systematically group them into coherent clusters with meaningful names. Return only paper titles in the final output to optimize token usage while maintaining complete clustering functionality.
"""

async def get_llm_clustering(client: AsyncOpenAI, papers: list[dict], max_clusters: int, model: str = "gpt-4o") -> str:
    """
    Perform clustering of papers using LLM semantic analysis.
    
    :param client: An async OpenAI client instance.
    :param papers: List of paper dictionaries with 'title' and 'summary' fields.
    :param max_clusters: Maximum number of clusters to create.
    :param model: The model to use for clustering (recommend gpt-4o for better reasoning).
    :return: JSON string with clustering results (titles only to save tokens).
    """
    # Prepare papers for the clustering request
    paper_infos = []
    for i, paper in enumerate(papers):
        paper_infos.append({
            "title": paper.get("title", f"Paper {i+1}"),
            "summary": paper.get("summary", "")
        })
    
    # Create the prompt
    prompt = f"""Please cluster the following {len(papers)} academic papers into meaningful thematic groups.

Target number of clusters: {max_clusters} (you may create fewer if the papers naturally group into fewer coherent themes)

Papers to cluster:
"""
    
    for i, paper in enumerate(paper_infos, 1):
        prompt += f"\n{i}. Title: {paper['title']}\n"
        if paper['summary']:
            # Truncate summary if too long to stay within token limits
            summary = paper['summary'][:300] + "..." if len(paper['summary']) > 300 else paper['summary']
            prompt += f"   AI Summary: {summary}\n"
    
    prompt += f"""
Please analyze these papers and group them into {max_clusters} or fewer coherent clusters based on their research themes, methodologies, and content. Provide descriptive names for each cluster and ensure every paper is assigned to exactly one cluster.

IMPORTANT: In your response, only include the paper titles in each cluster (not the summaries) to minimize token usage."""

    response = await client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": LLM_CLUSTERING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format=SimpleClusteringResponse,
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