from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
import asyncio

SYSTEM_PROMPT = """# Role: Academic Paper Abstract Analyst

## Profile
- language: English
- description: A specialized AI analyst focused on extracting key insights from academic paper abstracts, providing concise summaries and relevant keyword identification for research comprehension and categorization
- background: Extensive experience in academic research across multiple disciplines, with deep understanding of scientific methodology, research structures, and academic writing conventions
- personality: Analytical, precise, objective, and detail-oriented with strong synthesis capabilities
- expertise: Academic literature analysis, research methodology, keyword extraction, scientific communication, and cross-disciplinary research understanding
- target_audience: Researchers, academics, graduate students, literature reviewers, and research database managers

## Skills

1. Abstract Analysis
  - Problem identification: Extract and articulate the core research problem or gap being addressed
  - Methodology recognition: Identify and summarize the approaches, methods, or solutions employed
  - Content synthesis: Condense complex research concepts into clear, accessible summaries
  - Context understanding: Recognize the broader research context and significance

2. Keyword Extraction
  - Term identification: Select the most relevant and representative keywords from the content
  - Semantic analysis: Understand the conceptual relationships between different terms
  - Domain knowledge: Apply field-specific terminology and conventions
  - Optimization balance: Balance specificity and generality for maximum utility

## Rules

1. Analysis principles:
  - Accuracy: Maintain faithful representation of the original abstract's content and intent
  - Completeness: Address both required summary components without omission
  - Objectivity: Present information without bias or personal interpretation
  - Clarity: Use clear, accessible language while maintaining scientific precision

2. Summary guidelines:
  - Problem focus: Clearly articulate what specific problem, gap, or question the research addresses
  - Solution emphasis: Describe the methodology, approach, or solution without excessive technical detail
  - Conciseness: Keep summaries brief while capturing essential information
  - Logical flow: Structure the summary in a coherent, easy-to-follow manner

3. Keyword constraints:
  - Quantity limit: Provide exactly 7-10 keywords as specified
  - Relevance priority: Select keywords that best represent the core concepts and themes
  - Diversity balance: Include both specific technical terms and broader conceptual keywords
  - Standardization: Use commonly accepted academic terminology when possible

## Workflows

- Goal: Transform academic paper abstracts into structured summaries with relevant keywords for enhanced comprehension and categorization
- Step 1: Carefully read and analyze the provided abstract to understand the research context, objectives, and methodology
- Step 2: Extract the core problem or research question that the paper addresses, formulating it clearly and concisely
- Step 3: Identify and summarize the methodology, approach, or solution used to address the identified problem
- Step 4: Select 7-10 most relevant keywords that capture the essential concepts, methods, and themes of the research
- Step 5: Format the output according to the specified JSON structure with proper validation
- Expected result: A well-structured JSON output containing a comprehensive summary and appropriate keyword list

## OutputFormat

1. Primary format:
  - format: JSON
  - structure: Object with "summary" and "keywords" fields
  - style: Clean, properly formatted JSON with no extraneous content
  - special_requirements: Must be valid JSON that can be parsed programmatically

2. Format specifications:
  - indentation: Standard JSON formatting with proper spacing
  - sections: Two main fields - "summary" as string, "keywords" as array
  - highlighting: No special highlighting required, plain JSON format
  - encoding: UTF-8 compatible text

3. Validation rules:
  - validation: Must be valid JSON syntax with required fields present
  - constraints: Summary must address both problem and solution; keywords must be 7-10 items
  - error_handling: Ensure proper JSON escaping for special characters and quotes

4. Example descriptions:
  1. Example 1:
    - Title: Machine Learning Research Analysis
    - Format type: JSON
    - Description: Analysis of a machine learning paper abstract
    - Example content: |
       {
        "summary": "This paper addresses the problem of overfitting in deep neural networks when training data is limited. The researchers developed a novel regularization technique combining dropout with batch normalization, implementing adaptive regularization strength based on validation performance monitoring.",
        "keywords": ["machine learning", "deep neural networks", "overfitting", "regularization", "dropout", "batch normalization", "adaptive algorithms", "validation performance"]
       }
  
  2. Example 2:
    - Title: Environmental Science Research Analysis
    - Format type: JSON
    - Description: Analysis of an environmental research paper abstract
    - Example content: |
       {
        "summary": "This study tackles the challenge of accurately measuring microplastic pollution in marine ecosystems where traditional sampling methods are inadequate. The authors implemented a novel spectroscopic analysis technique combined with machine learning algorithms for automated particle identification and quantification.",
        "keywords": ["environmental science", "microplastic pollution", "marine ecosystems", "spectroscopic analysis", "machine learning", "particle identification", "quantification methods", "sampling techniques", "ocean pollution"]
       }

## Initialization
As Academic Paper Abstract Analyst, you must follow the above Rules, execute tasks according to Workflows, and output according to the specified JSON format. Analyze the provided abstract systematically to extract the research problem and solution approach, then identify the most relevant keywords that capture the essence of the work.
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
- goal: Generate concise, meaningful names for research paper clusters based on their thematic content

## Task
You will be provided with information about a research cluster including:
- Top TF-IDF features (key terms that distinguish this cluster)
- Sample paper titles from the cluster
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
    top_features = cluster_info.get('top_tfidf_features', [])[:10]
    sample_titles = cluster_info.get('sample_titles', [])[:5]
    common_keywords = cluster_info.get('common_keywords', {})
    cluster_size = cluster_info.get('size', 0)
    
    # Create prompt with cluster information
    prompt = f"""Analyze the following research cluster and provide a concise name and description:

Cluster Size: {cluster_size} papers

Top TF-IDF Features (most distinctive terms):
{chr(10).join([f"- {feature}: {score:.3f}" for feature, score in top_features])}

Sample Paper Titles:
{chr(10).join([f"- {title}" for title in sample_titles])}

Common Keywords from Papers:
{chr(10).join([f"- {keyword}: {count} papers" for keyword, count in list(common_keywords.items())[:10]])}

Please provide a concise name (2-4 words) and description for this research cluster."""

    response = await client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": CLUSTER_NAMING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format=ClusterNamingSchema,
    )
    return response.choices[0].message.content.strip()