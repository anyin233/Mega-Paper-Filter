from typing import Optional, List, Dict
import time
import random
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from loguru import logger
import re


class GoogleScholarExtractor:
    """
    A class to extract abstracts from Google Scholar and its mirrors.
    More reliable than publisher websites and handles connection issues.
    """
    
    # Google Scholar mirrors for different regions/accessibility
    SCHOLAR_MIRRORS = {
        'default': 'https://scholar.google.com',
        'mirror1': 'https://scholar.google.ac.uk',
        'mirror2': 'https://scholar.google.ca', 
        'mirror3': 'https://scholar.google.com.au',
        'mirror4': 'https://scholar.google.de',
        'scihub': 'https://sci-hub.se',  # Alternative source
        'semantic': 'https://www.semanticscholar.org'  # Semantic Scholar as backup
    }
    
    def __init__(self, headless: bool = False, timeout: int = 15, preferred_mirror: str = 'default'):
        """
        Initialize the Google Scholar extractor.
        
        :param headless: Whether to run browser in headless mode
        :param timeout: Timeout for web element waits
        :param preferred_mirror: Preferred Google Scholar mirror to use
        """
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        self.current_mirror = preferred_mirror
        self.available_mirrors = list(self.SCHOLAR_MIRRORS.keys())
        
        if preferred_mirror not in self.SCHOLAR_MIRRORS:
            logger.warning(f"Unknown mirror '{preferred_mirror}', using 'default'")
            self.current_mirror = 'default'
    
    def _setup_driver(self):
        """Setup Chrome WebDriver with Google Scholar optimized settings."""
        logger.debug("Setting up Chrome WebDriver for Google Scholar...")
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        # Anti-detection options for Google Scholar
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-extensions")
        
        # User agent rotation for Google Scholar
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        selected_ua = random.choice(user_agents)
        chrome_options.add_argument(f"--user-agent={selected_ua}")
        
        # Additional options
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            chrome_driver_path = ChromeDriverManager().install()
            service = Service(chrome_driver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Execute stealth scripts
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.debug("Google Scholar WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {str(e)}")
            raise
    
    def _teardown_driver(self):
        """Close the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.debug("WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {str(e)}")
    
    def _try_mirror(self, mirror_name: str) -> bool:
        """
        Test if a mirror is accessible.
        
        :param mirror_name: Name of the mirror to test
        :return: True if accessible, False otherwise
        """
        if mirror_name not in self.SCHOLAR_MIRRORS:
            return False
        
        url = self.SCHOLAR_MIRRORS[mirror_name]
        try:
            logger.debug(f"Testing mirror: {mirror_name} ({url})")
            self.driver.get(url)
            
            # Wait a moment and check if page loaded
            time.sleep(random.uniform(2, 4))
            
            # Check if we got a valid page (not blocked)
            page_title = self.driver.title.lower()
            if any(keyword in page_title for keyword in ['scholar', 'semantic', 'sci-hub']):
                logger.info(f"✓ Mirror '{mirror_name}' is accessible")
                return True
            else:
                logger.debug(f"Mirror '{mirror_name}' returned unexpected page: {page_title}")
                return False
                
        except Exception as e:
            logger.debug(f"Mirror '{mirror_name}' failed: {str(e)}")
            return False
    
    def _find_working_mirror(self) -> str:
        """
        Find a working Google Scholar mirror.
        
        :return: Name of working mirror or None if all fail
        """
        # Try current mirror first
        mirrors_to_try = [self.current_mirror] + [m for m in self.available_mirrors if m != self.current_mirror]
        
        for mirror in mirrors_to_try:
            if self._try_mirror(mirror):
                self.current_mirror = mirror
                return mirror
        
        logger.error("All mirrors failed!")
        return None
    
    def search_paper(self, title: str, authors: List[str] = None) -> Optional[str]:
        """
        Search for a paper on Google Scholar and return the abstract from search results.
        
        :param title: Paper title (authors parameter ignored - using title-only search)
        :param authors: List of author names (kept for compatibility but not used)
        :return: Abstract text if found, None otherwise
        """
        if not self.driver:
            self._setup_driver()
        
        # Find working mirror
        working_mirror = self._find_working_mirror()
        if not working_mirror:
            logger.error("No working mirrors found")
            return None
        
        try:
            return self._search_on_scholar(title, working_mirror)
        except Exception as e:
            logger.error(f"Error searching paper: {str(e)}")
            return None
    
    def _search_on_scholar(self, title: str, mirror: str) -> Optional[str]:
        """
        Perform the actual search on Google Scholar using title only.
        
        :param title: Paper title
        :param mirror: Mirror to use
        :return: Abstract text if found
        """
        base_url = self.SCHOLAR_MIRRORS[mirror]
        
        if mirror == 'semantic':
            return self._search_semantic_scholar(title)
        elif mirror == 'scihub':
            logger.debug("Sci-Hub doesn't support direct abstract search, skipping")
            return None
        else:
            return self._search_google_scholar_simple(base_url, title)
    
    def _prepare_search_query(self, title: str, authors: List[str] = None) -> str:
        """
        Prepare search query for Google Scholar.
        
        :param title: Paper title
        :param authors: Author names
        :return: Formatted search query
        """
        # Clean title
        title_clean = re.sub(r'[^\w\s]', ' ', title)
        title_clean = ' '.join(title_clean.split())
        
        query_parts = [f'"{title_clean}"']
        
        # Add first author if available
        if authors and len(authors) > 0:
            first_author = authors[0].split()[-1]  # Use last name
            query_parts.append(f'author:"{first_author}"')
        
        return ' '.join(query_parts)
    
    def _search_google_scholar_simple(self, base_url: str, title: str) -> Optional[str]:
        """
        Search Google Scholar with title only and extract abstract from search results.
        
        :param base_url: Base URL of the mirror
        :param title: Paper title
        :return: Abstract if found
        """
        try:
            # Simple title-only search - just encode the title
            clean_title = title.strip()
            encoded_title = urllib.parse.quote(clean_title)
            search_url = f"{base_url}/scholar?q={encoded_title}"
            
            logger.debug(f"Searching Google Scholar: {search_url}")
            logger.debug(f"Search query: {clean_title}")
            
            self.driver.get(search_url)
            
            # Wait for results to load
            time.sleep(random.uniform(2, 4))  # Shorter wait time
            
            # Look for search results
            results = self.driver.find_elements(By.CSS_SELECTOR, ".gs_r.gs_or.gs_scl, .gs_r")
            
            logger.debug(f"Found {len(results)} search results")
            
            # Check first few results for matching title and extract abstract
            for i, result in enumerate(results[:5]):  # Check first 5 results
                try:
                    abstract = self._extract_abstract_from_search_result(result, title)
                    if abstract:
                        logger.info(f"✓ Abstract found in result {i+1}")
                        return abstract
                
                except Exception as e:
                    logger.debug(f"Error processing result {i+1}: {str(e)}")
                    continue
            
            logger.warning("No matching results with abstracts found")
            return None
            
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {str(e)}")
            return None
    
    def _extract_abstract_from_search_result(self, result_element, original_title: str) -> Optional[str]:
        """
        Extract abstract from a Google Scholar search result.
        
        :param result_element: Selenium element of the search result
        :param original_title: Original paper title for matching
        :return: Abstract text if found and title matches
        """
        try:
            # Get the title from this result
            title_selectors = [
                "h3.gs_rt a",
                "h3.gs_rt",
                ".gs_rt a",
                ".gs_rt"
            ]
            
            result_title = None
            for selector in title_selectors:
                try:
                    title_element = result_element.find_element(By.CSS_SELECTOR, selector)
                    result_title = title_element.text.strip()
                    if result_title:
                        break
                except:
                    continue
            
            if not result_title:
                logger.debug("Could not find title in result")
                return None
            
            # Check if titles match (fuzzy matching)
            if not self._titles_match(original_title, result_title):
                logger.debug(f"Title mismatch: '{result_title[:50]}...' vs '{original_title[:50]}...'")
                return None
            
            logger.debug(f"Title match found: {result_title[:50]}...")
            
            # Extract abstract/snippet from the search result
            abstract_selectors = [
                ".gs_rs",  # Main snippet text
                ".gs_a + div",  # Text after authors line
                "[data-a] .gs_rs",  # Alternative snippet selector
                ".gs_fl + div",  # Text after footer links
                ".gs_ai .gs_rs"  # Snippet within author info section
            ]
            
            for selector in abstract_selectors:
                try:
                    abstract_element = result_element.find_element(By.CSS_SELECTOR, selector)
                    abstract_text = abstract_element.text.strip()
                    
                    # Validate and clean abstract
                    if self._is_valid_abstract(abstract_text):
                        cleaned_abstract = self._clean_abstract_text(abstract_text)
                        if len(cleaned_abstract) > 50:
                            logger.debug(f"Found abstract using selector '{selector}': {len(cleaned_abstract)} chars")
                            return cleaned_abstract
                        
                except:
                    continue
            
            logger.debug("No valid abstract found in result")
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting abstract from result: {str(e)}")
            return None
    
    def _is_valid_abstract(self, text: str) -> bool:
        """
        Check if text looks like a valid abstract.
        
        :param text: Text to check
        :return: True if it looks like an abstract
        """
        if not text or len(text) < 30:
            return False
        
        # Exclude obvious non-abstract content
        exclude_patterns = [
            'Cited by',
            'Related articles', 
            'All versions',
            '[PDF]',
            '[HTML]',
            'Save to Library',
            'Create Alert',
            'Export',
            'doi.org',
            'Download PDF'
        ]
        
        text_lower = text.lower()
        for pattern in exclude_patterns:
            if pattern.lower() in text_lower:
                return False
        
        return True
    
    def _search_google_scholar(self, base_url: str, search_query: str, original_title: str) -> Optional[str]:
        """
        Search Google Scholar for abstract.
        
        :param base_url: Base URL of the mirror
        :param search_query: Formatted search query
        :param original_title: Original paper title for matching
        :return: Abstract if found
        """
        try:
            # Construct search URL
            encoded_query = urllib.parse.quote(search_query)
            search_url = f"{base_url}/scholar?q={encoded_query}"
            
            logger.debug(f"Searching Google Scholar: {search_url}")
            self.driver.get(search_url)
            
            # Wait for results to load
            time.sleep(random.uniform(3, 6))
            
            # Find search results
            results = self.driver.find_elements(By.CSS_SELECTOR, ".gs_r.gs_or.gs_scl")
            
            logger.debug(f"Found {len(results)} search results")
            
            for i, result in enumerate(results[:3]):  # Check first 3 results
                try:
                    # Get result title
                    title_element = result.find_element(By.CSS_SELECTOR, "h3.gs_rt a, h3.gs_rt")
                    result_title = title_element.text.strip()
                    
                    # Check if this matches our paper (fuzzy match)
                    if self._titles_match(original_title, result_title):
                        logger.debug(f"Found matching result {i+1}: {result_title[:50]}...")
                        
                        # Try to find abstract in the result
                        abstract = self._extract_abstract_from_result(result)
                        if abstract:
                            logger.info("✓ Abstract found on Google Scholar")
                            return abstract
                
                except Exception as e:
                    logger.debug(f"Error processing result {i+1}: {str(e)}")
                    continue
            
            logger.warning("No matching results with abstracts found on Google Scholar")
            return None
            
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {str(e)}")
            return None
    
    def _extract_abstract_from_result(self, result_element) -> Optional[str]:
        """
        Extract abstract from a Google Scholar search result.
        
        :param result_element: Selenium element of the search result
        :return: Abstract text if found
        """
        try:
            # Try different selectors for abstract/snippet
            abstract_selectors = [
                ".gs_rs",  # Main abstract/snippet text
                ".gs_a + div",  # Text after authors
                ".gs_fl + div",  # Text after footer links
            ]
            
            for selector in abstract_selectors:
                try:
                    abstract_element = result_element.find_element(By.CSS_SELECTOR, selector)
                    abstract_text = abstract_element.text.strip()
                    
                    # Validate abstract (should be substantial and not just metadata)
                    if (len(abstract_text) > 50 and 
                        not abstract_text.startswith('Cited by') and
                        not abstract_text.startswith('Related articles')):
                        
                        # Clean up the abstract
                        abstract_text = self._clean_abstract_text(abstract_text)
                        if len(abstract_text) > 30:
                            return abstract_text
                        
                except NoSuchElementException:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting abstract from result: {str(e)}")
            return None
    
    def _search_semantic_scholar(self, title: str) -> Optional[str]:
        """
        Search Semantic Scholar for abstract using title only.
        
        :param title: Paper title
        :return: Abstract if found
        """
        try:
            # Semantic Scholar search with title only
            search_query = urllib.parse.quote(title.strip())
            search_url = f"https://www.semanticscholar.org/search?q={search_query}"
            
            logger.debug(f"Searching Semantic Scholar: {search_url}")
            self.driver.get(search_url)
            
            time.sleep(random.uniform(3, 5))
            
            # Look for search results
            results = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='search-result'], .result-card")
            
            logger.debug(f"Found {len(results)} results on Semantic Scholar")
            
            for i, result in enumerate(results[:3]):
                try:
                    # Check title match
                    title_element = result.find_element(By.CSS_SELECTOR, "h3 a, h2 a, .result-title a")
                    result_title = title_element.text.strip()
                    
                    if self._titles_match(title, result_title):
                        logger.debug(f"Found matching title on Semantic Scholar: {result_title[:50]}...")
                        
                        # Try to find abstract
                        abstract_selectors = [
                            ".cl-paper-abstract", 
                            ".result-abstract",
                            ".paper-abstract",
                            "[data-testid='abstract']"
                        ]
                        
                        for selector in abstract_selectors:
                            try:
                                abstract_element = result.find_element(By.CSS_SELECTOR, selector)
                                abstract_text = abstract_element.text.strip()
                                
                                if len(abstract_text) > 50:
                                    logger.info("✓ Abstract found on Semantic Scholar")
                                    return self._clean_abstract_text(abstract_text)
                            except:
                                continue
                
                except Exception as e:
                    logger.debug(f"Error processing Semantic Scholar result {i+1}: {str(e)}")
                    continue
            
            logger.debug("No matching results found on Semantic Scholar")
            return None
            
        except Exception as e:
            logger.debug(f"Error searching Semantic Scholar: {str(e)}")
            return None
    
    def _titles_match(self, original: str, candidate: str, threshold: float = 0.7) -> bool:
        """
        Check if two titles match (fuzzy matching).
        Lowered threshold for better matching with title-only search.
        
        :param original: Original title
        :param candidate: Candidate title to match
        :param threshold: Similarity threshold (lowered to 0.7 for better matching)
        :return: True if titles match
        """
        # Simple fuzzy matching based on common words
        orig_words = set(re.findall(r'\w+', original.lower()))
        cand_words = set(re.findall(r'\w+', candidate.lower()))
        
        if not orig_words or not cand_words:
            return False
        
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        orig_words = orig_words - common_words
        cand_words = cand_words - common_words
        
        if not orig_words or not cand_words:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(orig_words.intersection(cand_words))
        union = len(orig_words.union(cand_words))
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= threshold
    
    def _clean_abstract_text(self, text: str) -> str:
        """
        Clean and format abstract text.
        
        :param text: Raw abstract text
        :return: Cleaned abstract text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['Abstract', 'ABSTRACT', 'Summary:', 'Abstract:', '…']
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove trailing citations, URLs, etc.
        text = re.sub(r'\s*\[.*?\]$', '', text)  # Remove [citations]
        text = re.sub(r'\s*https?://\S+', '', text)  # Remove URLs
        
        return text.strip()
    
    def __enter__(self):
        """Context manager entry."""
        self._setup_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._teardown_driver()
        return False