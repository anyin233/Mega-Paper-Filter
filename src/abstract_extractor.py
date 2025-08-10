from typing import Optional
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from loguru import logger
from .google_scholar_extractor import GoogleScholarExtractor


class AbstractExtractor:
    """
    A class to extract abstracts from IEEE Xplore and ACM Digital Library using Selenium.
    Enhanced with comprehensive anti-spider protection and debugging capabilities.
    """
    
    def __init__(self, headless: bool = False, timeout: int = 20, debug: bool = False, 
                 use_google_scholar: bool = True, scholar_mirror: str = 'default'):
        """
        Initialize the AbstractExtractor with enhanced anti-spider protection.
        
        :param headless: Whether to run the browser in headless mode (default: False for debugging)
        :param timeout: Timeout for web element waits (increased default for dynamic content)
        :param debug: Enable debug mode with shorter timeouts for testing
        :param use_google_scholar: Use Google Scholar as primary source (recommended)
        :param scholar_mirror: Google Scholar mirror to use ('default', 'mirror1', etc.)
        """
        self.headless = headless
        self.timeout = timeout if not debug else 5  # Shorter timeout for debug mode
        self.debug = debug
        self.driver = None
        self.use_google_scholar = use_google_scholar
        self.scholar_mirror = scholar_mirror
        
        if debug:
            logger.info("🐛 Debug mode enabled: Using shorter timeouts and verbose logging")
        if not headless:
            logger.info("🖥️  Browser window will be visible for debugging")
        if use_google_scholar:
            logger.info(f"📚 Google Scholar mode enabled using mirror: {scholar_mirror}")
    
    def _setup_driver(self):
        """Setup Chrome WebDriver with anti-detection options."""
        logger.debug("Setting up Chrome WebDriver with anti-detection measures...")
        chrome_options = Options()
        
        # Basic options
        if self.headless:
            chrome_options.add_argument("--headless=new")  # Use new headless mode
            logger.debug("Running in headless mode")
        else:
            logger.debug("Running with visible browser window")
        
        # Anti-detection options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")  # Faster loading
        
        # User agent rotation
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0"
        ]
        selected_ua = random.choice(user_agents)
        chrome_options.add_argument(f"--user-agent={selected_ua}")
        logger.debug(f"Using user agent: {selected_ua}")
        
        # Additional stealth options
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0,
            "profile.managed_default_content_settings.images": 2,
            # Disable various background activities
            "profile.default_content_settings.geolocation": 2,
            "profile.default_content_settings.media_stream": 2,
        })
        
        # Add page load strategy for better control
        chrome_options.page_load_strategy = 'normal'  # Wait for all resources
        
        try:
            # Auto-download ChromeDriver if not present
            logger.debug("Checking for ChromeDriver and downloading if necessary...")
            chrome_driver_path = ChromeDriverManager().install()
            logger.debug(f"ChromeDriver path: {chrome_driver_path}")
            
            # Create service with the auto-downloaded driver
            service = Service(chrome_driver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Execute stealth scripts
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")
            self.driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})")
            self.driver.execute_script("window.chrome = {runtime: {}}")
            
            # Set window size to common resolution
            if not self.headless:
                self.driver.set_window_size(1366, 768)
            
            logger.debug("Chrome WebDriver initialized successfully with anti-detection measures")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {str(e)}")
            raise
    
    def _teardown_driver(self):
        """Close the WebDriver."""
        if self.driver:
            logger.debug("Closing Chrome WebDriver...")
            try:
                self.driver.quit()
                self.driver = None
                logger.debug("Chrome WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing Chrome WebDriver: {str(e)}")
                self.driver = None
    
    def _simulate_human_behavior_simple(self):
        """Simple human behavior simulation without mouse movements."""
        try:
            # Random delay
            delay = random.uniform(2.0, 5.0)
            logger.debug(f"Simple human behavior: waiting {delay:.2f}s")
            time.sleep(delay)
            
            # Simple page interaction - just scroll a bit
            try:
                scroll_amount = random.randint(50, 200)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.5, 1.0))
                self.driver.execute_script(f"window.scrollBy(0, -{scroll_amount});")
                logger.debug("Simple scroll behavior completed")
            except Exception as e:
                logger.debug(f"Even simple scroll failed: {str(e)}")
                # Just wait if everything fails
                time.sleep(random.uniform(1.0, 2.0))
                
        except Exception as e:
            logger.debug(f"Error in simple human behavior simulation: {str(e)}")
    
    def _simulate_human_behavior(self):
        """Simulate human-like behavior to avoid detection."""
        try:
            # Try enhanced behavior first
            self._simulate_human_behavior_enhanced()
        except Exception as e:
            logger.debug(f"Enhanced behavior failed: {str(e)}, trying simple behavior")
            self._simulate_human_behavior_simple()
    
    def _simulate_human_behavior_enhanced(self):
        """Enhanced human behavior with mouse movements (can fail)."""
        # Random delay
        delay = random.uniform(1.5, 4.0)
        logger.debug(f"Simulating human delay: {delay:.2f}s")
        time.sleep(delay)
        
        # Simulate mouse movement with bounds checking
        actions = ActionChains(self.driver)
        
        # Get window size and ensure we have valid dimensions
        window_size = self.driver.get_window_size()
        width = max(window_size.get('width', 1366), 800)  # Minimum width
        height = max(window_size.get('height', 768), 600)  # Minimum height
        
        logger.debug(f"Window size: {width}x{height}")
        
        # Safe mouse movement within reasonable bounds (avoiding edges)
        safe_margin = 100  # Pixels from edge
        safe_width = max(width - (2 * safe_margin), 200)
        safe_height = max(height - (2 * safe_margin), 200)
        
        # Random mouse movements within safe area
        for _ in range(random.randint(1, 2)):  # Fewer movements
            # Calculate safe coordinates
            x = random.randint(safe_margin, safe_margin + safe_width)
            y = random.randint(safe_margin, safe_margin + safe_height)
            
            logger.debug(f"Moving mouse to relative position ({x}, {y}) within safe bounds")
            
            # Move to body element first, then offset from there
            body = self.driver.find_element(By.TAG_NAME, "body")
            actions.move_to_element_with_offset(body, x // 6, y // 6)  # Even smaller offsets
            actions.pause(random.uniform(0.2, 0.8))
        
        # Sometimes scroll a bit (safer than mouse movements)
        if random.random() < 0.3:  # Reduced probability
            scroll_amount = random.randint(100, 300)  # Smaller scroll amounts
            logger.debug(f"Scrolling by {scroll_amount} pixels")
            self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            time.sleep(random.uniform(0.5, 1.0))
            self.driver.execute_script(f"window.scrollBy(0, -{scroll_amount});")
        
        # Perform the actions
        actions.perform()
    
    def _wait_for_page_load(self):
        """Wait for page to be fully loaded with multiple strategies."""
        try:
            wait = WebDriverWait(self.driver, self.timeout)
            
            # 1. Wait for document ready state
            logger.debug("Waiting for document ready state...")
            wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")
            
            # 2. Wait for jQuery to finish loading (if present)
            try:
                wait.until(lambda driver: driver.execute_script("return typeof jQuery === 'undefined' || jQuery.active == 0"))
                logger.debug("jQuery loading completed")
            except:
                logger.debug("jQuery not present or timeout")
            
            # 3. Wait for any pending network requests to complete
            time.sleep(random.uniform(2, 4))  # Additional buffer for dynamic content
            
            logger.debug("Page load waiting completed")
            
        except Exception as e:
            logger.debug(f"Error in page load waiting: {str(e)}")
    
    def _wait_for_content_load(self, site_type: str):
        """Wait specifically for abstract content to load based on site type."""
        try:
            wait = WebDriverWait(self.driver, max(self.timeout, 15))  # Ensure minimum 15s timeout
            
            if site_type.lower() == "ieee":
                # Wait for IEEE specific elements to load
                logger.debug("Waiting for IEEE content to load...")
                
                # Wait for main content area
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".document-main, .document, main")))
                    logger.debug("IEEE main content area loaded")
                except TimeoutException:
                    logger.debug("IEEE main content area not found, continuing...")
                
                # Wait for abstract section specifically
                abstract_indicators = [
                    ".abstract-text",
                    "[data-testid='abstract-text']",
                    ".abstract",
                    ".u-mb-1"
                ]
                
                for indicator in abstract_indicators:
                    try:
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, indicator)))
                        logger.debug(f"IEEE abstract indicator '{indicator}' found")
                        break
                    except TimeoutException:
                        continue
                        
            elif site_type.lower() == "acm":
                # Wait for ACM specific elements to load
                logger.debug("Waiting for ACM content to load...")
                
                # Wait for main article content
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".article-content, .main-content, main, article")))
                    logger.debug("ACM main content area loaded")
                except TimeoutException:
                    logger.debug("ACM main content area not found, continuing...")
                
                # Wait for abstract section specifically
                abstract_indicators = [
                    "section#abstract",  # The main abstract section
                    "#abstract",
                    "section#abstract div[role='paragraph']",  # The actual content divs
                    "#abstract div[role='paragraph']",
                    ".abstractSection",
                    ".abstract"
                ]
                
                for indicator in abstract_indicators:
                    try:
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, indicator)))
                        logger.debug(f"ACM abstract indicator '{indicator}' found")
                        break
                    except TimeoutException:
                        continue
            
            # Additional wait for dynamic content
            time.sleep(random.uniform(1, 3))
            logger.debug(f"{site_type} content loading completed")
            
        except Exception as e:
            logger.debug(f"Error waiting for {site_type} content: {str(e)}")
    
    def _debug_page_content(self, site_type: str = "unknown"):
        """
        Debug function to inspect page content and find potential selectors.
        
        :param site_type: Type of site for focused debugging
        """
        try:
            logger.info(f"=== DEBUG: Inspecting {site_type} page content ===")
            
            # Basic page info
            logger.info(f"Page title: {self.driver.title}")
            logger.info(f"Page URL: {self.driver.current_url}")
            
            # Check if page loaded properly
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            logger.info(f"Body text length: {len(body_text)} characters")
            
            # Look for abstract-related elements
            abstract_keywords = ['abstract', 'Abstract', 'ABSTRACT']
            for keyword in abstract_keywords:
                elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{keyword}')]")
                logger.info(f"Found {len(elements)} elements containing '{keyword}'")
                
                for i, element in enumerate(elements[:3]):  # Show first 3
                    try:
                        tag_name = element.tag_name
                        element_id = element.get_attribute('id')
                        element_class = element.get_attribute('class')
                        element_text = element.text[:100] if element.text else "No text"
                        
                        logger.info(f"  {i+1}. Tag: {tag_name}, ID: {element_id}, Class: {element_class}")
                        logger.info(f"     Text: {element_text}...")
                    except Exception as e:
                        logger.debug(f"Error inspecting element {i}: {str(e)}")
            
            # Check for common abstract containers
            potential_containers = [
                '#abstract', 'section#abstract', '.abstract', '.abstractSection',
                '.abstract-text', '[data-testid="abstract-text"]',
                '.document-abstract', '.article-abstract'
            ]
            
            logger.info("=== Checking potential abstract containers ===")
            for selector in potential_containers:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        element = elements[0]
                        logger.info(f"✓ Found: {selector}")
                        logger.info(f"  Tag: {element.tag_name}, ID: {element.get_attribute('id')}, Class: {element.get_attribute('class')}")
                        text_content = element.text[:200] if element.text else "No text content"
                        logger.info(f"  Text: {text_content}...")
                    else:
                        logger.debug(f"✗ Not found: {selector}")
                except Exception as e:
                    logger.debug(f"Error checking {selector}: {str(e)}")
            
            # Save page source for manual inspection if needed
            try:
                page_source = self.driver.page_source
                logger.info(f"Page source length: {len(page_source)} characters")
                
                # Check if we're blocked
                blocked_indicators = [
                    "access denied", "blocked", "captcha", "robot", "bot detection",
                    "403 forbidden", "not authorized", "security check"
                ]
                
                page_source_lower = page_source.lower()
                for indicator in blocked_indicators:
                    if indicator in page_source_lower:
                        logger.warning(f"⚠️  Possible blocking detected: '{indicator}' found in page source")
                
            except Exception as e:
                logger.debug(f"Error analyzing page source: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in page content debugging: {str(e)}", exc_info=True)
    
    def _verify_page_readiness(self) -> bool:
        """
        Verify that the page is ready for content extraction.
        
        :return: True if page appears ready, False otherwise
        """
        try:
            # Check basic page state
            if self.driver.execute_script("return document.readyState") != "complete":
                logger.debug("Document not in complete state")
                return False
            
            # Check if we have substantial content
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            if len(body_text.strip()) < 100:
                logger.debug(f"Body text too short: {len(body_text)} chars")
                return False
            
            # Check for common loading indicators
            loading_indicators = [
                ".loading",
                ".spinner",
                "[data-loading='true']",
                ".loader"
            ]
            
            for indicator in loading_indicators:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, indicator)
                    if elements and any(el.is_displayed() for el in elements):
                        logger.debug(f"Loading indicator '{indicator}' still visible")
                        return False
                except:
                    continue
            
            logger.debug("Page readiness verification passed")
            return True
            
        except Exception as e:
            logger.debug(f"Error verifying page readiness: {str(e)}")
            return False
    
    def _extract_abstract_fallback(self) -> Optional[str]:
        """
        Fallback method to extract abstract using more flexible approaches.
        
        :return: Abstract text if found, None otherwise
        """
        try:
            logger.info("Trying fallback abstract extraction methods...")
            
            # Method 1: Find by text content containing "Abstract"
            logger.debug("Fallback method 1: Looking for 'Abstract' heading followed by content")
            try:
                # Find elements containing "Abstract" text
                abstract_headers = self.driver.find_elements(
                    By.XPATH, 
                    "//*[contains(translate(text(), 'ABSTRACT', 'abstract'), 'abstract')]"
                )
                
                for header in abstract_headers:
                    try:
                        # Look for content after the header
                        parent = header.find_element(By.XPATH, "./..")
                        text = parent.text.strip()
                        
                        if len(text) > 100:  # Substantial content
                            # Try to extract just the abstract part
                            lines = text.split('\n')
                            abstract_started = False
                            abstract_lines = []
                            
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                if 'abstract' in line.lower() and len(line) < 50:
                                    abstract_started = True
                                    continue
                                
                                if abstract_started:
                                    if any(keyword in line.lower() for keyword in ['introduction', 'keywords', 'references', '1.']):
                                        break
                                    abstract_lines.append(line)
                                    
                                    # Stop if we have enough content
                                    if len(' '.join(abstract_lines)) > 200:
                                        break
                            
                            if abstract_lines:
                                abstract_text = ' '.join(abstract_lines).strip()
                                if len(abstract_text) > 50:
                                    logger.info(f"Fallback method 1 successful: {len(abstract_text)} chars")
                                    return abstract_text
                    except Exception as e:
                        logger.debug(f"Error with abstract header: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Fallback method 1 failed: {str(e)}")
            
            # Method 2: Search in page text for abstract-like content
            logger.debug("Fallback method 2: Pattern matching in page text")
            try:
                body_text = self.driver.find_element(By.TAG_NAME, "body").text
                
                # Look for abstract pattern in text
                import re
                
                # Pattern: Abstract followed by text until next section
                patterns = [
                    r'(?i)abstract[:\s]*\n*(.+?)(?=\n\s*(?:introduction|keywords|references|1\.|©|\d+\.\s+[A-Z]))',
                    r'(?i)abstract[:\s]*(.+?)(?=introduction|keywords|references)',
                    r'(?i)abstract\s*[:\-–—]?\s*(.{100,500}?)(?=\n\s*[A-Z][a-z]+:|\n\s*\d+\.|\n\s*Keywords|$)'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, body_text, re.DOTALL)
                    for match in matches:
                        abstract_text = match.strip()
                        # Clean up the text
                        abstract_text = re.sub(r'\s+', ' ', abstract_text)
                        abstract_text = abstract_text.replace('\n', ' ').strip()
                        
                        if 50 < len(abstract_text) < 2000:  # Reasonable length
                            logger.info(f"Fallback method 2 successful: {len(abstract_text)} chars")
                            return abstract_text
                            
            except Exception as e:
                logger.debug(f"Fallback method 2 failed: {str(e)}")
            
            # Method 3: Look for any substantial paragraph near "abstract" mention
            logger.debug("Fallback method 3: Finding paragraphs near abstract mentions")
            try:
                paragraphs = self.driver.find_elements(By.TAG_NAME, "p")
                
                for i, p in enumerate(paragraphs):
                    text = p.text.strip()
                    
                    # Check if this or nearby paragraphs mention abstract
                    context_range = 3
                    start_idx = max(0, i - context_range)
                    end_idx = min(len(paragraphs), i + context_range + 1)
                    
                    context_text = ""
                    for j in range(start_idx, end_idx):
                        context_text += paragraphs[j].text.lower()
                    
                    if 'abstract' in context_text and len(text) > 100:
                        # This paragraph is near an abstract mention and has content
                        if not any(word in text.lower() for word in ['copyright', 'permission', 'citation', 'download']):
                            logger.info(f"Fallback method 3 successful: {len(text)} chars")
                            return text
                            
            except Exception as e:
                logger.debug(f"Fallback method 3 failed: {str(e)}")
            
            logger.warning("All fallback methods failed")
            return None
            
        except Exception as e:
            logger.error(f"Error in fallback extraction: {str(e)}", exc_info=True)
            return None
    
    def extract_abstract_from_paper_info(self, paper_title: str, authors: list = None) -> Optional[str]:
        """
        Extract abstract using Google Scholar search based on paper information.
        
        :param paper_title: Title of the paper
        :param authors: List of author names
        :return: Abstract text if found
        """
        if not self.use_google_scholar:
            logger.debug("Google Scholar disabled, skipping")
            return None
        
        try:
            logger.info(f"🔍 Searching Google Scholar for: {paper_title[:60]}{'...' if len(paper_title) > 60 else ''}")
            
            with GoogleScholarExtractor(
                headless=self.headless, 
                timeout=self.timeout,
                preferred_mirror=self.scholar_mirror
            ) as scholar:
                abstract = scholar.search_paper(paper_title, authors or [])
                
                if abstract:
                    logger.info(f"✅ Google Scholar abstract found ({len(abstract)} chars)")
                    return abstract
                else:
                    logger.warning("❌ No abstract found on Google Scholar")
                    return None
                    
        except Exception as e:
            logger.error(f"Error extracting from Google Scholar: {str(e)}")
            return None
    
    def _human_like_page_load(self, url: str, wait_time: tuple = (3, 6)):
        """Load a page with comprehensive waiting for full content load."""
        logger.debug(f"Loading page with comprehensive waiting: {url}")
        self.driver.get(url)
        
        # 1. Basic wait for initial load
        initial_wait = random.uniform(*wait_time)
        logger.debug(f"Initial wait: {initial_wait:.2f}s")
        time.sleep(initial_wait)
        
        # 2. Wait for page to be fully loaded
        self._wait_for_page_load()
        
        # 3. Simulate human behavior
        self._simulate_human_behavior()
        
        # 4. Determine site type and wait for specific content
        current_url = self.driver.current_url
        if "ieee" in current_url.lower():
            self._wait_for_content_load("ieee")
        elif "acm.org" in current_url.lower():
            self._wait_for_content_load("acm")
        
        # 5. Verify page readiness with retry
        max_readiness_attempts = 3
        for attempt in range(max_readiness_attempts):
            if self._verify_page_readiness():
                logger.debug(f"Page readiness verified on attempt {attempt + 1}")
                break
            else:
                if attempt < max_readiness_attempts - 1:
                    wait_time = random.uniform(2, 4)
                    logger.debug(f"Page not ready, waiting {wait_time:.2f}s before retry {attempt + 2}")
                    time.sleep(wait_time)
                else:
                    logger.warning("Page readiness could not be verified after max attempts")
        
        # 6. Final buffer wait
        final_wait = random.uniform(1, 2)
        logger.debug(f"Final buffer wait: {final_wait:.2f}s")
        time.sleep(final_wait)
    
    def extract_ieee_abstract(self, url: str) -> Optional[str]:
        """
        Extract abstract from IEEE Xplore paper page with human-like behavior.
        
        :param url: The IEEE Xplore paper URL
        :return: The abstract text or None if extraction fails
        """
        if not url or "ieee" not in url.lower():
            logger.warning(f"URL does not appear to be from IEEE Xplore: {url}")
            return None
        
        logger.debug(f"Starting IEEE abstract extraction for URL: {url}")
        
        try:
            logger.debug("Loading IEEE page with human-like behavior...")
            self._human_like_page_load(url)
            
            return self._extract_ieee_abstract_from_loaded_page()
                
        except Exception as e:
            logger.error(f"Unexpected error extracting abstract from IEEE URL {url}: {str(e)}", exc_info=True)
            return None
    
    def extract_acm_abstract(self, url: str) -> Optional[str]:
        """
        Extract abstract from ACM Digital Library paper page with human-like behavior.
        
        :param url: The ACM Digital Library paper URL
        :return: The abstract text or None if extraction fails
        """
        if not url or "acm.org" not in url.lower():
            logger.warning(f"URL does not appear to be from ACM Digital Library: {url}")
            return None
        
        logger.debug(f"Starting ACM abstract extraction for URL: {url}")
        
        try:
            logger.debug("Loading ACM page with human-like behavior...")
            self._human_like_page_load(url)
            
            return self._extract_acm_abstract_from_loaded_page()
                
        except Exception as e:
            logger.error(f"Unexpected error extracting abstract from ACM URL {url}: {str(e)}", exc_info=True)
            return None
    
    def _extract_ieee_abstract_from_loaded_page(self) -> Optional[str]:
        """
        Extract abstract from IEEE page that's already loaded in the driver.
        Waits for content to be fully loaded before extraction.
        
        :return: The abstract text or None if extraction fails
        """
        try:
            # Ensure content is fully loaded
            self._wait_for_content_load("ieee")
            
            wait = WebDriverWait(self.driver, self.timeout)
            
            # Try multiple selectors for abstract with progressive waiting
            abstract_selectors = [
                "div.abstract-text div.u-mb-1",
                ".abstract-text",
                "[data-testid='abstract-text']",
                ".abstract .u-mb-1",
                ".abstract-desktop-div .u-mb-1",
                # Additional selectors for different IEEE layouts
                ".document-abstract-content",
                ".abstract-content p",
                ".Abstract .u-mb-1",
                # More generic fallbacks
                ".abstract p",
                ".abstract div",
                "[class*='abstract'] p",
                "[id*='abstract'] p"
            ]
            
            abstract_text = None
            for i, selector in enumerate(abstract_selectors, 1):
                logger.debug(f"Trying IEEE selector {i}/{len(abstract_selectors)}: '{selector}'")
                try:
                    # Wait with longer timeout for dynamic content
                    abstract_element = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    
                    # Additional wait for text content to populate
                    time.sleep(random.uniform(0.5, 1.5))
                    
                    abstract_text = abstract_element.text.strip()
                    if abstract_text and len(abstract_text) > 50:  # Ensure we have substantial content
                        logger.debug(f"Found abstract text using selector '{selector}' (length: {len(abstract_text)} chars)")
                        logger.info(f"Successfully extracted IEEE abstract: {abstract_text[:100]}...")
                        break
                    elif abstract_text:
                        logger.debug(f"Selector '{selector}' found text but too short ({len(abstract_text)} chars): {abstract_text[:50]}")
                    else:
                        logger.debug(f"Selector '{selector}' found element but text was empty")
                except TimeoutException:
                    logger.debug(f"Timeout waiting for selector '{selector}'")
                    continue
                except Exception as e:
                    logger.debug(f"Error with selector '{selector}': {str(e)}")
                    continue
            
            if abstract_text:
                logger.info("IEEE abstract extraction successful")
                return abstract_text
            else:
                logger.warning("Could not find abstract text on IEEE page after comprehensive waiting")
                
                # Debug the page to understand what went wrong
                logger.warning("Running debug analysis to identify issues...")
                self._debug_page_content("IEEE")
                
                # Try fallback extraction methods
                logger.warning("Attempting fallback extraction methods...")
                fallback_abstract = self._extract_abstract_fallback()
                if fallback_abstract:
                    logger.info("✓ Fallback extraction successful!")
                    return fallback_abstract
                
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error extracting abstract from loaded IEEE page: {str(e)}", exc_info=True)
            return None

    def _extract_acm_abstract_from_loaded_page(self) -> Optional[str]:
        """
        Extract abstract from ACM page that's already loaded in the driver.
        Waits for content to be fully loaded before extraction.
        
        :return: The abstract text or None if extraction fails
        """
        try:
            # Ensure content is fully loaded
            self._wait_for_content_load("acm")
            
            # Updated selectors based on real ACM DL structure
            abstract_selectors = [
                # Primary selectors based on the actual ACM DL structure
                "section#abstract div[role='paragraph']",  # Most specific - targets the content divs
                "#abstract div[role='paragraph']",
                "section[id='abstract'] div[role='paragraph']",
                
                # Alternative approaches for the same structure
                "section#abstract div",  # All divs in abstract section
                "#abstract div",
                
                # Fallback to get all text content from abstract section
                "section#abstract",  # Get entire abstract section text
                "#abstract",
                
                # Legacy selectors for older ACM layouts
                "section#abstract p",
                "#abstract p",
                "div#abstract p",
                
                # More specific selectors for nested structures
                "section[id='abstract'] div p",
                "#abstract div p",
                "section#abstract div.abstractSection p",
                "#abstract .abstract-content p",
                
                # Backup selectors for different ACM layouts
                ".abstractSection p",
                ".abstract p",
                "[data-title='Abstract'] p",
                ".hlFld-Abstract p",
                ".abstractInFull p",
                ".section-abstract p",
                
                # Additional modern ACM selectors
                ".article-section[data-title='Abstract'] p",
                "[role='article'] #abstract p",
                
                # More generic fallbacks
                "[class*='abstract'] p",
                "[id*='abstract'] p",
                "section[class*='abstract'] p",
                ".article-content .abstract p"
            ]
            
            abstract_text = None
            for i, selector in enumerate(abstract_selectors, 1):
                logger.debug(f"Trying ACM selector {i}/{len(abstract_selectors)}: '{selector}'")
                try:
                    # Wait with longer timeout for dynamic content
                    abstract_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    if abstract_elements:
                        # Additional wait for text content to populate
                        time.sleep(random.uniform(0.5, 1.5))
                        
                        # Handle multiple elements (like multiple div[role='paragraph'])
                        if len(abstract_elements) > 1:
                            # Combine text from multiple elements
                            text_parts = []
                            for element in abstract_elements:
                                element_text = element.text.strip()
                                if element_text and len(element_text) > 10:  # Avoid empty or very short elements
                                    text_parts.append(element_text)
                            
                            if text_parts:
                                abstract_text = ' '.join(text_parts).strip()
                                logger.debug(f"Combined {len(text_parts)} elements using selector '{selector}' (total length: {len(abstract_text)} chars)")
                        else:
                            # Single element
                            abstract_text = abstract_elements[0].text.strip()
                            logger.debug(f"Single element using selector '{selector}' (length: {len(abstract_text)} chars)")
                        
                        if abstract_text and len(abstract_text) > 50:  # Ensure we have substantial content
                            logger.info(f"Successfully extracted ACM abstract: {abstract_text[:100]}...")
                            break
                        elif abstract_text:
                            logger.debug(f"Selector '{selector}' found text but too short ({len(abstract_text)} chars): {abstract_text[:50]}")
                            abstract_text = None  # Reset for next iteration
                        else:
                            logger.debug(f"Selector '{selector}' found elements but no text content")
                    else:
                        logger.debug(f"No elements found for selector '{selector}'")
                        
                except Exception as e:
                    logger.debug(f"Error with selector '{selector}': {str(e)}")
                    continue
            
            if abstract_text:
                logger.info("ACM abstract extraction successful")
                return abstract_text
            else:
                logger.warning("Could not find abstract text on ACM page after comprehensive waiting")
                
                # Debug the page to understand what went wrong
                logger.warning("Running debug analysis to identify issues...")
                self._debug_page_content("ACM")
                
                # Try fallback extraction methods
                logger.warning("Attempting fallback extraction methods...")
                fallback_abstract = self._extract_abstract_fallback()
                if fallback_abstract:
                    logger.info("✓ Fallback extraction successful!")
                    return fallback_abstract
                
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error extracting abstract from loaded ACM page: {str(e)}", exc_info=True)
            return None
    
    def _is_url_supported(self, url: str) -> tuple[bool, str]:
        """
        Check if a URL is supported after following redirects.
        
        :param url: The URL to check
        :return: Tuple of (is_supported, final_url_after_redirect)
        """
        if not url or url == "N/A":
            return False, ""
        
        if not self.driver:
            self._setup_driver()
        
        try:
            logger.debug(f"Checking URL support: {url}")
            self.driver.get(url)
            
            # Wait for redirects to complete
            time.sleep(2)
            
            final_url = self.driver.current_url
            logger.debug(f"Final URL after redirect: {final_url}")
            
            # Check if the final URL is from a supported site
            is_supported = ("ieee" in final_url.lower() or "acm.org" in final_url.lower())
            
            return is_supported, final_url
            
        except Exception as e:
            logger.error(f"Error checking URL support for {url}: {str(e)}")
            return False, ""

    def check_papers_url_support(self, papers: list) -> dict:
        """
        Check which papers have supported URLs after redirect resolution.
        
        :param papers: List of PaperInfo objects
        :return: Dictionary with support statistics and unsupported papers
        """
        logger.info(f"Checking URL support for {len(papers)} papers...")
        
        if not self.driver:
            self._setup_driver()
        
        supported_count = 0
        unsupported_papers = []
        results = {}
        
        try:
            for i, paper in enumerate(papers):
                logger.info(f"Checking paper {i+1}/{len(papers)}: '{paper.title[:60]}{'...' if len(paper.title) > 60 else ''}'")
                
                is_supported, final_url = self._is_url_supported(paper.url)
                
                if is_supported:
                    supported_count += 1
                    logger.info(f"✓ Supported - redirects to: {final_url}")
                else:
                    unsupported_papers.append({
                        'title': paper.title,
                        'original_url': paper.url,
                        'final_url': final_url,
                        'doi': paper.doi
                    })
                    logger.warning(f"✗ Unsupported - redirects to: {final_url}")
        
        except Exception as e:
            logger.error(f"Error during URL support checking: {str(e)}")
        finally:
            results = {
                'total_papers': len(papers),
                'supported_count': supported_count,
                'unsupported_count': len(unsupported_papers),
                'unsupported_papers': unsupported_papers
            }
            
            logger.info(f"URL support check completed: {supported_count}/{len(papers)} papers supported")
        
        return results

    def extract_abstract(self, url: str) -> Optional[str]:
        """
        Extract abstract from either IEEE Xplore or ACM Digital Library.
        Handles DOI URL redirections to determine the final destination.
        Uses human-like behavior to avoid anti-spider detection.
        
        :param url: The paper URL (may be a DOI URL that redirects)
        :return: The abstract text or None if extraction fails
        """
        if not url or url == "N/A":
            logger.warning("No valid URL provided for abstract extraction")
            return None
        
        logger.info(f"Starting abstract extraction for URL: {url}")
        
        if not self.driver:
            logger.debug("WebDriver not initialized, setting up driver...")
            self._setup_driver()
        
        try:
            # Load the URL with human-like behavior and let it redirect if it's a DOI
            logger.debug(f"Loading URL with human-like behavior (may redirect): {url}")
            self._human_like_page_load(url, wait_time=(3, 6))  # Longer wait for redirects
            
            # Get the final URL after any redirects
            final_url = self.driver.current_url
            logger.debug(f"Final URL after redirect: {final_url}")
            
            # Check if the final URL is from a supported site
            if "ieee" in final_url.lower():
                logger.debug("Final URL detected as IEEE Xplore, using IEEE extractor")
                return self._extract_ieee_abstract_from_loaded_page()
            elif "acm.org" in final_url.lower():
                logger.debug("Final URL detected as ACM Digital Library, using ACM extractor")
                return self._extract_acm_abstract_from_loaded_page()
            else:
                logger.warning(f"Final URL is not from supported sites (IEEE or ACM): {final_url}")
                # Try to detect other common academic sites
                if any(site in final_url.lower() for site in ["springer", "sciencedirect", "arxiv"]):
                    logger.info(f"Detected unsupported academic site: {final_url}")
                    return None
                else:
                    logger.warning(f"Unknown site after redirect: {final_url}")
                    return None
                    
        except Exception as e:
            logger.error(f"Unexpected error during abstract extraction from {url}: {str(e)}", exc_info=True)
            return None
    
    def extract_abstracts_batch(self, papers: list, delay: float = 5.0) -> list:
        """
        Extract abstracts for a batch of papers with comprehensive anti-spider protection.
        Uses Google Scholar as primary source, falls back to publisher sites.
        
        :param papers: List of PaperInfo objects
        :param delay: Base delay between requests (increased for anti-spider protection)
        :return: List of PaperInfo objects with abstracts populated
        """
        logger.info(f"Starting batch abstract extraction for {len(papers)} papers")
        
        if self.use_google_scholar:
            logger.info("📚 Using Google Scholar as primary source")
        else:
            logger.info("🏢 Using publisher websites (IEEE/ACM)")
        
        updated_papers = []
        successful_extractions = 0
        failed_extractions = 0
        
        try:
            for i, paper in enumerate(papers):
                logger.info(f"Processing paper {i+1}/{len(papers)}: '{paper.title[:60]}{'...' if len(paper.title) > 60 else ''}'")
                
                start_time = time.time()
                abstract = None
                
                # Try Google Scholar first if enabled
                if self.use_google_scholar:
                    logger.debug("🔍 Trying Google Scholar...")
                    abstract = self.extract_abstract_from_paper_info(paper.title, paper.authors)
                
                # Fall back to publisher sites if Google Scholar fails
                if not abstract:
                    if self.use_google_scholar:
                        logger.debug("🏢 Google Scholar failed, trying publisher sites...")
                    else:
                        logger.debug("🏢 Using publisher sites...")
                    
                    abstract = self.extract_abstract(paper.url)
                
                extraction_time = time.time() - start_time
                
                if abstract:
                    paper.abstract = abstract
                    successful_extractions += 1
                    source = "Google Scholar" if self.use_google_scholar else "Publisher"
                    logger.info(f"✓ Abstract extracted from {source} in {extraction_time:.2f}s (length: {len(abstract)} chars)")
                else:
                    failed_extractions += 1
                    logger.warning(f"✗ Failed to extract abstract after {extraction_time:.2f}s")
                
                updated_papers.append(paper)
                
                # Add randomized delay between requests to avoid rate limiting
                if i < len(papers) - 1:
                    # More conservative delay range
                    actual_delay = random.uniform(delay * 0.8, delay * 1.8)
                    logger.debug(f"Waiting {actual_delay:.2f}s before next request...")
                    time.sleep(actual_delay)
        
        except KeyboardInterrupt:
            logger.warning("Batch extraction interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error during batch extraction: {str(e)}", exc_info=True)
        finally:
            logger.info(f"Batch extraction completed: {successful_extractions} successful, {failed_extractions} failed, {len(updated_papers)} total processed")
            if self.driver:
                self._teardown_driver()
        
        return updated_papers
    
    def __enter__(self):
        """Context manager entry."""
        self._setup_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._teardown_driver()
        # Suppress exceptions from context manager
        return False