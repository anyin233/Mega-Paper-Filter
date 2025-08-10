from typing import List
import requests
from loguru import logger
from pydantic import BaseModel


class VenueInfo(BaseModel):
    """
    A Pydantic model representing a venue.
    """
    name: str
    acronym: str
    url: str
    
class ConferenceInfo(BaseModel):
    """
    A Pydantic model representing a conference.
    """
    name: str
    url: str
    year: int
    
class PaperInfo(BaseModel):
    """
    A Pydantic model representing a paper.
    """
    title: str
    doi: str
    url: str
    authors: List[str]
    year: int
    type: str = ""
    abstract: str = ""

BASE_URL = "https://dblp.org/"
PUBLICATION_PATH = "search/publ/api"
VENUE_PATH = "search/venue/api"

def query_venue(venue_name):
    """
    Query the DBLP API for a specific venue.
    
    :param venue_name: The name of the venue to query.
    :return: A dictionary containing the venue information or an error message.
    """
    params = {
        'q': venue_name,
        'format': 'json'
    }
    
    response = requests.get(f"{BASE_URL}{VENUE_PATH}", params=params)
    
    response_data = response.json()['result']
    if response.status_code == 200:
        return response_data['hits']
    else:
        logger.error("Failed to retrieve data from DBLP API.")
        logger.error(f"Error {response.status_code}: {response.text}")
        return {"error": "Failed to retrieve data from DBLP API."}
      
def get_candidcate_venues(venue_name) -> List[VenueInfo]:
    """
    Get candidate venues from the DBLP API based on a given venue name.
    
    :param venue_name: The name of the venue to search for.
    :return: A list of candidate venues or an error message.
    """
    result = query_venue(venue_name)
    
    if 'error' in result:
        return result['error']
    
    candidates = []
    for hit in result['hit']:
        candidates.append(
            VenueInfo(
                name=hit['info']['venue'],
                url=hit['info']['url'],
                acronym=hit['info'].get('acronym', 'N/A')
            )
        )
    
    return candidates
  
def get_paper_list(venue_name: str, year: int, filter_conference_papers: bool = True) -> List[PaperInfo]:
    """
    Get a list of papers from a specific venue and year.

    :param venue_name: The name of the venue.
    :param year: The year of the conference.
    :param filter_conference_papers: If True, only return "Conference and Workshop Papers"
    :return: A list of PaperInfo objects containing paper details.
    """
    params = {
        'q': f'{venue_name}$ {year}',
        'format': 'json',
        'h': 1000,  # Adjust the number of hits to retrieve as needed
        'f': 0
    }
    
    cur_first = 0

    response = requests.get(f"{BASE_URL}{PUBLICATION_PATH}", params=params)
    response_data = response.json()['result']

    if response.status_code == 200:
        response_body = response_data
        total_hits = int(response_body['hits']['@total'])
        cur_first += int(response_body['hits']['@sent'])
        
        logger.info(f"Total hits for {venue_name} in {year}: {total_hits}")
        papers = []
        filtered_count = 0
        
        for hit in response_body['hits']['hit']:
            paper_info = hit['info']
            paper_type = paper_info.get('type', '')
            
            # Filter for conference papers if requested
            if filter_conference_papers and paper_type != "Conference and Workshop Papers":
                filtered_count += 1
                logger.debug(f"Filtered out paper with type '{paper_type}': {paper_info.get('title', 'Unknown')}")
                continue
            
            authors = paper_info.get('authors', {}).get('author', [])
            if isinstance(authors, dict):
                authors = [authors]
            
            papers.append(
                PaperInfo(
                    title=paper_info['title'],
                    doi=paper_info.get('doi', 'N/A'),
                    url=paper_info.get('ee', 'N/A'),
                    authors=[author['text'] for author in authors],
                    year=int(paper_info['year']),
                    type=paper_type
                )
            )
        
        while cur_first < total_hits:
            params['f'] = cur_first
            response = requests.get(f"{BASE_URL}{PUBLICATION_PATH}", params=params)
            if response.status_code == 200:
                response_body = response.json()
                for hit in response_body['hits']['hit']:
                    paper_info = hit['info']
                    paper_type = paper_info.get('type', '')
                    
                    # Filter for conference papers if requested
                    if filter_conference_papers and paper_type != "Conference and Workshop Papers":
                        filtered_count += 1
                        logger.debug(f"Filtered out paper with type '{paper_type}': {paper_info.get('title', 'Unknown')}")
                        continue
                    
                    authors = paper_info.get('authors', {}).get('author', [])
                    if isinstance(authors, dict):
                        authors = [authors]
                    
                    papers.append(
                        PaperInfo(
                            title=paper_info['title'],
                            doi=paper_info.get('doi', 'N/A'),
                            authors=[author['text'] for author in authors],
                            year=int(paper_info['year']),
                            url=paper_info.get('ee', 'N/A'),
                            type=paper_type
                        )
                    )
                cur_first += int(response_body['hits']['@sent'])
            else:
                logger.error("Failed to retrieve additional data from DBLP API.")
                break
        
        if filter_conference_papers:
            logger.info(f"Filtered out {filtered_count} non-conference papers. Returning {len(papers)} conference papers.")
        else:
            logger.info(f"Returning {len(papers)} papers of all types.")
        
        return papers
    else:
        logger.error("Failed to retrieve data from DBLP API.")
        logger.error(f"Error {response.status_code}: {response.text}")
        return []

def get_paper_type_statistics(venue_name: str, year: int) -> dict:
    """
    Get statistics about paper types for a venue and year.

    :param venue_name: The name of the venue.
    :param year: The year of the conference.
    :return: Dictionary with type counts and examples.
    """
    papers = get_paper_list(venue_name, year, filter_conference_papers=False)
    
    type_stats = {}
    for paper in papers:
        paper_type = paper.type or "Unknown"
        if paper_type not in type_stats:
            type_stats[paper_type] = {
                'count': 0,
                'examples': []
            }
        type_stats[paper_type]['count'] += 1
        if len(type_stats[paper_type]['examples']) < 3:
            type_stats[paper_type]['examples'].append(paper.title)
    
    return type_stats
        