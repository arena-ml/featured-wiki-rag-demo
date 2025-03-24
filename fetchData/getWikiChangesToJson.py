#!/usr/bin/env python3
"""
Fetch Wikipedia articles with recent changes in readable format

This script fetches recent changes of Wikipedia articles ,extracts article content,
and formats the data into a structured JSON file.
"""

import argparse
import hashlib
import json
import logging
import requests
import sys
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fetch_wikiArticles")


class ArticlesWithRecentChanges:
    """fetch Wikipedia articles  with recent changes."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize  with configuration settings.

        Args:
            config: Dictionary containing configuration settings
        """
        self.hours = config["hours"]
        self.output_path = config["output_path"]
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.cutoff_time = datetime.now(tz=timezone.utc) - timedelta(hours=self.hours)

    def hash_to_sha512_string(self, article: str) -> Optional[str]:
        """
        Generates a 10-character hash from a given string.

        Args:
            article: The string to hash.

        Returns:
            A 10-character hexadecimal hash string, or None if input is not a string.
        """
        if not isinstance(article, str):
            return None  # Handle non-string input

        hash_object = hashlib.sha512(article.encode())
        hex_digest = hash_object.hexdigest()
        return hex_digest[:10]  # Return the first 10 characters

    # Function to get the date for yesterday
    def todays_date():
        return datetime.now(timezone.utc)
    
    def remove_underscores(self,input_string):
        return input_string.replace("_", " ")
    
    # Function to clean and format text
    def clean_text(self,text: str) -> str:
        text = re.sub(r"==\s*(References|External links)\s*==.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\[[0-9]+\]", "", text)  # Remove citation numbers
        text = re.sub(r"\n{2,}", "\n", text).strip()
        return text
    
    def get_featuredArticlesList(self,date: datetime)->List[str]:
        url = f"https://api.wikimedia.org/feed/v1/wikipedia/en/featured/{date.year}/{date.month:02}/{date.day:02}"
        
        # Add headers to simulate a browser request
        headers = {
            "accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        print(f"Fetching data from: {url}")
        response = requests.get(url, headers=headers)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Failed to fetch data for {date}. Status code: {response.status_code}")
            print(f"Response content: {response.text}")  # Print the response content for debugging
            exit(1)
        
        articlesList = []

        data = response.json()
        try:
            # get articles title
            # todays featured article
            tfa = data.get("tfa","")
            if tfa['type'] == "standard":
                articlesList.append(tfa['title'])
        except Exception as e:
            logger.error(f"Error fetching featured article title: {e}")
            
        try:
            mostReadArticles = data.get("mostread",{})
            for article in mostReadArticles['articles']:
                if article.get('type',"") == "standard":
                    articlesList.append(article['title'])
        except Exception as e:
            logger.error(f"Error fetching most read article titles: {e}")
            
        
        try:
            for news in data['news']:
                links = news.get("links","")
                for link in links:
                    if link.get("type","") == "standard":
                        articlesList.append( link['title'])
        except Exception as e:
            logger.error(f"Error fetching news article titles: {e}")
            
        
        try:
            # on this day related articles
            for otd in data["onthisday"]:
                pages = otd['pages']
                for page in pages:
                        articlesList.append(page['title'])
        except Exception as e:
            logger.error(f"Error fetching on this day articles titles, error: {e}")
            
        print(*articlesList,sep='\n')
        return articlesList
    
    
    def getArticleLists(self)->List[str]:
        """
            To get list of articles titles using featured content or
            most viewed/edited content api.
        """
        articleTitles = []
        tDate = datetime.now(timezone.utc)
        mostViewedArticles = self.get_featuredArticlesList(date=tDate)
        
        articleTitles.extend(mostViewedArticles)

        return articleTitles

    def fetch_article_text(self, page_title: str) -> str:
        """
        Fetches the full content of a Wikipedia article.

        Args:
            page_title: The title of the Wikipedia page.

        Returns:
            The content of the article or an empty string if fetching fails.
        """
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,  # Fetch plain text, no HTML
            "titles": page_title,
            "format": "json",
        }

        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching article text for '{page_title}': {e}")
            exit(1)

        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            return page_data.get("extract", "")  # Extract plain text content

        return ""

    def format_diff(self, diff_html: str) -> Optional[str]:
        """
        Cleans and simplifies the diff content.

        Args:
            diff_html: HTML content of the diff.

        Returns:
            Simplified and cleaned diff or None if no meaningful changes.
        """
        if not diff_html:
            return None
            
        soup = BeautifulSoup(diff_html, "html.parser")

        additions = [ins.get_text() for ins in soup.find_all("ins")]
        deletions = [del_tag.get_text() for del_tag in soup.find_all("del")]
        
        simplified_diff = []

        if additions:
            simplified_diff.append(f"Added: {', '.join(additions[:])}")
            
                
        if deletions:
            simplified_diff.append(f"Removed: {', '.join(deletions[:])}")
            
                

        return "\n".join(simplified_diff) if simplified_diff else None

    
    def get_recent_changes_within_timeframe(
        self, page_title: str, cutoff_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Fetches recent changes made to a Wikipedia page within the specified timeframe.

        Args:
            page_title: The title of the Wikipedia page.
            cutoff_time: Time limit for fetching changes.

        Returns:
            A list of changes with metadata and diffs.
        """
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": page_title,
            "rvprop": "timestamp|comment|ids|user|diff",
            "rvdiffto": "prev",
            "rvlimit": 50,  # Increased limit to catch more changes
            "format": "json",
        }

        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching revisions for '{page_title}': {e}")
            return []

        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        revisions_data = []

        for page_id, page_info in pages.items():
            revisions = page_info.get("revisions", [])
            for rev in revisions:
                timestamp = rev.get("timestamp")
                revision_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
                if revision_time < cutoff_time:
                    continue

                comment = rev.get("comment", "No comment")
                user = rev.get("user", "Anonymous")
                raw_diff = rev.get("diff", {}).get("*", "No diff available")

                clean_diff = self.format_diff(raw_diff)
                if clean_diff:
                    revisions_data.append(
                        {
                            "change_id": f"{page_id}_{rev.get('revid')}",
                            "timestamp": timestamp,
                            "user": user,
                            "change_summary": comment,
                            "diff": clean_diff,
                        }
                    )

        return revisions_data

    def storeArticles(self) -> None:
        """
        Process all articleTitles and generate output.
        """
        dataset = []
        articleTitles = []

        logger.info(f"fetching articles with changes in the last {self.hours} hours")
        
        try:
            logger.info(f"Fetching articles most viewed, most edited and articles linked to wikinews")
            articleTitles = self.getArticleLists()
            logger.info(f"Found {len(articleTitles)} articles ")
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
        
        # Remove duplicates
        articleTitles = list(set(articleTitles))
        logger.info(f"Processing {len(articleTitles)} unique articles")
        
        # Process each article
        for articleTitle in articleTitles:
            try:
                changes = self.get_recent_changes_within_timeframe(articleTitle, self.cutoff_time)
                
                # Only add articles with changes to the dataset
                if changes:
                    article_text = self.clean_text(self.fetch_article_text(articleTitle))

                    article_id = self.hash_to_sha512_string(article_text)
                    formatedTitle = self.remove_underscores(articleTitle)
                    # Treat the entire article as one section
                    article_data = {
                    "article_id": article_id,  # Unique identifier based on article title
                        "title": formatedTitle,
                        "content": {
                            "sections": [
                                {
                                    "section_title": "Main Article",
                                    "text": article_text
                                    if article_text
                                    else f"Could not fetch text for '{articleTitle}'.",
                                    "changes": changes,
                                }
                            ]
                        },
                    }
                    dataset.append(article_data)
                    logger.info(f"Found {len(changes)} recent changes for '{articleTitle}'")
            except Exception as e:
                logger.error(f"Error processing article '{articleTitle}': {e}")
        
        # Generate summary statistics
        if dataset:
            
            # Save the dataset as JSON
            with open(self.output_path, "w") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved data for {len(dataset)} articles with recent changes to {self.output_path}")
        else:
            logger.info("No recent changes found. Dataset not created.")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="fetches Wikipedia articles with recent changes")
    
    
    parser.add_argument(
        "--hours", 
        type=int, 
        default=72,
        help="Number of hours to look back for changes"
    )
    
    parser.add_argument(
        "--output", 
        "-o", 
        default="WikiRC.json",
        help="Output JSON file path"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    
    config = {
        "hours": args.hours,
        "output_path": args.output,
    }
    
    wrc = ArticlesWithRecentChanges(config)
    wrc.storeArticles()


if __name__ == "__main__":
    main()