#!/usr/bin/env python3
"""
Fetch Wikipedia articles with recent changes in readable format and store in json file
"""

import argparse
import hashlib
import json
import logging
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Any, Set

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fetch_wikiArticles")

# Configure session for reuse
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
})


class ArticlesWithRecentChanges:
    """Fetch Wikipedia articles with recent changes."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration settings.

        Args:
            config: Dictionary containing configuration settings
        """
        self.hours = config["hours"]
        self.output_path = config["output_path"]
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.cutoff_time = datetime.now(tz=timezone.utc) - timedelta(hours=self.hours)
        self.max_workers = config.get("max_workers", 10)
        self.max_articles = config.get("max_articles",10)

    # Slightly improves performance (since no self binding is needed: Prevents unnecessary access to the class instance)
    @staticmethod
    def hash_to_sha512_string(article: str) -> Optional[str]:
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

    @staticmethod
    def todays_date():
        """Get today's date in UTC."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def remove_underscores(input_string):
        """Replace underscores with spaces in a string."""
        return input_string.replace("_", " ")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and format article text.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        # Compile regex patterns once for better performance
        ref_pattern = re.compile(r"==\s*(References|External links)\s*==.*", re.DOTALL)
        citation_pattern = re.compile(r"\[[0-9]+\]")
        newline_pattern = re.compile(r"\n{2,}")
        
        # Apply patterns
        text = ref_pattern.sub("", text)
        text = citation_pattern.sub("", text)
        text = newline_pattern.sub("\n", text).strip()
        return text

    def get_featuredArticlesList(self, date: datetime) -> List[str]:
        """
        Get list of featured, most read, news and on-this-day articles.
        
        Args:
            date: Date to fetch featured articles for.
            
        Returns:
            List of article titles.
        """
        url = f"https://api.wikimedia.org/feed/v1/wikipedia/en/featured/{date.year}/{date.month:02}/{date.day:02}"
        
        # Add headers to simulate a browser request
        headers = {"accept": "application/json"}
        
        logger.info(f"Fetching data from: {url}")
        try:
            response = session.get(url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data for {date}. Error: {e}")
            return []
        
        articlesList = []
        data = response.json()
        
        # Process today's featured article
        try:
            tfa = data.get("tfa", "")
            if tfa and tfa.get('type') == "standard":
                articlesList.append(tfa['title'])
        except Exception as e:
            logger.error(f"Error fetching featured article title: {e}")
            
        # Process most read articles
        try:
            mostReadArticles = data.get("mostread", {})
            for article in mostReadArticles.get('articles', []):
                if article.get('type', "") == "standard":
                    articlesList.append(article['title'])
        except Exception as e:
            logger.error(f"Error fetching most read article titles: {e}")
            
        # Process news articles
        try:
            for news in data.get('news', []):
                links = news.get("links", [])
                for link in links:
                    if link.get("type", "") == "standard":
                        articlesList.append(link['title'])
        except Exception as e:
            logger.error(f"Error fetching news article titles: {e}")
            
        # Process on this day articles
        try:
            for otd in data.get("onthisday", []):
                pages = otd.get('pages', [])
                for page in pages:
                    articlesList.append(page['title'])
        except Exception as e:
            logger.error(f"Error fetching on this day articles titles, error: {e}")
            
        logger.info(f"Found {len(articlesList)} featured articles")
        return articlesList
    
    def getArticleLists(self) -> Set[str]:
        """
        Get list of articles titles using featured content or
        most viewed/edited content API.
        
        Returns:
            Set of unique article titles.
        """
        tDate = self.todays_date()
        mostViewedArticles = self.get_featuredArticlesList(date=tDate)
        
        # Using a set for automatic deduplication
        return set(mostViewedArticles)

    @lru_cache(maxsize=128)
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
            response = session.get(self.api_url, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching article text for '{page_title}': {e}")
            return ""

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
            response = session.get(self.api_url, params=params)
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
        
    def process_article(self, article_title: str) -> Optional[Dict[str, Any]]:
        """
        Process a single article and collect its changes.
        
        Args:
            article_title: Title of the article to process.
            
        Returns:
            Article data dictionary if changes were found, None otherwise.
        """
        try:
            changes = self.get_recent_changes_within_timeframe(article_title, self.cutoff_time)
            
            # Only return articles with changes
            if changes:
                article_text = self.clean_text(self.fetch_article_text(article_title))
                article_id = self.hash_to_sha512_string(article_text)
                formatted_title = self.remove_underscores(article_title)
                
                # Treat the entire article as one section
                article_data = {
                    "article_id": article_id,
                    "title": formatted_title,
                    "content": {
                        "sections": [
                            {
                                "section_title": "Main Article",
                                "text": article_text if article_text else f"Could not fetch text for '{article_title}'.",
                                "changes": changes,
                            }
                        ]
                    },
                }
                logger.info(f"Found {len(changes)} recent changes for '{article_title}'")
                return article_data
        except Exception as e:
            logger.error(f"Error processing article '{article_title}': {e}")
        
        return None

    def storeArticles(self) -> None:
        """
        Process all articleTitles and generate output using parallel processing.
        """
        dataset = []
        
        logger.info(f"Fetching articles with changes in the last {self.hours} hours")
        
        try:
            logger.info("Fetching most viewed, most edited and articles linked to wikinews")
            
            article_titles = self.getArticleLists()

            logger.info(f"Processing {len(article_titles)} unique and randomly choosen articles")
        except Exception as e:
            logger.error(f"Error fetching articles titles: {e}")
            article_titles = set()
        
        # Process articles in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_article = {
                executor.submit(self.process_article, title): title for title in article_titles
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_article):
                article_title = future_to_article[future]
                try:
                    result = future.result()
                    if result:
                        dataset.append(result)
                except Exception as e:
                    logger.error(f"Error processing article '{article_title}': {e}")
        
        # Generate summary statistics
        if dataset:
            if len(article_titles) > self.max_articles:
                atSet = random.sample(dataset,self.max_articles)
                dataset = atSet
            # Save the dataset as JSON
            with open(self.output_path, "w") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved data for {len(dataset)} articles with recent changes to {self.output_path}")
        else:
            logger.info("No recent changes found. Dataset not created.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetches Wikipedia articles with recent changes")
    
    parser.add_argument(
        "--hours", 
        type=int, 
        default=72,
        help="Number of hours to look back for changes"
    )
    
    parser.add_argument(
        "--output", 
        "-o", 
        default="WikiRC_StepOne.json",
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Maximum number of threads to use for parallel processing"
    )

    parser.add_argument(
        "--articles",
        type=int,
        default=10,
        help="max number of articles to process, articles will be choosen randomly"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    
    config = {
        "hours": args.hours,
        "output_path": args.output,
        "max_workers": args.threads,
        "max_articles":args.articles,
    }
    
    wrc = ArticlesWithRecentChanges(config)
    wrc.storeArticles()


if __name__ == "__main__":
    main()