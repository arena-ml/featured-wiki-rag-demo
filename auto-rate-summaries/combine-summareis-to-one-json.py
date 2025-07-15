import os
import json
import re
import sys
import logging

def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def get_json_file(file_path):
    print("File path", file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            articles = json.load(file)
            return articles
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON. Check file format.")
        sys.exit(1)

def merge_json_files(json_files_paths):
    """
    Advanced version with customizable summary key detection patterns.
    """
    merged_articles = {}
    summary_keys = set()

    # Default patterns for summary key detection
    default_patterns = [
        r'.*summary.*',
        r'.*Summary.*',
    ]

    patterns = default_patterns

    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    # First pass: discover summary keys
    print("Discovering summary keys with patterns...")
    for file_path in json_files_paths:
        try:
            articles = get_json_file(file_path)
            for article in articles:
                if not isinstance(article, dict):
                    continue

                for key in article.keys():
                    # Skip standard article fields
                    if key in ['article_id', 'title', 'content', 'summaries']:
                        continue

                    # Check if key matches any summary pattern
                    if any(pattern.match(key) for pattern in compiled_patterns):
                        summary_keys.add(key)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    summary_keys = sorted(list(summary_keys))
    print(f"Found summary keys: {summary_keys}")

    # Second pass: merge articles (same as before)
    for file_path in json_files_paths:
        try:
            articles = get_json_file(file_path)

            for article in articles:
                if not isinstance(article, dict):
                    continue

                article_id = article.get("article_id")
                if not article_id:
                    continue

                if article_id not in merged_articles:
                    merged_articles[article_id] = {
                        "article_id": article_id,
                        "title": article.get("title", ""),
                        "content": article.get("content", {}),
                        "summaries": {key: "" for key in summary_keys},
                    }

                summaries = merged_articles[article_id]["summaries"]
                for key in summary_keys:
                    if key in article and article[key]:
                        summaries[key] = article[key]

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return merged_articles


if __name__ == "__main__":
    current_dir = os.getcwd()

    directory_path = current_dir

    json_file_paths = find_json_files(directory_path)

    merged_file = merge_json_files(json_file_paths)

    merged_articles_list = list(merged_file.values())

    # Save to a new JSON file
    with open("merged_articles.json", "w", encoding="utf-8") as out_file:
        json.dump(merged_articles_list, out_file, indent=2, ensure_ascii=False)
