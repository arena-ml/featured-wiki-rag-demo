import os
import json
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
    merged_articles = {}

    for file_path in json_files_paths:
        articles = get_json_file(file_path)

        for article in articles:
            print("Type:", type(article))
            if type(article) != dict:
                continue
            article_id = article.get("article_id")

            if article_id not in merged_articles:
                # Copy the structure excluding summaries
                merged_articles[article_id] = {
                    "article_id": article_id,
                    "title": article.get("title", ""),
                    "content": article.get("content", {}),
                    "summaries": {
                        "llm1embResponse": "",
                        "llm1oneShotResponse": "",
                        "llm2oneShotResponse": "",
                    },
                }

            # Update summaries if present
            summaries = merged_articles[article_id]["summaries"]
            for key in [
                "llm1embResponse",
                "llm1oneShotResponse",
                "llm2oneShotResponse",
            ]:
                if article.get(key):
                    summaries[key] = article[key]

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
