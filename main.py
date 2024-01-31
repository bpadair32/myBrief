from transformers import BartForConditionalGeneration, BartTokenizer
from bs4 import BeautifulSoup
import requests
import sys


def summarize_article(text: str, maxLength: int = 150) -> str:
    modelName = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(modelName)
    model = BartForConditionalGeneration.from_pretrained(modelName)

    inputs = tokenizer.encode(
        "summarize: " + text, return_tensors="pt", max_length=1024, truncation=True
    )
    summary_ids = model.generate(
        inputs,
        max_length=maxLength,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def scrape_article(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        allText = soup.get_text(separator="\n", strip=True)
        return allText
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return str(e)


def main() -> None:
    articleUrl = sys.argv[1]
    articleText = scrape_article(articleUrl)

    if articleText:
        summary = summarize_article(articleText)
        print("\nSummarized Article:")
        print(summary)
    else:
        print("Scraping failed")


if __name__ == "__main__":
    main()
