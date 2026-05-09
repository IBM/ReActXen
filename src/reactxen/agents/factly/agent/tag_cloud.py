# Benchmark Dataset Word Cloud Generator (2023–2024)

# pip install openreview-py wordcloud tqdm beautifulsoup4 requests arxiv

import re
import openreview
import requests
import arxiv
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

KEYWORDS = [
    "dataset", "benchmark", "evaluation", "corpus", "testbed", "data collection", 
    "benchmarking", "large-scale dataset", "benchmark suite", "labelled data"
]

def is_relevant(title, abstract):
    content = (title + " " + abstract).lower()
    return any(kw in content for kw in KEYWORDS)

def clean_text(texts):
    all_text = ' '.join(texts).lower()
    tokens = re.findall(r'\b[a-z]{3,}\b', all_text)
    stopwords = set(WordCloud().stopwords)
    return ' '.join(w for w in tokens if w not in stopwords)

def make_wordcloud(text):
    wc = WordCloud(width=1600, height=800, background_color='white').generate(text)
    plt.figure(figsize=(20,10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def show_top_words(text, top_n=20):
    tokens = text.split()
    freq = Counter(tokens).most_common(top_n)
    words, counts = zip(*freq)
    plt.figure(figsize=(12,6))
    plt.bar(words, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Top Words in Dataset/Benchmark Papers")
    plt.show()

def fetch_openreview(conference_id):
    client = openreview.Client(baseurl='https://api.openreview.net')
    notes = client.get_notes(invitation=conference_id)
    print(f"Fetched {len(notes)} papers from {conference_id}")
    return [(n.content['title'], n.content.get('abstract', '')) for n in notes]

def fetch_iclr2024():
    client = openreview.Client(baseurl='https://api.openreview.net')
    posters = client.get_notes(invitation='ICLR.cc/2024/Conference/-/Poster')
    orals = client.get_notes(invitation='ICLR.cc/2024/Conference/-/Oral')
    spotlights = client.get_notes(invitation='ICLR.cc/2024/Conference/-/Spotlight')
    all_notes = posters + orals + spotlights
    print('ccccc--------')
    print(f"Fetched {len(all_notes)} ICLR 2024 accepted papers")
    return [(n.content['title'], n.content.get('abstract', '')) for n in all_notes]

def fetch_icml_papers(volume_url):
    base = "https://proceedings.mlr.press"
    r = requests.get(volume_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    papers = []
    
    for div in soup.select("div.paper"):
        title = div.find("p", class_="title").text.strip()
        abstract_link = base + div.find("a", text="abs")["href"]
        abs_page = requests.get(abstract_link)
        abs_soup = BeautifulSoup(abs_page.text, "html.parser")
        abstract = abs_soup.find("div", class_="abstract").text.strip().replace("Abstract: ", "")
        papers.append((title, abstract))
    
    return papers

import openreview
import requests
from bs4 import BeautifulSoup

def fetch_iclr2024():
    client = openreview.Client(baseurl='https://api.openreview.net')
    posters = client.get_notes(invitation='ICLR.cc/2024/Conference/-/Poster')
    orals = client.get_notes(invitation='ICLR.cc/2024/Conference/-/Oral')
    spotlights = client.get_notes(invitation='ICLR.cc/2024/Conference/-/Spotlight')
    all_notes = posters + orals + spotlights
    print(f"Fetched {len(all_notes)} ICLR 2024 accepted papers")
    return [(n.content['title'], n.content.get('abstract', '')) for n in all_notes]

def fetch_icml2024():
    client = openreview.Client(baseurl='https://api.openreview.net')
    posters = client.get_notes(invitation='ICML.cc/2024/Conference/-/Poster')
    orals = client.get_notes(invitation='ICML.cc/2024/Conference/-/Oral')
    spotlights = client.get_notes(invitation='ICML.cc/2024/Conference/-/Spotlight')
    all_notes = posters + orals + spotlights
    print(f"Fetched {len(all_notes)} ICML 2024 accepted papers")
    return [(n.content['title'], n.content.get('abstract', '')) for n in all_notes]

def fetch_neurips(year=2023):
    url = f"https://papers.neurips.cc/paper_files/paper/{year}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    papers = soup.find_all("li")
    result = []
    for paper in papers:
        a_tag = paper.find("a")
        if a_tag:
            title = a_tag.text.strip()
            link = "https://papers.neurips.cc" + a_tag["href"]
            result.append((title, link))
    print(f"Fetched {len(result)} NeurIPS {year} papers")
    return result

def fetch_aaai2024():
    url = "https://ojs.aaai.org/index.php/AAAI/issue/view/651"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    entries = soup.select(".title a")
    results = []
    for entry in entries:
        title = entry.text.strip()
        link = entry["href"]
        abs_resp = requests.get(link)
        abs_soup = BeautifulSoup(abs_resp.text, "html.parser")
        abstract = abs_soup.select_one(".abstract").text.strip() if abs_soup.select_one(".abstract") else ""
        results.append((title, abstract))
    print(f"Fetched {len(results)} AAAI 2024 papers")
    return results

def fetch_ijcai2023():
    url = "https://ijcai.org/proceedings/2023/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    entries = soup.select("li.paper")
    results = []
    for entry in entries:
        title = entry.select_one("a").text.strip()
        link = "https://ijcai.org" + entry.select_one("a")["href"]
        results.append((title, link))
    print(f"Fetched {len(results)} IJCAI 2023 papers")
    return results


def fetch_aaai_arxiv(year):
    search = arxiv.Search(
        query=f"all:dataset OR benchmark AND comment:AAAI {year}",
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    return [(r.title, r.summary) for r in search.results()]

if __name__ == "__main__":
    fetch_neurips(2024)
    exit(0)
    iclr2023 = fetch_openreview('ICLR.cc/2023/Conference/-/Blind_Submission')
    iclr2024 = fetch_iclr2024()
    icml2023 = fetch_icml_papers("https://proceedings.mlr.press/v202/")
    icml2024 = fetch_icml_papers("https://proceedings.mlr.press/v224/")
    aaai2023 = fetch_aaai_arxiv(2023)
    aaai2024 = fetch_aaai_arxiv(2024)

    neurips_sample = [
        ("A New Benchmark for 3D Scene Understanding", "We present a new dataset for..."),
        ("Self-Supervised Dataset Curation", "We evaluate methods for dataset creation and curation..."),
    ]

    all_sources = iclr2023 + iclr2024 + neurips_sample + icml2023 + icml2024 + aaai2023 + aaai2024
    relevant = [(t, a) for t, a in all_sources if is_relevant(t, a)]

    print(f"Found {len(relevant)} dataset/benchmark papers")

    text_blob = clean_text([title + " " + abstract for title, abstract in relevant])
    make_wordcloud(text_blob)
    show_top_words(text_blob)
