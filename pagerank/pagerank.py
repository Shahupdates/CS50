import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    model = dict()
    links = corpus[page]

    # If page has no outgoing links, return equal probability for every page
    if len(links) == 0:
        for page in corpus:
            model[page] = 1 / len(corpus)
        return model

    # Calculate probabilities
    for link in corpus:
        model[link] = (1 - damping_factor) / len(corpus)
        if link in links:
            model[link] += damping_factor / len(links)

    return model

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = {page: 0 for page in corpus}
    sample = random.choice(list(corpus.keys()))

    for i in range(n):
        model = transition_model(corpus, sample, damping_factor)
        sample = random.choices(
            population=list(model.keys()),
            weights=list(model.values()),
            k=1
        )[0]
        pageranks[sample] += 1

    # Normalize the counts to sum to 1
    pageranks = {page: count / n for page, count in pageranks.items()}
    return pageranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = {page: 1 / len(corpus) for page in corpus}
    new_pageranks = pageranks.copy()

    # Loop until convergence
    while True:
        for page in pageranks:
            total = float(0)
            for possible_page in corpus:
                # Check all pages linking to current page
                if page in corpus[possible_page]:
                    total += pageranks[possible_page] / len(corpus[possible_page])
                # A page that has no links is interpreted as having one link for every page in the corpus
                elif len(corpus[possible_page]) == 0:
                    total += pageranks[possible_page] / len(corpus)
            new_pageranks[page] = (1 - damping_factor) / len(corpus) + damping_factor * total

        # Check convergence, i.e., | PR(p) - newPR(p) | < 0.001 for all p
        if all(abs(new_pageranks[page] - pageranks[page]) < 0.001 for page in pageranks):
            break

        pageranks = new_pageranks.copy()

    return pageranks


if __name__ == "__main__":
    main()
