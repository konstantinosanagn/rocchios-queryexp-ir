# Information Retrieval System with User Feedback using Google Search API

In this repo, you will find the following files:
- search.py: implementation of IR system
- requirements.txt: required modules for running the program
- .gitignore: ignores `venv` files generated, prevents pushing unrelated files to repo
- transcript.txt: runs of our program on the 3 test cases (per se, wojcicki, milky way)

### How To Run
1. Clone this repository 
```
git clone <url>
```
2. Create a virtual environment

MacOS/Linux:
```
python3 -m venv venv
```
Windows:
```
python -m venv venv
```
3. Activate the virtual environment

MacOS/Linux:
```
source venv/bin/activate
```
Windows:
```
venv\Scripts\activate
```
4. Install dependencies from `requirements.txt` 
```
pip install -r requirements.txt
```
5. Run the program

MacOS/Linux:
```
python3 search.py <precision> "<query>"
```
Windows:
```
python search.py <precision> "<query>"
```
\* Note: `query` must be wrapped inside double-quotation marks

### High-level Description
#### Code Structure
- `CustomSearchEngine` Class: class that handles search, feedback collection, and query augmentation + refinement
  - Attributes
    - `self.api_key`: Google API key for fetching query results
    - `self.engine_id`: Google Search Engine ID
    - `self.original_query`: original query used in first iteration -- used for reordering query terms 
    - `self.query`: current query
    - `self.precision`: target search precision
  - Methods
    - `fetch_results()`: calls Google Custom Search API and retrieves top 10 search results
    - `get_relevance_feedback(results) `: displays search results and gathers user feedback (Y/N). Returns lists of relevant and non-relevant documents based on the response
    - `rocchio_algorithm(relevant_docs, non_relevant_docs)`: implements Rocchio's algorithm with TF-IDF weighting for query expansion. Returns a list of weighted keywords
    - `refine_query()`: appends at most top 2 weighted words from Rocchio's algorithm. Reorders query based on the weights in the current iteration
#### External Libraries
- `googleapiclient.discovery` -- fetches search results
- `nltk.corpus.stopwords` -- filters out common words
- `re` -- regular expression for extracting key terms
- `collections.Counter ` -- ranks query terms by frequency
- `sys` -- extracts arguments from terminal 

### Query Modification Method
This project enhances search precision using Rocchio’s algorithm, refining queries based on user feedback. The process consists of three key steps:
1. Collecting User Feedback
- The system fetches top-10 search results using Google’s Custom Search API
- Based on user feedback, results are categorized into:
  - Relevant documents (relevant_docs)
  - Non-relevant documents (non_relevant_docs)
2. Applying Rocchio's Algorithm
- Once feedback is collected, Rocchio's algorithm adjusts term importance with following parameter values:
  - alpha = 1.0 -- keeps original query terms
  - beta = 0.75 -- boosts relevant document terms
  - gamma = 0.15 -- reduces non-relevant document terms
3. Selecting and Updating Query Terms
- Extracts words from titles and snippets of search results
- Calculates term weights with Rocchio's and TF-IDF, and ranks them by their weights
- Filters out stopwords and existing query terms
- Appends up to two new words with the highest weight
- Reorders query terms based on their weights, minus the original query (always placed first)

### Environment Variables (API Key & Engine ID)
API Key = 'GOOGLE_SEARCH_API_KEY'
Engine ID = 'SEARCH_ENGINE_ID'
