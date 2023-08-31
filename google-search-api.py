import requests
import json

def google_search(api_key, query, cx):
    # Define the endpoint URL
    url = "https://www.googleapis.com/customsearch/v1"

    # Define the parameters for the API request
    params = {
        "key": api_key,
        "q": query,
        "cx": cx,
    }

    # Make the API request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        results = json.loads(response.text)

        # Print out the search results
        for i, item in enumerate(results["items"], start=1):
            print(f"{i}. {item['title']}\n   {item['link']}\n")

    else:
        print("Failed to make the API request.")
        print(response.text)

# Your API Key
API_KEY = "AIzaSyDDlpAt1gs1XHhaR9SoeYjslz4p3eB5WDU"

# Your Custom Search Engine ID
CX = "72dc43fe854b848c0"

# Query you want to search
QUERY = "Python programming"

# Perform the search
google_search(API_KEY, QUERY, CX)
