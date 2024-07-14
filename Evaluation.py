import csv
from serpapi import GoogleSearch

params = {
  "engine": "google",
  "q": "Fresh Bagels",
  "location": "Seattle-Tacoma, WA, Washington, United States",
  "hl": "en",
  "gl": "us",
  "google_domain": "google.com",
  "num": "10",
  "start": "10",
  "safe": "active",
  "api_key": "dfd45aee8aedb0c6ef516d3856413f0cfd5db84bfe287920fefd11428095c474"
}

search = GoogleSearch(params)
results = search.get_dict()
organic_results = results["organic_results"]

# Get all unique field names from the organic_results
fieldnames = set()
for result in organic_results:
    fieldnames.update(result.keys())

# Convert the set to a sorted list
fieldnames = sorted(fieldnames)

# Write the results to a CSV file
with open('organic_results.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(organic_results)

print("Results have been saved to organic_results.csv")
