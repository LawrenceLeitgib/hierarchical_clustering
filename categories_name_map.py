import requests
import json
import re
from bs4 import BeautifulSoup

# URL of the arXiv category taxonomy
url = 'https://arxiv.org/category_taxonomy'

# Fetch the taxonomy page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')


# Initialize dictionaries for mappings
tag_to_info = {}


# Find all group sections
group = soup.find_all('div', class_='large-data-list')[1]


print(len(group))

# Extract the group name (main category)
group_name = group.find_all('h2')

print(group_name)

# Find all archive sections within the group
groups = group.find_all('div', class_='accordion-body')


for g,name in zip(groups,group_name):
    subCategories = g.find_all('h4')
    print(name.text)
    print("-----------------------------------")
    for s in subCategories:
        match = re.match(r"^(.*?)\s+\((.*?)\)$", s.text)

        if match:
            v1 = match.group(1)
            v2 = match.group(2)
            print(f"v1 = {v1}, v2 = {v2}")
        else:
            print(f"Failed to parse: {s}")



        
        tag_to_info[v1]=(v2,name.text)


   


# Example usage:
print(tag_to_info['cs.AI'])  # Output: Artificial Intelligence

# Save the dictionary to a JSON file

with open('resources/categories_name_map.json', 'w') as f:
    json.dump(tag_to_info, f, indent=4)

