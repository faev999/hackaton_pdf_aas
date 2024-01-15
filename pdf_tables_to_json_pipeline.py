# Script that transforms pdfs to hmtl code and then cleans this code until only tables are left and then using gpt4 transforms this tables into json objects

import subprocess
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import VectorStoreIndex
from llama_index.readers.file.flat_reader import FlatReader
from pathlib import Path
import os
import pickle
from llama_index.node_parser import (
    UnstructuredElementNodeParser,
)
from bs4 import BeautifulSoup, NavigableString
from openai import OpenAI
import json


# Convert pdf to html
def convert_pdf_to_html(pdf_path, html_path):
    command = f"pdf2htmlEX {pdf_path} --dest-dir {html_path} --font-size-multiplier 1 --zoom 25"
    subprocess.call(command, shell=True)


input_pdf = "XS630B1MAL2_document.pdf"
output_pdf = "XS630B1MAL2_document"

convert_pdf_to_html(input_pdf, output_pdf)

# Open the HTML file and read its content

with open(f"{output_pdf}/{output_pdf}.html", "r") as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, "html.parser")

# Prettify the HTML content
prettified_html = soup.prettify()

# Remove only head element from soup
soup_without_head = soup.find("body")

# Remove img elements
for img in soup_without_head.find_all("img"):
    img.decompose()

# Convert  soup back to string so it can be used by the LLM
result = str(soup_without_head)

# Prettify soup without head
# TODO: is this better than unprettified soup?
prettified_soup_without_head = soup_without_head.prettify()


# Save to html file
with open(f"{output_pdf}/no_img.html", "w") as file:
    file.write(str(prettified_soup_without_head))

# # Save the prettified HTML to a file
# with open(f"{output_pdf}/prettified_html.html", "w") as file:
#     file.write(prettified_html)


# # Find all div elements with a "data-page-no" attribute
# div_elements = soup.find_all("div", attrs={"data-page-no": True})

# # Open the output file in write mode
# with open("output3.html", "w") as file:
#     # Loop over the div elements
#     for div_element in div_elements:
#         # Find and remove all img elements within the div
#         for img in div_element.find_all("img"):
#             img.decompose()

#         # Find and remove all div elements that don't contain any text
#         for div in div_element.find_all("div"):
#             if not div.find_all(string=lambda text: isinstance(text, NavigableString)):
#                 div.decompose()

#         # Convert the div element back to string
#         result = str(div_element)

#         # Write the result to the file
#         file.write(result + "\n")

client = OpenAI()

print("OpenAI client created")
print("sending request")
response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
            "role": "system",
            "content": "Good morning, you are a helpful assistant and an expert web developer. Some tables were converted into the HTML code inside triple quotes. Please turn the code into several JSON objects that represent the original tables. only return the json objects, no additional commentary or content.",
        },
        {
            "role": "user",
            "content": '"""\n' + result + '"""\n',
        },
    ],
)
print(response.choices[0].message.content)

# # a JSON string
# json_string = response.choices[0].message.content

# # convert string to JSON
# json_obj = json.loads(json_string)

# # pretty print JSON
# pretty_json = json.dumps(json_obj, indent=4)
# print(pretty_json)
