# Script that transforms pdfs to hmtl code and then cleans this code until only tables are left and then using LMStudio transforms this tables into json objects

import subprocess
import os
from bs4 import BeautifulSoup, NavigableString
from openai import OpenAI
import json


# Convert pdf to html
def convert_pdf_to_html(pdf_path, html_path):
    command = f"pdf2htmlEX {pdf_path} --dest-dir {html_path} --font-size-multiplier 1 --zoom 25"
    subprocess.call(command, shell=True)


input_pdf = "XS630B1.pdf"
output_pdf = "XS630B1"

convert_pdf_to_html(input_pdf, output_pdf)

# Open the HTML file and read its content
with open(f"{output_pdf}/{output_pdf}.html", "r") as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, "html.parser")

# Find all div elements with a "data-page-no" attribute
div_elements = soup.find_all("div", attrs={"data-page-no": True})

results = ""

# Open the output file in write mode
with open(f"{output_pdf}/processed_{output_pdf}.html", "w") as file:
    # Loop over the div elements
    for div_element in div_elements:
        # Find and remove all img elements within the div
        for img in div_element.find_all("img"):
            img.decompose()

        # Find and remove all div elements that don't contain any text
        for div in div_element.find_all("div"):
            if not div.find_all(string=lambda text: isinstance(text, NavigableString)):
                div.decompose()

        # Convert the div element back to string
        result_div = str(div_element)

        # Append the result to variable
        results = results + result_div

        # Write the result to the file
        file.write(results + "\n")

local_ip = "172.31.48.1"
client = OpenAI(base_url=f"http://{local_ip}:5000/v1", api_key="not-needed")

print("Local client created")
print("sending request")
response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    stream=True,
    temperature=0.0,
    messages=[
        {
            "role": "system",
            "content": "Good morning, you are a helpful assistant and an expert web developer. Some tables were converted into the HTML code inside triple quotes. Please turn the code into several JSON objects that represent the original tables. only return the json objects, no additional commentary or content.",
        },
        {
            "role": "user",
            "content": '"""\n' + results + '"""\n',
        },
    ],
)

# Print stream
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# # Print whole result
# print(response.choices[0].message.content)


# # a JSON string
# json_string = response.choices[0].message.content

# # convert string to JSON
# json_obj = json.loads(json_string)

# # pretty print JSON
# pretty_json = json.dumps(json_obj, indent=4)
# print(pretty_json)