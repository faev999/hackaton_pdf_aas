# MIT License
#
# Copyright (c) [year] [fullname]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Script that transforms pdfs to html code, then cleans this code of style components, and then using gpt4 transforms this code into json objects that represent tables in the pdf.

# Author: Fabio Espinosa, fabio.espinosa(at)dfki.de

import subprocess
import time
from bs4 import BeautifulSoup, NavigableString
from openai import OpenAI
import json
import tiktoken
import os


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Calculate number of tokens in a string depending on the model"""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def convert_pdf_to_html(pdf_path, html_path):
    """Converts a PDF to HTML."""
    command = f"pdf2htmlEX {pdf_path} --dest-dir {html_path} --font-size-multiplier 1 --zoom 25"
    subprocess.call(command, shell=True)


def preprocess_html(output_path, file_name):
    """Cleans HTML code from images, styles and other elements"""
    # Open the HTML file and read its content
    with open(f"{output_path}/{file_name}.html", "r") as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all div elements with a "data-page-no" attribute
    div_elements = soup.find_all("div", attrs={"data-page-no": True})

    complete_results = ""
    results_as_array = []
    # Open the output file in write mode
    with open(f"{output_path}/processed_{file_name}.html", "w") as file:
        # Loop over the div elements
        for div_element in div_elements:
            # Find and remove all img elements within the div
            for img in div_element.find_all("img"):
                img.decompose()

            # Find and remove all div elements that don't contain any text
            for div in div_element.find_all("div"):
                if not div.find_all(
                    string=lambda text: isinstance(text, NavigableString)
                ):
                    div.decompose()
                # Check if the div has a class style attribute
                elif "class" in div.attrs:
                    # Remove the class attribute
                    del div.attrs["class"]

            # Convert the div element back to string
            result_div = str(div_element)
            results_as_array.append(result_div)
            # Append the result to variable
            complete_results = complete_results + result_div

            # Write the result to the file
        file.write(complete_results + "\n")
        return complete_results, results_as_array


def run_inference(query: str, llm_model: str):
    """Sends query to openai and returns the response"""
    if llm_model == "local-model":
        local_ip = "172.31.48.1"
        client = OpenAI(base_url=f"http://{local_ip}:5000/v1", api_key="not-needed")
    else:
        client = OpenAI()
    print("LLM client created")
    print(
        "Number of tokens to send: ",
        num_tokens_from_string(query, "gpt-4-1106-preview"),
    )
    print("sending request")
    response = client.chat.completions.create(
        model=llm_model,
        stream=True,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Good morning, you are a helpful assistant and an expert web developer. Some tables were converted into the HTML code that's inside triple quotes. Please turn that code into several JSON objects that represent the original tables. Only return the json objects, no additional commentary or content.",
            },
            {
                "role": "user",
                "content": '"""\n' + query + '"""\n',
            },
        ],
    )
    complete_response = ""
    # Print stream
    print("Response:")
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            complete_response += chunk.choices[0].delta.content
    return complete_response


def save_inference_as_json(response: str, output_path: str, file_name: str):
    """Verifies if response is valid JSON and saves it to json file"""

    # Remove ```json and ``` from response
    response = response.replace("```json", "").replace("```", "")
    try:
        json.loads(response)
    except ValueError:
        print("Response is not valid JSON")
    else:
        with open(f"{output_path}/{file_name}.json", "w") as file:
            file.write(response)


def main():
    llm_model = "gpt-4-1106-preview"
    # llm_model = "local-model"
    # Get a list of all pdfs in the folder
    pdfs_folder_name = "pdfs_to_test"
    pdfs_to_test = []

    # Get the current working directory
    current_directory = os.getcwd()

    # Create the full path to the folder
    folder_path = os.path.join(current_directory, pdfs_folder_name)

    # Get a list of all files and folders in the specified folder
    contents = os.listdir(folder_path)

    for item in contents:
        if item.endswith(".pdf"):
            pdfs_to_test.append(f"{folder_path}/{item}")

    start_time = time.time()
    # process each pdf
    for pdf in pdfs_to_test:
        # Split file name from path by removing everything behind the last "/"
        file_name = pdf.split("/")[-1].replace(".pdf", "")
        output_path = pdf.replace(".pdf", "")

        # Convert pdf to html and writes to disk
        convert_pdf_to_html(pdf, output_path)

        # get whole html and array of htmls preprocessed
        whole_html, array_of_htmls = preprocess_html(output_path, file_name)
        complete_response = ""

        complete_response = run_inference(whole_html, llm_model)
        save_inference_as_json(complete_response, output_path, f"{file_name}_whole")

        # # Run inference for each html page
        # for html_page in array_of_htmls:
        #     json_filename = file_name + "_page_" + str(array_of_htmls.index(html_page))

        #     # finds the tables in the html page and converts them to json
        #     individual_response = run_inference(html_page, llm_model)

        #     # save inference result as json
        #     save_inference_as_json(individual_response, output_path, json_filename)
        #     complete_response += individual_response

        end_time = time.time()
        running_time = end_time - start_time
        print("Running time:", running_time, "seconds")


if __name__ == "__main__":
    main()
