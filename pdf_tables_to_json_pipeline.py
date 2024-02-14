# MIT License
#
# Copyright (c) [2024] [Fabio Espinosa]
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
from typing import Tuple, List


def calculate_token_count(text: str, model_identifier: str) -> int:
    """
    Calculate the number of tokens in a given text based on a specified model's encoding.

    Parameters:
    - text (str): The input text to encode.
    - model_identifier (str): The identifier of the model to use for encoding.

    Returns:
    - int: The number of tokens in the encoded text.

    Raises:
    - ValueError: If the model_identifier does not correspond to any known model encoding.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_identifier)
    except KeyError:
        raise ValueError(f"Unknown model identifier: {model_identifier}")
    return len(encoding.encode(text))


def convert_pdf_to_html(pdf_file_path: str, output_directory: str) -> None:
    """
    Converts a PDF file to HTML format, saving the output in a specified directory.

    Parameters:
    - pdf_file_path (str): The file path of the PDF to convert.
    - output_directory (str): The directory where the HTML file will be saved.

    Raises:
    - FileNotFoundError: If the specified PDF file does not exist.
    - subprocess.CalledProcessError: If the pdf2htmlEX command fails.
    """
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(
            f"The specified PDF file does not exist: {pdf_file_path}"
        )

    command = f"pdf2htmlEX '{pdf_file_path}' --dest-dir '{output_directory}' --font-size-multiplier 1 --zoom 25"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to convert PDF to HTML: {e}")


def clean_html_content(html_file_path: str) -> Tuple[str, List[str]]:
    """
    Removes images, styles, and non-textual elements from an HTML file,
    and saves the cleaned content to a new file.

    Parameters:
    - html_file_path (str): The path to the original HTML file.

    Returns:
    - Tuple[str, List[str]]: A tuple containing the complete cleaned HTML content as a string
      and a list of cleaned div elements as strings.

    Raises:
    - IOError: If the specified HTML file cannot be read or written.
    """
    try:
        with open(html_file_path, "r") as file:
            html_content = file.read()
    except IOError as e:
        raise IOError(f"Failed to read HTML file at {html_file_path}: {e}")

    soup = BeautifulSoup(html_content, "html.parser")
    page_divs = soup.find_all("div", attrs={"data-page-no": True})

    cleaned_html = ""
    divs_as_strings = []
    for div in page_divs:
        _remove_images_and_empty_divs(div)
        div_as_str = str(div)
        divs_as_strings.append(div_as_str)
        cleaned_html += div_as_str

    processed_html_path = html_file_path.replace(".html", "_processed.html")
    try:
        with open(processed_html_path, "w") as file:
            file.write(cleaned_html + "\n")
    except IOError as e:
        raise IOError(f"Failed to write cleaned HTML to {processed_html_path}: {e}")

    return cleaned_html, divs_as_strings


def _remove_images_and_empty_divs(div_element) -> None:
    """
    Helper function to remove image tags and empty divs from a div element.

    Parameters:
    - div_element (bs4.element.Tag): The BeautifulSoup tag object representing a div element.
    """
    for img in div_element.find_all("img"):
        img.decompose()
    for div in div_element.find_all("div"):
        if not div.find_all(string=lambda text: isinstance(text, NavigableString)):
            div.decompose()
        elif "class" in div.attrs:
            del div.attrs["class"]


def run_inference(query: str, llm_model: str, local_ip=None):
    """Sends query to openai and returns the response"""
    if llm_model == "local-model":
        if local_ip is None:
            raise ValueError("local_ip must be provided when using local-model")
        client = OpenAI(base_url=f"http://{local_ip}:5000/v1", api_key="not-needed")
    else:
        client = OpenAI()
    print("LLM client created")
    print(
        "Number of tokens to send: ",
        calculate_token_count(query, "gpt-4-1106-preview"),
    )
    print("sending request")
    response = client.chat.completions.create(
        model=llm_model,
        response_format={"type": "json_object"},
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
        raise
    else:
        with open(f"{output_path}/{file_name}.json", "w") as file:
            file.write(response)


def main():
    llm_model = "gpt-4-0125-preview"
    # llm_model = "local-model"

    # Get a list of all pdfs in the folder
    pdfs_folder_name = "test"
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
        whole_html, array_of_htmls = clean_html_content(
            output_path + "/" + file_name + ".html"
        )
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
