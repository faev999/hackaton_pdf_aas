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
from typing import Tuple, List, Optional, NoReturn


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


def execute_model_inference(
    query: str, model_identifier: str, api_endpoint: Optional[str] = None
) -> str:
    """
    Executes an inference query using a specified language model, optionally via a local server.

    Parameters:
    - query (str): The query to send to the model.
    - model_identifier (str): The identifier of the model to use for the inference.
    - api_endpoint (Optional[str]): The base URL for the API, if using a local model server. Defaults to None.

    Returns:
    - str: The complete response from the model.

    Raises:
    - ValueError: If `api_endpoint` is required but not provided.
    """
    if model_identifier == "local-model" and api_endpoint is None:
        raise ValueError("API endpoint must be provided when using a local model.")

    client = OpenAI(base_url=f"http://{api_endpoint}/v1" if api_endpoint else None)

    print(f"Model client for {model_identifier} created. Preparing to send query...")

    token_count = calculate_token_count(query, model_identifier)
    print(f"Number of tokens to send: {token_count}")

    print("Sending request...")
    response_stream = client.chat.completions.create(
        model=model_identifier,
        response_format={"type": "json_object"},
        stream=True,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Good morning, you are a helpful assistant and an expert web developer. Some tables were converted into the HTML code that's inside triple backticks. Please turn that code into several JSON objects that represent the original tables. Only return the json objects, no additional commentary or content.",
            },
            {"role": "user", "content": "```\n" + query + "```"},
        ],
    )

    complete_response = ""
    for chunk in response_stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            complete_response += chunk.choices[0].delta.content
    return complete_response


def save_response_as_json(
    response: str, output_directory: str, output_file_name: str
) -> NoReturn:
    """
    Validates the given response string as JSON and saves it to a specified file in JSON format.

    Parameters:
    - response (str): The response string to validate and save.
    - output_directory (str): The directory path where the JSON file will be saved.
    - output_file_name (str): The name of the file to which the JSON data will be written.

    Raises:
    - ValueError: If the response string is not valid JSON.
    - IOError: If there is an issue writing the file.
    """
    cleaned_response = response.strip(
        "```"
    )  # More robust stripping of potential formatting characters

    try:
        parsed_response = json.loads(
            cleaned_response
        )  # Attempt to parse the string as JSON
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode response as JSON: {e}")

    output_file_path = f"{output_directory}/{output_file_name}.json"
    try:
        with open(output_file_path, "w") as file:
            json.dump(
                parsed_response, file, indent=4
            )  # Write the parsed JSON back out, nicely formatted
    except IOError as e:
        raise IOError(f"Error writing JSON to file {output_file_path}: {e}")


def process_single_pdf(
    pdf_folder_path: str, pdf_file: str, model_identifier: str
) -> None:
    """
    Converts a single PDF to HTML, cleans the HTML, runs model inference, and saves the result.

    Parameters:
    - pdf_folder_path (str): The full path to the folder containing the PDF.
    - pdf_file (str): The name of the PDF file to process.
    - model_identifier (str): The model identifier for running inference.
    """
    file_name = pdf_file.replace(".pdf", "")
    output_path = os.path.join(pdf_folder_path, file_name)

    # Convert PDF to HTML and clean the HTML content
    convert_pdf_to_html(os.path.join(pdf_folder_path, pdf_file), output_path)
    whole_html, _ = clean_html_content(os.path.join(output_path, f"{file_name}.html"))

    # Execute model inference and save the result as JSON
    complete_response = execute_model_inference(whole_html, model_identifier)
    save_response_as_json(complete_response, output_path, f"{file_name}_whole")


def print_running_time(start_time: float) -> None:
    """
    Prints the total running time since a given start time.

    Parameters:
    - start_time (float): The start time in seconds.
    """
    end_time = time.time()
    running_time = end_time - start_time
    print(f"Running time: {running_time:.2f} seconds")


def main():
    """
    Processes all PDF files in a given folder: converting them to HTML, cleaning the HTML content,
    running model inference on the cleaned content, and saving the inference results as JSON.

    Parameters:
    - pdf_folder (str): The folder containing PDF files to process.
    - model_identifier (str): The identifier of the model used for inference.
    """
    model_identifier = "gpt-4-0125-preview"
    pdf_folder = "test"
    start_time = time.time()

    # Resolve the full path to the PDF folder and gather all PDF files
    pdf_folder_path = os.path.join(os.getcwd(), pdf_folder)
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        process_single_pdf(pdf_folder_path, pdf_file, model_identifier)

    print_running_time(start_time)


if __name__ == "__main__":
    main()
