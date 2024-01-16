# Script that transforms pdfs to hmtl code and then cleans this code until only tables are left and then using gpt4 transforms this tables into json objects

import subprocess
import time
from bs4 import BeautifulSoup, NavigableString
from openai import OpenAI
import json
import pandas as pd
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


def process_html(output_path, file_name):
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
    client = OpenAI()
    print("OpenAI client created")
    print("Number of tokens to send: ", num_tokens_from_string(query, llm_model))
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


def save_json(response: str, output_path: str, file_name: str):
    """Verifies if response is valid JSON and saves it to json file"""
    print(output_path, " ", file_name)
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
    # Create array of PDFs to be processed by reading contents of the folder "testing_pdfs" and getting their paths

    pdfs_folder_name = "pdfs_to_test"
    pdfs_to_test = []

    # Get the current working directory
    current_directory = os.getcwd()

    # Create the full path to the folder
    folder_path = os.path.join(current_directory, pdfs_folder_name)

    # Get a list of all files and folders in the specified folder
    contents = os.listdir(folder_path)

    # Print the contents
    for item in contents:
        if item.endswith(".pdf"):
            pdfs_to_test.append(item)
            print(item)

    print(pdfs_to_test)

    openai_model = "gpt-4-1106-preview"
    # process each pdf
    for pdf in pdfs_to_test:
        # Split file name from path by removing everything behind the last "/"
        file_name = pdf.split("/")[-1].replace(".pdf", "")
        output_path = pdf.replace(".pdf", "")
        start_time = time.time()

        convert_pdf_to_html(pdf, output_path)
        whole_html, array_of_html = process_html(output_path, file_name)

        complete_response = ""
        for html in array_of_html:
            json_filename = file_name + "_page_" + str(array_of_html.index(html))
            print(output_path)
            individual_response = run_inference(html, openai_model)

            save_json(individual_response, output_path, json_filename)
            # complete_response += individual_response

        end_time = time.time()
        running_time = end_time - start_time
        print("Running time:", running_time, "seconds")

    # input_pdf = "134132_eng.pdf"
    # output_name = input_pdf.replace(".pdf", "")

    # start_time = time.time()

    # convert_pdf_to_html(input_pdf, output_name)

    # whole_html, array_of_html = process_html(output_name)

    # openai_model = "gpt-4-1106-preview"

    # # complete_response = ""
    # # for html in array_of_html:
    # #     individual_response = run_inference(html, openai_model)
    # #     complete_response += individual_response

    # complete_response = run_inference(whole_html, openai_model)
    # # save_json(complete_response, output_name)

    # end_time = time.time()
    # running_time = end_time - start_time
    # print("Running time:", running_time, "seconds")


if __name__ == "__main__":
    main()
