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
import html2text
import yaml

class PdfToJsonPipeline:
    def __init__(self, model_identifier: str, api_endpoint: Optional[str] = None):
        self.model_identifier = model_identifier
        self.api_endpoint = api_endpoint

    def calculate_token_count(self, text: str) -> int:
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
            encoding = tiktoken.encoding_for_model(self.model_identifier)
        except KeyError:
            raise ValueError(f"Unknown model identifier: {self.model_identifier}")
        return len(encoding.encode(text))

    def convert_pdf_to_html(self, pdf_file_path: str, output_directory: str) -> None:
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

    def clean_html_content(self, html_file_path: str) -> Tuple[str, List[str]]:
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
            cleaned_div = self._remove_images_and_empty_divs(div)
            div_as_str = str(cleaned_div)
            divs_as_strings.append(div_as_str)
            cleaned_html += div_as_str

        processed_html_path = html_file_path.replace(".html", "_processed.html")
        try:
            with open(processed_html_path, "w") as file:
                file.write(cleaned_html + "\n")
        except IOError as e:
            raise IOError(f"Failed to write cleaned HTML to {processed_html_path}: {e}")

        return cleaned_html, divs_as_strings

    def _remove_images_and_empty_divs(self, div_element) -> None:
        """
        Helper function to remove image tags and empty divs from a div element.

        Parameters:
        - div_element (bs4.element.Tag): The BeautifulSoup tag object representing a div element.
        """
        for img in div_element.find_all("img"):
            img.decompose()
        for div in div_element.find_all("div"):
           
            if "class" in div.attrs:
                del div.attrs["class"]
            if not div.find_all(string=lambda text: isinstance(text, NavigableString)):
                div.decompose()
            if not div.text.strip():
                div.decompose()
        for span in div_element.find_all("span"):
            if not span.text.strip():
                span.decompose()
        return div_element

    def html_tables_to_json_llm(self, query: str, model:str, streaming:bool, json_mode:bool) -> str:
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
        
        prompt = "Good morning, you are a helpful assistant and an expert web developer. Some  in a pdf file were converted into the HTML code that's inside triple backticks. Please turn that code into several JSON structures that represent the original tables. Try to recognize and exclude headers and footers from the structures. Only return the JSON structures, no additional commentary or content."
        response = self.run_inference(query, model, streaming, prompt, json_mode)
        return response
    
    def html_to_text(self, html_data) -> str:
        """
        Converts a HTML file to text format, saving the output in a specified directory.

        Parameters:
        - html_file_path (str): The file path of the HTML to convert.

        Returns:
        - str: The complete text content of the HTML file.
        """
        h = html2text.HTML2Text()
        # Ignore converting links from HTML
        h.ignore_links = True
        

        text_data = h.handle(html_data)
        return text_data
    
    def text_tables_to_json_llm(self, query: str, model:str, streaming:bool, json_mode:bool) -> str:
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
       
        prompt = "Good morning, you are a helpful assistant and an expert web developer. Some tables were converted into HTML code and then into the text that's inside triple backticks. Please turn that text into several JSON structures that represent the original tables.Try to recognize and exclude headers and footers from the structures. Only return the JSON structures, no additional commentary or content."
        response = self.run_inference(query, model, streaming, prompt, json_mode)
        return response

    def html_tables_to_yaml_llm(self, query: str, model:str, streaming:bool, json_mode:bool) -> str:
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
       
        prompt = "Good morning, you are a helpful assistant and an expert web developer. Some tables were converted into the HTML code that's inside triple backticks. Please turn that code into several valid YAML structures that represent the original tables. Make yure the YAML structures are valid with no invalid chracters inside the values or the keys. Try to recognize and exclude headers and footers from the structures. Only return the YAML structures, no additional commentary or content."
        response = self.run_inference(query, model, streaming, prompt, json_mode)
        return response

    
    def text_tables_to_yaml_llm(self, query: str, model:str, streaming:bool, json_mode:bool) -> str:
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
        
        prompt = "Good morning, you are a helpful assistant and an expert web developer. Some tables were converted into HTML code and then into the text that's inside triple backticks. Please turn that text into several valid YAML structures that represent the original tables. Make yure the YAML structures are valid with no invalid chracters inside the values or the keys. Try to recognize and exclude headers and footers from the structures. Only return the YAML structures, no additional commentary or content."
        response = self.run_inference(query, model, streaming, prompt, json_mode)
        return response
    
    def run_inference(self, query: str, model:str, streaming:bool, prompt:str, json_mode:bool) -> str:
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
        if model == "local-model" and self.api_endpoint is None:
            raise ValueError("API endpoint must be provided when using a local model.")

        client = OpenAI(
            base_url=f"http://{self.api_endpoint}/v1" if self.api_endpoint else None
        )

        print(
            f"Model client for {model} created. Preparing to send query..."
        )

        token_count = self.calculate_token_count(query)
        print(f"Number of tokens to send: {token_count}")

        print("Sending request...")
        soup = BeautifulSoup(query, 'html.parser')
        prettified_html = soup.prettify()
        if json_mode :
           
            format_type = {"type": "json_object"}
            if  model == "gpt-4":
                print("gpt-4 doesnt support json.Using gpt-4-turbo-preview instead")
                model = "gpt-4-turbo-preview"
        
        if not json_mode:
            # TODO is text the correct type for "normal" inference?
            format_type = {"type": "text"}
        response_stream = client.chat.completions.create(
            model=model,
            stream=streaming,
            temperature=0.0,
            response_format=format_type,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": "```\n" + prettified_html + "```"},
            ],
        )
        if streaming:
            complete_response = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
                    complete_response += chunk.choices[0].delta.content
            return complete_response
        else:
            # print(response_stream.choices[0].message.content)
            return response_stream.choices[0].message.content

    
    def save_response_as_json(
        self, response: str, output_directory: str, output_file_name: str
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
        
    def save_response_as_yaml(
        self, response: str, output_directory: str, output_file_name: str
    ) -> NoReturn:
        """
        Validates the given response string as YAML and saves it to a specified file in YAML format.

        Parameters:
        - response (str): The response string to validate and save.
        - output_directory (str): The directory path where the YAML file will be saved.
        - output_file_name (str): The name of the file to which the YAML structures will be written.

        Raises:
        - ValueError: If the response string is not valid YAML.
        - IOError: If there is an issue writing the file.
        """
        # print(response)
        cleaned_response = response.replace("```yaml", "")
        cleaned_response = cleaned_response.replace("```", "")
        print(cleaned_response)
        
        # TODO is this the correct way to clean brakets? 
        cleaned_response = cleaned_response.replace("[", "\"")
        cleaned_response = cleaned_response.replace("]", "\"")
        # cleaned_response = cleaned_response.replace("\"", "")
    
        try:
            parsed_response = yaml.load(
                cleaned_response, Loader=yaml.UnsafeLoader 
            )  # Attempt to parse the string as YAML
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to decode response as YAML: {e}")

        output_file_path = f"{output_directory}/{output_file_name}.yaml"
        try:
            with open(output_file_path, "w") as file:
                yaml.dump(parsed_response, file)
        except IOError as e:
            raise IOError(f"Error writing YAML to file {output_file_path}: {e}")
        
    def yaml_to_json(self, yaml_data: str,output_directory: str, output_file_name: str) -> str:
        """
        Converts a YAML string to a JSON string.

        Parameters:
        - yaml_data (str): The YAML string to convert.

        Returns:
        - str: The JSON string equivalent to the given YAML string.

        Raises:
        - ValueError: If the given YAML string is not valid.
        """
        cleaned_response = yaml_data.replace("```yaml", "")
        cleaned_response = cleaned_response.replace("```", "")
        try:
            parsed_yaml = yaml.load(cleaned_response, Loader=yaml.UnsafeLoader)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to decode YAML data: {e}")
        json_data = json.dumps(parsed_yaml, indent=4)
        output_file_path = f"{output_directory}/{output_file_name}.json"
        try:
            with open(output_file_path, "w") as file:
                file.write(json_data)
        except IOError as e:
            raise IOError(f"Error writing JSON to file {output_file_path}: {e}")
        return json_data
    
    def save_response_as_txt(
        self, response: str, output_directory: str, output_file_name: str
    ) -> NoReturn:
        """
        Validates the given response string as YAML and saves it to a specified file in YAML format.

        Parameters:
        - response (str): The response string to validate and save.
        - output_directory (str): The directory path where the YAML file will be saved.
        - output_file_name (str): The name of the file to which the YAML structures will be written.

        Raises:
        - ValueError: If the response string is not valid YAML.
        - IOError: If there is an issue writing the file.
        """
       
       

        output_file_path = f"{output_directory}/{output_file_name}.yaml"
        try:
            with open(output_file_path, "w") as file:
                file.write(
                    response
                )  # Write the parsed YAML back out as text
        except IOError as e:
            raise IOError(f"Error writing YAML to file {output_file_path}: {e}")
        
      
