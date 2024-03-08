import os
import time
from pdf_tables_to_json_pipeline import PdfToJsonPipeline


class ProcessPdfs:
    def __init__(self, model_identifier: str, pdf_folder: str):
        self.model_identifier = model_identifier
        self.pdf_folder = pdf_folder
        self.pipeline = PdfToJsonPipeline(model_identifier)

    def process_single_pdf(self, pdf_folder_path: str, pdf_file: str) -> None:
        """
        Converts a single PDF to HTML, cleans the HTML, runs model inference, and saves the result.

        Parameters:
        - pdf_folder_path (str): The full path to the folder containing the PDF.
        - pdf_file (str): The name of the PDF file to process.
        """
        file_name = pdf_file.replace(".pdf", "")
        output_path = os.path.join(pdf_folder_path, file_name)

        # Convert PDF to HTML and clean the HTML content
        self.pipeline.convert_pdf_to_html(
            os.path.join(pdf_folder_path, pdf_file), output_path
        )
        cleaned_html, divs = self.pipeline.clean_html_content(
            os.path.join(output_path, f"{file_name}.html")
        )
        
        text_data = self.pipeline.html_to_text(cleaned_html)
        
        start_time = time.time()
        # Execute model inference and save the result as JSON
        html_tables_as_json = self.pipeline.html_tables_to_json_llm(cleaned_html, "gpt-4-turbo-preview", False, True)
        # print(html_tables_as_json)
        self.pipeline.save_response_as_json(
        html_tables_as_json, output_path, f"{file_name}_whole"
        )
        self.print_running_time(start_time)
        
        start_time = time.time()
        text_tables_as_json = self.pipeline.text_tables_to_json_llm(text_data,"gpt-4-turbo-preview", False, True)
        # print(text_tables_as_json)
        self.pipeline.save_response_as_json(
            text_tables_as_json, output_path, f"{file_name}_whole_text"
        )
        self.print_running_time(start_time)
        
        start_time = time.time()
        html_tables_as_yaml = self.pipeline.html_tables_to_yaml_llm(cleaned_html,"gpt-4", False, False)
        # print(html_tables_as_yaml)
        self.pipeline.save_response_as_yaml(
            html_tables_as_yaml, output_path, f"{file_name}_whole"
        )
        self.print_running_time(start_time)
        
        start_time = time.time()
        text_tables_as_yaml = self.pipeline.text_tables_to_yaml_llm(text_data, "gpt-4", False, False)
        # print(text_tables_as_yaml)
        self.pipeline.save_response_as_yaml(
            text_tables_as_yaml, output_path, f"{file_name}_whole_text"
        )
        self.print_running_time(start_time)
        
        # self.pipeline.save_response_as_txt(
        #     html_tables_as_yaml, output_path, f"{file_name}_whole"
        # )
      
        

    def print_running_time(self, start_time: float) -> None:
        """
        Prints the total running time since a given start time.

        Parameters:
        - start_time (float): The start time in seconds.
        """
        end_time = time.time()
        running_time = end_time - start_time
        print(f"Running time: {running_time:.2f} seconds")

    def run(self):
        """
        Processes all PDF files in a given folder: converting them to HTML, cleaning the HTML content,
        running model inference on the cleaned content, and saving the inference results as JSON.
        """
        start_time = time.time()

        # Resolve the full path to the PDF folder and gather all PDF files
        pdf_folder_path = os.path.join(os.getcwd(), self.pdf_folder)
        pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

        for pdf_file in pdf_files:
            self.process_single_pdf(pdf_folder_path, pdf_file)

        self.print_running_time(start_time)


def main():
    model_identifier = "gpt-4-turbo-preview"
    pdf_folder = "test"
    process_pdfs = ProcessPdfs(model_identifier, pdf_folder)
    process_pdfs.run()


if __name__ == "__main__":
    main()
