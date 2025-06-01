import os
import sys

from PyPDF2 import PdfReader, PdfWriter


def trim_pdf(input_path, start_page, end_page, output_path):
    try:
        if not os.path.isfile(input_path):
            print(f"Error: Input file '{input_path}' not found")
            return False

        reader = PdfReader(input_path)
        writer = PdfWriter()

        if start_page < 0 or end_page >= len(reader.pages) or start_page > end_page:
            print(
                f"Error: Invalid page range. PDF has {len(reader.pages)} pages (0-{len(reader.pages)-1})"
            )
            return False

        for page_num in range(start_page, end_page + 1):
            writer.add_page(reader.pages[page_num])

        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        print(
            f"Successfully created '{output_path}' with pages {start_page} to {end_page}"
        )
        return True

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return False
