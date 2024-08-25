import os
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import re


def sort_key(filename):
    # Extract numbers from the filename for sorting
    numbers = re.findall(r'\d+', filename)
    return list(map(int, numbers)) if numbers else [0]


def merge_pdfs_with_page_numbers(input_folder, output_pdf):
    # Get a list of all PDF files in the directory and sort by the custom key
    pdf_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.pdf')], key=sort_key)
    print(pdf_files)
    pdf_writer = PdfWriter()

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        pdf_reader = PdfReader(pdf_path)

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            # Create a new PDF with the page number
            packet = BytesIO()
            can = canvas.Canvas(packet, pagesize=letter)
            can.drawString(500, 10, str(len(pdf_writer.pages) + 1))  # Adjust the position as needed
            can.save()

            packet.seek(0)
            new_pdf = PdfReader(packet)
            page.merge_page(new_pdf.pages[0])

            pdf_writer.add_page(page)

    # Write the merged PDF to the output file
    with open(output_pdf, 'wb') as output_pdf_file:
        pdf_writer.write(output_pdf_file)

    print(f"Merged PDF created: {output_pdf}")


# Example usage
input_folder = r'C:\Users\ASUS\Desktop\קידוד גילוי'  # Replace with your folder path
output_pdf = r'C:\Users\ASUS\Desktop\HW_all_Q.pdf'  # Replace with the desired output path

merge_pdfs_with_page_numbers(input_folder, output_pdf)
