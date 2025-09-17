import fitz  # PyMuPDF is imported as 'fitz'
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None
    return text

# Step 1: Get the text from our sample PDF
pdf_path = "sample.pdf"
full_text = extract_text_from_pdf(pdf_path)

if full_text:
    # Step 2: Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    # Step 3: Split the text into smaller chunks
    max_char_count = 1000
    chunks = [full_text[i:i + max_char_count] for i in range(0, len(full_text), max_char_count)]

    # Step 4: Summarize each chunk
    summaries = []
    for chunk in chunks:
        try:
            chunk_summary = summarizer(chunk, max_length=130, min_length=10, do_sample=False)
            summaries.append(chunk_summary[0]['summary_text'])
        except IndexError:
            print("A chunk could not be summarized and was skipped.")
            continue

    # Step 5: Combine the summaries into a final summary
    final_summary = " ".join(summaries)

    print("--- Summary of PDF Document ---")
    print(final_summary)

else:
    print("Could not process the PDF. Please check the file path and type.")