from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "tests/trimmed_eis_FirstNet.pdf"

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

with open("tests/trimmed_eis_FirstNet.txt", "w") as f:
    for page_num, doc in enumerate(docs):
        f.write(f"Page {page_num + 1}:\n")
        f.write(doc.page_content)
        f.write("\n\n")



