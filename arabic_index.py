


import os
import json
import PyPDF2
import textwrap
import numpy as np

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("M-CLIP/M-BERT-Distil-40")

model = AutoModel.from_pretrained("M-CLIP/M-BERT-Distil-40")



"""
This code iterates over a list of pdfs, extracts text and divide it into chunks. Then it saves the chunks into a json file.
"""
pdf_dir = os.path.join(os.getcwd(), 'Arabic pdfs')
json_file = os.getcwd()+'/index.json'

pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

output = []

for pdf_file in pdf_files:
    with open(os.path.join(pdf_dir, pdf_file), "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        print(f'Reading pdf: {f}')
        num_pages = len(pdf_reader.pages)
        print(f'This pdf has {num_pages} pages')
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            content = page.extract_text()
            chunks = textwrap.wrap(content, 1000)
            
            for chunk in chunks:
                # Tokenize the chunk
                inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
                input_ids = inputs["input_ids"]
                
                # Pass the tokenized input to the model for embedding
                embeddings = model(**inputs).last_hidden_state
                embed = embeddings.detach().numpy()

                
                output.append({
                    "filename": pdf_file,
                    "page_number": page_num+1,
                    "content": chunk,
                    "embedding" : np.array(embed).tolist(),
                })

with open(json_file, "w") as f:
    json.dump(output, f)


