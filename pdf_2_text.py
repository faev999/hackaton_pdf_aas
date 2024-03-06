import pdftotext

# Load your PDF
with open("test/134132_eng.pdf", "rb") as f:
    pdf = pdftotext.PDF(f, physical=True)


# How many pages?
print(len(pdf))

# Iterate over all the pages
for page in pdf:
    print(page)

# Read some individual pages
print(pdf[0])
print(pdf[1])

# Read all the text into one string
print("\n\n".join(pdf))

# Save the PDF text to a file
with open("test/134132_eng.txt", "w") as text_file:
    text_file.write("\n\n".join(pdf))
