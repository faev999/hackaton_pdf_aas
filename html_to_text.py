import html2text


h = html2text.HTML2Text()

# Ignore converting links from HTML
h.ignore_links = True

with open('test/134132_eng/134132_eng_processed.html', 'r') as file:
    html_data = file.read()

text_data = h.handle(html_data)

with open('text_from_html.txt', 'w') as file:
    file.write(text_data)



print(text_data)


