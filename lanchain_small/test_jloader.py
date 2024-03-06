from langchain_community.document_loaders import JSONLoader
from pathlib import Path
import json
from pprint import pprint


folder_path = './data/'
file_path = './data/langchain_intro.json'

# data = json.loads(Path(file_path).read_text())
# pprint(Path(file_path).read_text())

loader = JSONLoader(
    file_path=file_path,
    jq_schema='.[]',
    content_key='html',
)

data = loader.load()

pprint(data)

