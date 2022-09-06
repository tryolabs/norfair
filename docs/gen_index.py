# This script generates a docs/index.md on the fly from the README.md
# fixing some of the links
import re

import mkdocs_gen_files

# read README on the root of the repo
with open("README.md") as f:
    content = f.read()

# remove "docs" from gifs and images
content = re.sub(r"\]\(/?docs/", r"](", content)
# remove "docs" from src fields in html
content = re.sub(r"src=\"/?docs/", 'src="', content)

# write the index
with mkdocs_gen_files.open("index.md", "w") as fd:  #
    print(content, file=fd)
