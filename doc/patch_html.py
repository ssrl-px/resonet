import sys
code="""code {
  white-space : pre-wrap !important;
}
"""
s = open("index.html", 'r').read()
s = s.replace('<style type="text/css">', '<style type="text/css">\n%s' % code)
print(s)
with open("index.html", "w") as o:
    o.write(s)
