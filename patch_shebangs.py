import os
import shutil

"""
Usage: if using resonet with a CCTBX Python installation
  then use this script to update the shebang in 
  simulate and train scripts
"""

shebang="#!/usr/bin/env libtbx.python\n"

scripts = ["resonet-simulate", "resonet-train", "resonet-imgeater"]
for s in scripts:
    s_path = shutil.which(s)
    lines = open(s_path, "r").readlines()
    lines[0] = shebang
    with open(s_path, "w") as o:
        o.writelines(lines)
    print("Updateed", s, s_path)

