Build instructions for PYPA resonet release

To install resonet, clone this repository, then 

```
cd resonet
python -m build
python -m pip install dist/resonet-0.1.tar.gz
# optional if building resonet for training data simulation (i.e., if using a CCTBX build)
python update_shebang.py
```

Tutorial can be found [here](https://github.com/dermen/resonet/blob/master/README.md)
