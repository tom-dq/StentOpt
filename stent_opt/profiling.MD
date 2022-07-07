
Install profiling deps:

`pip install snakeviz`

Generate the profile data:

```
$env:PYTHONPATH = "C:\Users\Tom Wilson\Documents\source\StentOpt";
cd stent_opt
python -m cProfile -o ABC123.prof make_stent.py
```

Visualise it:
`snakeviz ABC123.prof`
