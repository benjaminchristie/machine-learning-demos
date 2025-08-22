# TransformerDecoder Example 


the following textfiles should be put in the `./data/` directory:

- [shake.txt](https://gist.githubusercontent.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254/raw/76fe1b5e9efcf0d2afdfd78b0bfaa737ad0a67d3/shakespeare.txt)

to download the dataset and train the decoder, run the following commands:

```bash
<cd into this directory>
mkdir -p data
wget -O data/shake.txt https://gist.githubusercontent.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254/raw/76fe1b5e9efcf0d2afdfd78b0bfaa737ad0a67d3/shakespeare.txt
python main.py
```

run `python main.py --demo` to spit out the example (if the decoder has already been trained)
