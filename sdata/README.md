### How to use

1. Create a python env and install pipx into it.
2. Run `pipx install . --force`

You may now run the following commands

```
eod AAPL
eod AAPL -f 20250301 -t 20250307
```

or

```
intraday AAPL
intraday AAPL -f 20250301 -t 20250307
```

your outputs will be in `/output/...`
