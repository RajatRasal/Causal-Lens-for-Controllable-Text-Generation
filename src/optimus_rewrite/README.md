Download and unzip models from
```
https://github.com/ChunyuanLI/Optimus/blob/master/doc/optimus_finetune_language_models.md
https://github.com/ChunyuanLI/Optimus/blob/master/doc/optimius_for_snli.md
```

Running experiments on pretrained model:
```
poetry run -m src.optimus_rewrite.experiments \
    --output-dir=./checkpoint-31250 \
    --step=31250 \
    --latent-size=768
```
