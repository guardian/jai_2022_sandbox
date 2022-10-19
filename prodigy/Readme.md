# Custom Entity Linking Prodigy Recipe 

This folder contains the custom recipe and configuration for running the Guardian-JAI 2022 entity linking candidate annotation.

## How to run?
Please ensure that you run this code in the same directory in order to allow Prodigy to pick up the local `prodigy.json` file.

```bash
 python -m prodigy entity_linker.manual <dataset> <source_file> <nlp_model> <kb_file> <additional_info_entities_file -F el_recipe.py>
```