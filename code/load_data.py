import yaml
import pandas as pd
import os
import json
import logging
from pprint import pprint


def load_data(config):
    chunk_size = 100
    batch_no = 1
    data = json.load(open(config["data_path"], "r", encoding="UTF-8"))
    pprint("json load finishes")

    for chunk in pd.read_json(
        config["data_path"], chunksize=chunk_size, lines=True, orient="records"
    ):
        chunk.to_csv(
            config["data_out_path"] + "chunk" + str(batch_no) + ".csv", index=False
        )
        batch_no += 1


def main():
    config_path = "../config/load_yelp.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # end with
    pprint("=" * 20 + "Configs" + "=" * 20)
    pprint(config)
    load_data(config)


main()
