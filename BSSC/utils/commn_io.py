import json
import os

import pandas as pd
import yaml  # type: ignore


def read_config(config_file):
    if isinstance(config_file, str):  # it is a path
        assert os.path.isfile(config_file), f"file does not exist {config_file}"
        if config_file.lower().endswith("yaml") or config_file.lower().endswith("yml"):
            with open(config_file, "r") as stream:
                conf = yaml.safe_load(stream)
        elif config_file.lower().endswith("json"):
            with open(config_file, "r") as stream:
                conf = json.load(stream)
        else:
            print(f"not support type of config file {config_file}")
            with open(config_file, "r") as stream:
                conf = json.load(stream)
        return conf
    elif isinstance(config_file, dict):  # if it is already a dict
        return config_file


def read_relative_csv(
    csv_path: str,
    csv_filename="labels.csv",
    image_path=None,
) -> pd.DataFrame:
    """read relative csv file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    if isinstance(csv_path, str) and not csv_path.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_path, csv_filename))
        data_folder = csv_path
    elif isinstance(csv_path, str):
        df = pd.read_csv(csv_path)
        data_folder = os.path.dirname(csv_path)
    if image_path is None:
        images_path = data_folder
    else:
        images_path = os.path.join(data_folder, image_path)
    df["path"] = df["path"].apply(lambda x: images_path + "/" + x)
    return df
