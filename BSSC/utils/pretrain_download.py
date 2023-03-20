import hashlib
import os

import gdown


def extract_md5(x):
    return hashlib.md5(open(x, "rb").read()).hexdigest()


def download_pretrain_model(name):
    if name == "wavlm_large":
        url = "https://drive.google.com/uc?id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU"
        output = "~/.cache/torch/hub/checkpoints/WavLM-Large.pt"
        output = os.path.expanduser(output)
        if (
            os.path.exists(output)
            and extract_md5(output) == "c3813bfe66fa1624e66b589ffd1e3429"
        ):
            return output
        else:
            gdown.download(url, output, quiet=False)
            print("MD5: ", extract_md5(output))

        return output
    else:
        raise NotImplementedError
