import click
import json
import numpy as np
import thre3d_atom.data.constants as thre3d_dat_consts

from easydict import EasyDict
from imageio import imread
from pathlib import Path
from thre3d_atom.utils.logging import log

# ------------------------------------------------------------------------------------
# Hardcoded constants for the script                                                 |
# ------------------------------------------------------------------------------------
SPLITS = ["train", "val", "test"]
NEAR, FAR = 2.0, 6.0
# ------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()

# Required arguments:
@click.option("-d", "--data_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path to the original nerf synthetic dataset scene")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for outputting the converted scene")
# -------------------------------------------------------------------------------------
# fmt: on
def main(**kwargs) -> None:
    # extract the required paths:
    config = EasyDict(kwargs)
    data_path = Path(config.data_path)
    output_path = Path(config.output_path)

    # load all the source json data in the nerf transforms_<split>.json files
    log.info(f"loading the data from source path: {data_path}")
    meta_jsons = {}
    for split in SPLITS:
        with open(str(data_path / f"transforms_{split}.json"), "r") as json_fp:
            meta_jsons[split] = json.load(json_fp)

    # convert the json data into 3d-atom data format :)
    log.info(f"converting the data ...")
    thre3d_atom_meta_jsons = {}
    for split in meta_jsons.keys():
        thre3d_atom_meta_jsons[split] = {}

        # extract the height, width and focal:
        starting_file_name = (
            meta_jsons[split]["frames"][0]["file_path"].split("/")[-1] + ".png"
        )
        img = imread(data_path / split / starting_file_name)
        height, width = img.shape[:2]
        focal = 0.5 * width / np.tan(0.5 * float(meta_jsons[split]["camera_angle_x"]))

        for idx, frame in enumerate(meta_jsons[split]["frames"]):
            filename = frame["file_path"].split("/")[-1] + ".png"

            # FILL new json:
            camera_param = {
                thre3d_dat_consts.INTRINSIC: {
                    thre3d_dat_consts.BOUNDS: [NEAR, FAR],
                    thre3d_dat_consts.HEIGHT: height,
                    thre3d_dat_consts.WIDTH: width,
                    thre3d_dat_consts.FOCAL: focal,
                },
                thre3d_dat_consts.EXTRINSIC: {
                    thre3d_dat_consts.ROTATION: np.array(frame["transform_matrix"])[
                        :3, :3
                    ].tolist(),
                    thre3d_dat_consts.TRANSLATION: np.array(frame["transform_matrix"])[
                        :3, -1:
                    ].tolist(),
                },
            }

            thre3d_atom_meta_jsons[split][filename] = camera_param

    log.info(f"writing the converted data ...")
    for split in thre3d_atom_meta_jsons.keys():
        with open(
            str(output_path / f"{split}_camera_params.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(thre3d_atom_meta_jsons[split], f, ensure_ascii=False, indent=4)

    log.info(f"converted data is available at: {output_path}")


if __name__ == "__main__":
    main()
