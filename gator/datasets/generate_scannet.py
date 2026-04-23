import argparse
import os
import json
from pathlib import Path
import shutil
import subprocess
import trimesh

SCANNET_DOWNLOAD_SCRIPT = Path("/scratch/izar/skorokho/test/download_scannet.py")
SCANNET_CACHE_DIR = Path("/scratch/izar/skorokho/test/scannet_cached/")

def main():
    # get from running datasets/habitat_sim/generate_from_metadata_files.py
    with open("/home/skorokho/coding/gator/gator/datasets/generate_scannet.sh") as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        if not line.startswith("python3"):
            continue
        parts = line.split()
        if len(parts) < 5:
            print("Skipping line: ", line)
            continue

        metadata_filename = parts[3].split("=")[1]
        with open(metadata_filename, "r") as f:
            metadata = json.load(f)
        scene_name = Path(metadata["scene"]).stem

        if (SCANNET_CACHE_DIR).exists():
            shutil.rmtree(SCANNET_CACHE_DIR)
            SCANNET_CACHE_DIR.mkdir()

        cmd = [
            "python3", str(SCANNET_DOWNLOAD_SCRIPT),
            "--id", scene_name, 
            "--out_dir", str(SCANNET_CACHE_DIR),
            "--type", "_vh_clean_2.ply"
        ]
        subprocess.run(cmd)

        src = Path(SCANNET_CACHE_DIR, "scans", scene_name, f"{scene_name}_vh_clean_2.ply")
        dst = src.with_suffix(".glb")
        print("src:", src)
        print("dst:", dst)
        mesh = trimesh.load(src, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh, got {type(mesh)}")
        scene = trimesh.Scene(mesh)
        glb_bytes = trimesh.exchange.gltf.export_glb(scene)
        dst.write_bytes(glb_bytes)

        metadata["scene"] = f"scannet/{scene_name}/{scene_name}_vh_clean_2.glb"
        new_metadata_filename = metadata_filename.replace("metadata.json", "metadata_updated.json")
        with open(new_metadata_filename, "w") as f:
            json.dump(metadata, f)
        parts[3] = "--metadata_filename=" + new_metadata_filename
        print(f"Running script: {' '.join(parts)}")
        os.system(" ".join(parts))

        print(f"Done for {i}")

if __name__ == "__main__":
    main()
