import argparse
import os
import json

# produced by generate_replica_metadata.py
SCRIPTS_TO_RUN = (
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_19/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_19"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc2_staging_13/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc2_staging_13"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_04/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_04"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc2_staging_14/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc2_staging_14"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc0_staging_03/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc0_staging_03"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_02/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_02"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc3_staging_04/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc3_staging_04"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc0_staging_15/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc0_staging_15"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_09/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_09"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc3_staging_14/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc3_staging_14"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_01/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc1_staging_01"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc0_staging_02/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc0_staging_02"),
    ("python3", "-m", "datasets.habitat_sim.generate_from_metadata", "--metadata_filename", "/scratch/izar/skorokho/multiview_habitat_metadata/replica_cad_baked_lighting/remake_v0_v3_sc3_staging_01/metadata.json", "--output_dir", "/scratch/izar/skorokho/croco-dataset/replica_cad_baked_lighting/remake_v0_v3_sc3_staging_01"),
)

def main():
    for script in SCRIPTS_TO_RUN:
        script = list(script)
        metadata_filename = script[4]
        with open(metadata_filename, "r") as f:
            metadata = json.load(f)
        metadata["scene"] = metadata["scene"].replace("/remake_v0_v3_", "/stages/Baked_")
        metadata["navmesh"] = metadata["scene"].replace("/stages", "/navmeshes").replace(".glb", ".navmesh")

        new_metadata_filename = metadata_filename.replace("metadata.json", "metadata_updated.json")
        with open(new_metadata_filename, "w") as f:
            json.dump(metadata, f)

        script[4] = new_metadata_filename
        # replica_cad_baked_lighting/remake_v0_v3_sc1_staging_19.glb ->
        # replica_cad_baked_lighting/stages/Baked_sc0_staging_00.glb        

        print(f"Running script: {' '.join(script)}")
        os.system(" ".join(script))

if __name__ == "__main__":
    main()
