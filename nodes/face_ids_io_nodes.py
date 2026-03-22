"""
Face IDs I/O Nodes for ComfyUI.
Provides nodes for saving and loading FACE_IDS to/from JSON or CSV files.
"""

import json
import csv
import numpy as np
import folder_paths
import os
import trimesh
from pathlib import Path


class ExtractFaceIDsFromMesh:
    """
    Extract FACE_IDS from a segmented mesh.

    P3SAMSegmentMesh stores segmentation labels in mesh.face_attributes['part_id'].
    This node converts that representation back into the FACE_IDS payload used by
    the I/O nodes and any downstream tooling that expects per-face segment IDs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("FACE_IDS",)
    RETURN_NAMES = ("face_ids",)
    FUNCTION = "extract"
    CATEGORY = "Hunyuan3D/IO"

    def extract(self, mesh):
        try:
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError(
                    "ExtractFaceIDsFromMesh expects a segmented TRIMESH from "
                    "P3SAMSegmentMesh."
                )

            face_attributes = getattr(mesh, "face_attributes", None) or {}
            face_ids_array = face_attributes.get("part_id")
            if face_ids_array is None:
                raise ValueError(
                    "Mesh has no face_attributes['part_id']. Connect the mesh "
                    "output of P3SAMSegmentMesh."
                )

            face_ids_array = np.asarray(face_ids_array, dtype=np.int32)
            valid_ids = face_ids_array[face_ids_array >= 0]
            num_parts = int(len(np.unique(valid_ids))) if len(valid_ids) else 0

            face_ids_output = {
                "face_ids": face_ids_array,
                "num_parts": num_parts,
            }

            print(
                f"[ExtractFaceIDsFromMesh] Extracted {len(face_ids_array)} face IDs "
                f"({num_parts} parts)"
            )
            return (face_ids_output,)

        except Exception as e:
            print(f"[ExtractFaceIDsFromMesh] Error extracting face IDs: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


class SaveFaceIDs:
    """
    Save FACE_IDS to JSON or CSV file in ComfyUI output directory.
    Useful for exporting P3-SAM segmentation results for external tools.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_ids": ("FACE_IDS",),
                "filename": ("STRING", {
                    "default": "face_ids.json",
                    "multiline": False,
                    "tooltip": "Output filename. Use .json, .csv, or .npy extension."
                }),
                "format": (["json", "csv", "npy"], {
                    "default": "json",
                    "tooltip": "Export format: json (structured), csv (one face_id per row), or npy (raw numpy array)."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "Hunyuan3D/IO"
    OUTPUT_NODE = True

    def save(self, face_ids, filename, format):
        """Save face IDs to file."""
        try:
            face_ids_array = face_ids['face_ids']
            num_parts = face_ids['num_parts']

            # Ensure correct extension
            expected_ext = f".{format}"
            base = filename.rsplit('.', 1)[0] if '.' in filename else filename
            filename = f"{base}{expected_ext}"

            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, filename)

            if format == "json":
                data = {
                    "num_parts": int(num_parts),
                    "num_faces": len(face_ids_array),
                    "face_ids": face_ids_array.tolist()
                }
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

            elif format == "csv":
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["face_index", "part_id"])
                    for i, part_id in enumerate(face_ids_array):
                        writer.writerow([i, int(part_id)])

            elif format == "npy":
                np.save(output_path, face_ids_array.astype(np.int32))

            print(f"[SaveFaceIDs] Saved {len(face_ids_array)} face IDs "
                  f"({num_parts} parts) to: {output_path}")

            return (output_path,)

        except Exception as e:
            print(f"[SaveFaceIDs] Error saving face IDs: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


class LoadFaceIDs:
    """
    Load FACE_IDS from JSON, CSV, or NPY file.
    Useful for restoring cached P3-SAM segmentation results.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to face_ids JSON, CSV, or NPY file."
                }),
            },
        }

    RETURN_TYPES = ("FACE_IDS",)
    RETURN_NAMES = ("face_ids",)
    FUNCTION = "load"
    CATEGORY = "Hunyuan3D/IO"

    def load(self, file_path):
        """Load face IDs from JSON or CSV file."""
        try:
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"Face IDs file not found: {file_path}")

            ext = Path(file_path).suffix.lower()

            if ext == ".json":
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if 'face_ids' not in data:
                    raise ValueError("Invalid JSON format: missing 'face_ids' key.")

                face_ids_array = np.array(data['face_ids'], dtype=np.int32)
                num_parts = data.get('num_parts',
                                     len(np.unique(face_ids_array[face_ids_array >= 0])))

            elif ext == ".csv":
                face_ids_list = []
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # skip header
                    for row in reader:
                        face_ids_list.append(int(row[1]))

                face_ids_array = np.array(face_ids_list, dtype=np.int32)
                num_parts = len(np.unique(face_ids_array[face_ids_array >= 0]))

            elif ext == ".npy":
                face_ids_array = np.asarray(np.load(file_path), dtype=np.int32)
                if face_ids_array.ndim != 1:
                    raise ValueError(
                        f"Invalid NPY shape: {face_ids_array.shape}. Expected a 1D array of face IDs."
                    )
                num_parts = len(np.unique(face_ids_array[face_ids_array >= 0]))

            else:
                raise ValueError(f"Unsupported file format: {ext}. Use .json, .csv, or .npy")

            face_ids_output = {
                'face_ids': face_ids_array,
                'num_parts': num_parts
            }

            print(f"[LoadFaceIDs] Loaded {len(face_ids_array)} face IDs "
                  f"({num_parts} parts) from: {file_path}")

            return (face_ids_output,)

        except Exception as e:
            print(f"[LoadFaceIDs] Error loading face IDs: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ExtractFaceIDsFromMesh": ExtractFaceIDsFromMesh,
    "SaveFaceIDs": SaveFaceIDs,
    "LoadFaceIDs": LoadFaceIDs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractFaceIDsFromMesh": "Extract Face IDs From Mesh",
    "SaveFaceIDs": "Save Face IDs",
    "LoadFaceIDs": "Load Face IDs",
}
