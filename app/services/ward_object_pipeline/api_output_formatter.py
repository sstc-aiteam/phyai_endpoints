# api_output_formatter.py

import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError as e:
    raise ImportError(
        "pycocotools is required for compressed COCO RLE. "
        "Install it with: pip install pycocotools"
    ) from e


def binary_mask_to_compressed_rle(mask):
    """
    Convert binary mask to compressed COCO RLE.

    Output format:
        {
            "size": [height, width],
            "counts": "compressed_string"
        }

    Notes:
        - COCO RLE expects Fortran-contiguous array.
        - pycocotools returns counts as bytes, so we decode to UTF-8
          for JSON serialization.
    """

    mask = mask.astype(np.uint8)

    # COCO RLE requires Fortran order
    mask_fortran = np.asfortranarray(mask)

    rle = mask_utils.encode(mask_fortran)

    # pycocotools returns bytes for "counts"; JSON needs string
    rle["counts"] = rle["counts"].decode("utf-8")

    return {
        "size": [int(x) for x in rle["size"]],
        "counts": rle["counts"]
    }


def compressed_rle_to_binary_mask(rle):
    """
    Optional debug helper.
    Decode compressed COCO RLE back to binary mask.

    Args:
        rle:
            {
                "size": [height, width],
                "counts": "compressed_string"
            }

    Returns:
        HxW uint8 binary mask.
    """

    rle_for_decode = {
        "size": rle["size"],
        "counts": rle["counts"].encode("utf-8")
    }

    mask = mask_utils.decode(rle_for_decode)

    return mask.astype(np.uint8)


def format_api_output(classified_items):
    """
    Minimal API output.

    Exposes only:
        - class_name
        - bbox
        - compressed RLE mask

    Output:
        {
            "success": True,
            "num_objects": N,
            "objects": [
                {
                    "class_name": str,
                    "bbox": [x1, y1, x2, y2],
                    "rle_mask": {
                        "size": [height, width],
                        "counts": "compressed_string"
                    }
                }
            ]
        }
    """

    objects = []

    for item in classified_items:
        if "mask" not in item:
            raise KeyError(
                "Each classified item must contain a 'mask' field."
            )

        if "bbox" not in item:
            raise KeyError(
                "Each classified item must contain a 'bbox' field."
            )

        mask = item["mask"].astype(bool)

        obj = {
            "class_name": item.get("final_class", "unknown"),

            "bbox": [
                float(x)
                for x in item["bbox"]
            ],

            "rle_mask": binary_mask_to_compressed_rle(mask)
        }

        objects.append(obj)

    return {
        "success": True,
        "num_objects": len(objects),
        "objects": objects
    }


def format_failure_output(reason):
    """
    Format failed pipeline output.
    """

    return {
        "success": False,
        "reason": str(reason),
        "num_objects": 0,
        "objects": []
    }
