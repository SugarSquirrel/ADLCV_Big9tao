"""Preprocess training images for SD3.5 LoRA v9/v10.

Captions written from direct inspection of each raw image.
Rules:
- Trigger token <rfblk> always first.
- NO traffic cones mentioned (prevents model from associating <rfblk> with cones).
- Always describe BOTH large boulders AND smaller rock fragments/debris.
- Describe road surface and lane markings explicitly.
"""

import os
import json
from PIL import Image

RAW_DIR = "data/raw"
OUT_DIR = "data/train"
META_PATH = os.path.join(OUT_DIR, "metadata_v10.jsonl")
IMAGE_SIZE = 512

os.makedirs(OUT_DIR, exist_ok=True)


def center_crop_resize(img, size):
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
    return img.resize((size, size), Image.BICUBIC)


# Per-image captions from direct image inspection, biased toward the target
# generation task: rockfall debris directly on an asphalt driving lane.
# Traffic cones are intentionally omitted. Small rock debris, lane placement,
# and road markings are always mentioned so the model learns rocks-on-road,
# not rocks-as-object or rocks-on-roadside.
captions = [
    "<rfblk>, realistic photograph of a rockfall event on an asphalt mountain road, large fallen boulders and smaller rock fragments scattered directly on the driving lane, visible lane markings, partially blocked road, rocky hillside, outdoor daylight",
    "<rfblk>, realistic photograph of a rockfall event on a wet asphalt mountain road, fallen boulders and broken stones spread from a rocky hillside onto the driving lane, visible road edge line, partially blocked lane, forest mountain setting, outdoor daylight",
    "<rfblk>, realistic photograph of a road rockfall hazard, a large fallen boulder with smaller fragments sitting on an asphalt driving lane, visible yellow center lines, cracked pavement, snowy forest background, outdoor daylight",
    "<rfblk>, realistic photograph of a rockfall event on a curved asphalt forest road, several boulders and loose gravel scattered across the traffic lane, visible center lines, partially blocked road, outdoor natural light",
    "<rfblk>, realistic photograph of a rockfall event on an asphalt road, large angular boulders and smaller stones lying on the paved lane, visible lane marking, rocky roadside slope, partially blocked road, outdoor daylight",
    "<rfblk>, realistic photograph of a rockfall event below a steep cliff, a fallen boulder and gravel debris resting on the asphalt driving lane, visible yellow road line, partially blocked mountain road, outdoor daylight",
    "<rfblk>, realistic photograph of a rockfall event on a coastal mountain road, two fallen boulders and small rock fragments scattered across asphalt traffic lanes, visible road markings, partially blocked lanes, cloudy outdoor light",
    "<rfblk>, realistic photograph of a road rockfall event, a fallen boulder with broken asphalt and rock fragments around it on the driving lane, visible yellow lane markings, rocky hillside background, partially blocked road, outdoor daylight",
    "<rfblk>, realistic photograph of a rockfall event on an asphalt mountain road, multiple boulders and smaller rocks spread across the driving lane, partial yellow center line visible, dense vegetation beside the road, outdoor daylight",
    "<rfblk>, realistic photograph of a coastal road rockfall event, a fallen boulder and rock debris blocking part of an asphalt highway lane, visible yellow center lines, guardrail and rocky cliffside, outdoor daylight",
    "<rfblk>, realistic photograph of a forest road rockfall event, large boulders and smaller stones extending onto the asphalt traffic lane, visible white edge line, pine forest background, partially blocked road, outdoor daylight",
    "<rfblk>, realistic photograph of a rockfall event on a wet asphalt road, a large fallen boulder and smaller flat stones scattered on the driving lane, visible road markings, hillside vegetation, overcast outdoor light",
    "<rfblk>, realistic photograph of a rockfall event near a reinforced hillside, fallen boulders and gravel debris spread across an asphalt mountain road lane, visible lane markings, rockfall protection netting, foggy outdoor light",
    "<rfblk>, realistic photograph of a rockfall event on a curved wet mountain road, one large fallen boulder and small stones scattered along the driving lane, visible yellow center lines, partially blocked road, overcast daylight",
    "<rfblk>, realistic photograph of a rockfall event below a rocky cliff, large boulders and loose debris extending onto an asphalt forest road lane, visible yellow lane markings, partially blocked lane, outdoor daylight"
]

assert len(captions) == 15, f"Caption count mismatch: {len(captions)}"

files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg")))
assert len(files) == 15, f"Expected 15 raw images, found {len(files)}"

metadata = []
for idx, fname in enumerate(files):
    img = Image.open(os.path.join(RAW_DIR, fname)).convert("RGB")
    img = center_crop_resize(img, IMAGE_SIZE)
    new_name = f"{idx+1:03d}.png"
    img.save(os.path.join(OUT_DIR, new_name))
    metadata.append({
        "file_name": f"{OUT_DIR}/{new_name}",
        "caption": captions[idx],
        "source_file": os.path.join(RAW_DIR, fname),
    })

with open(META_PATH, "w", encoding="utf-8") as f:
    for item in metadata:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done! {len(metadata)} images → {META_PATH}")
print("Trigger token: <rfblk> | No traffic cones | Small rock debris included")
