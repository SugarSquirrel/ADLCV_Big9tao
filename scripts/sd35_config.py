"""Config for SD3.5 Medium + LoRA.

Trigger token:
- <rfblk> is the actual learned trigger token.
- rockfall_scene is only a semantic caption tag, not the learned token.

Prompt design rules:
- Under 45 tokens (CLIP-L/G 77-token limit is never hit).
- Lead with <rfblk> then "photograph of" to anchor photorealistic style.
- Always describe BOTH large boulders AND small rock fragments/debris.
- Rocks must be ON the driving lane / traffic lane (not roadside).
- NO traffic cones mentioned (negative prompt blocks them too).
- Describe road surface, lane markings, and environment.
- Avoid nighttime / aerial angle prompts (cause weird camera angles).
"""

PLACEHOLDER_TOKEN = "<rfblk>"
INITIALIZER_TOKEN = "rockfall"
TOKEN_EMBEDDING_FILE = "concept_token_embeddings.pt"

# 10 inference prompts — follow prompt 2's style (user's favorite).
# Pattern: "<rfblk>, realistic photograph of a [WEATHER] rockfall event on
# [ROAD TYPE], [ROCKS] lying across the traffic lane, [ROAD MARKING]"
# All prompts verified under 45 tokens.
# PROMPTS = [
#     "<rfblk>, realistic photograph of a rockfall event on an asphalt mountain road, fallen boulders and small rock fragments lying across the traffic lane, visible yellow lane markings",

#     "<rfblk>, realistic photograph of a rainy rockfall event on a wet asphalt mountain road, several boulders and gravel debris lying across the traffic lane, visible white edge line",

#     "<rfblk>, realistic photograph of a foggy rockfall event on a curved forest road, fallen boulders and loose stones lying across the asphalt traffic lane, visible center lines",

#     "<rfblk>, realistic photograph of an overcast rockfall event below a rocky cliff, boulders and broken stones lying across the asphalt driving lane, visible yellow road markings",

#     "<rfblk>, realistic photograph of a cloudy rockfall event on a winding mountain road, multiple boulders and gravel debris lying across the traffic lane, visible lane markings",

#     "<rfblk>, realistic photograph of a coastal road rockfall event, fallen boulders and small rock fragments lying across an asphalt highway lane, visible yellow center lines",

#     "<rfblk>, realistic photograph of a hazy rockfall event below a rocky hillside, boulders and loose stone debris lying across the asphalt traffic lane, visible lane markings",

#     "<rfblk>, realistic photograph of a drizzly rockfall event on a forest mountain road, a large boulder and small stones lying across the driving lane, visible white edge line",

#     "<rfblk>, realistic photograph of a rockfall event near a reinforced hillside, boulders and gravel debris lying across an asphalt road lane, visible yellow lane markings",

#     "<rfblk>, realistic photograph of an overcast rockfall event on a winding asphalt road, fallen boulders and smaller rock fragments lying across the traffic lane, visible road markings"
# ]

''' 10 inference prompts — smoke2 feedback applied.
Good (keep): P1, P3, P8
Fixed (perspective + rocks-on-lane): P2, P4, P5, P9
Replaced entirely: P6, P7, P10
'''
PROMPTS = [
    # P1 ✅ keep — rainy, "lying across the traffic lane" works well
    "<rfblk>, realistic photograph of a rainy rockfall event on a wet asphalt mountain road, several boulders and gravel debris lying across the traffic lane, visible white edge line",

    # P2 — boulders as primary subject, fragments secondary; driver's perspective
    "<rfblk>, realistic photograph of a sudden rockfall, driver's perspective of large fallen boulders with scattered rock fragments on the asphalt lane surface ahead, visible road markings, sharp detail",

    # P3 — night road hazard photo; vehicle headlights light up rocks on asphalt
    "<rfblk>, nighttime road hazard photograph, fallen boulders and rocks on the asphalt surface lit by vehicle headlights, high contrast, dark mountain road surroundings, raw photographic detail",

    # P4 fix — driver's perspective replaces "road continuing behind"
    "<rfblk>, realistic photograph of an overcast mountain road rockfall, driver's perspective of fallen boulders and gravel debris blocking the driving lane ahead, visible white edge line",

    # P5 fix — remove "roadside view" (wrong angle cue); anchor to "lane ahead"
    "<rfblk>, realistic photograph of a rainy road rockfall, driver's perspective of large boulders and stones covering the asphalt lane ahead, wet road surface, visible road markings",

    # P6 replace — cliff + cloudy; strong perspective + rocks blocking road
    "<rfblk>, realistic photograph of a cloudy rockfall below a steep rocky cliff, driver's perspective of large boulders and rock fragments blocking the mountain lane ahead, yellow center lines",

    # P7 replace — dashcam style forces rocks-on-road; "resting on" is explicit
    "<rfblk>, dashcam still of a forest mountain road rockfall, large boulders and rock fragments resting on the asphalt road surface blocking the lane ahead, white edge line visible",

    # P8 ✅ keep — "front road view" anchor works well
    "<rfblk>, realistic photograph of an overcast road rockfall event, several large boulders and smaller rock fragments lying across both lanes, visible yellow center lines, front road view",

    # P9 fix — front road view anchor; explicit "covering the lane surface"
    "<rfblk>, realistic photograph of a wet mountain road rockfall, front road view of fallen boulders and rock debris covering the lane surface ahead, visible white edge line",

    # P10 — distinct fallen boulders (not debris mass); road surface still visible
    "<rfblk>, dashcam still of a mountain road below a rocky cliff, several distinct fallen boulders and broken rock fragments on the asphalt lane, surrounding road surface still visible, overcast daylight"
]

# Setting A — CLIP-L/G hard limit is 77 tokens (incl. BOS/EOS = 75 usable).
# Previous version was ~74 tokens → "text, watermark" got truncated.
# This version targets ~40 tokens, well within the limit.
NEGATIVE_PROMPT = (
    "cartoon, 3d render, cgi, painting, fantasy, "
    "empty road, no rocks on road, rocks on roadside only, traffic cone, "
    "aerial view, top-down view, floating rock, blurry, watermark"
)

# Setting B (8 steps) — shorter for few-step stability.
FEW_STEP_NEGATIVE_PROMPT = (
    "cartoon, 3d render, cgi, empty road, stream, traffic cone, "
    "giant single rock, close-up, roadside only, clear driving lane, "
    "floating rock, blurry, text, watermark"
)
