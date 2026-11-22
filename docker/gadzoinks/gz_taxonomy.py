"""The generic layer: what's true of the *tagger*, not of anyone's library.

A curated vocabulary.yaml doesn't travel — it encodes one person's subjects.
But most of the work that file was doing isn't personal at all, and can be
derived mechanically from two facts:

  1. Danbooru labels are morphologically regular. `blonde_hair`, `black_hair`,
     `long_hair` are one pattern; `white_shirt`, `red_dress` are another. A
     few dozen suffix rules generate hierarchy for thousands of labels, with
     no reference to any library.

  2. Some labels are never subjects. Counts (`1girl`, `6+boys`), framing
     (`cowboy_shot`, `from_above`), ratings, meta (`simple_background`), body
     descriptors. Those are properties of the label set, so the stop list
     ships with the product.

What's left — which of the structured labels matter in THIS library — is
answered by measurement rather than opinion: see `in_band`. A tag on 78% of
assets doesn't discriminate; a tag on 0.2% is noise. The useful band is
narrow, and finding it needs no curation at all.

So the pipeline is:

    label set ──(morphology + stop list)──> structure      ships
    library   ──(frequency band)──────────> relevance      measured
                                                ↓
                                      generated vocabulary
                                                ↓
                                     user overlay (small)

The user's own file stops being 128 rules and becomes the handful of things
the generic layer got wrong for them.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

# ---------------------------------------------------------------------------
# Not subjects. A property of the label set, so this ships.
# ---------------------------------------------------------------------------

STOP_EXACT = {
    # counts and subject framing
    "solo", "solo_focus", "no_humans", "multiple_views",
    # posture and gaze — true of the shot, not the subject
    "sitting", "standing", "lying", "kneeling", "squatting", "walking",
    "running", "flying", "floating", "holding", "looking_at_viewer",
    "looking_away", "looking_back", "looking_down", "looking_up",
    "closed_eyes", "closed_mouth", "open_mouth", "smile", "frown", "blush",
    "parted_lips", "teeth", "tongue", "lips",
    # framing and crop
    "cowboy_shot", "upper_body", "lower_body", "full_body", "close-up",
    "from_above", "from_below", "from_side", "from_behind", "dutch_angle",
    "wide_shot", "profile", "facing_viewer", "facing_away", "straight-on",
    # capture artefacts
    "blurry", "blurry_background", "blurry_foreground", "depth_of_field",
    "motion_blur", "chromatic_aberration", "film_grain", "jpeg_artifacts",
    "letterboxed", "border", "artist_name", "signature", "watermark",
    "web_address", "character_name", "copyright_name", "dated", "logo",
    # body descriptors
    "breasts", "large_breasts", "medium_breasts", "small_breasts",
    "huge_breasts", "flat_chest", "cleavage", "navel", "midriff", "thighs",
    "armpits", "collarbone", "ribs", "muscular", "curvy", "wide_hips",
    "dark_skin", "pale_skin", "tan", "freckles",
    "dark-skinned_female", "dark-skinned_male", "very_dark_skin", "asian",
    "real_world_location",
    # ratings (also handled by drop_categories; belt and braces)
    "general", "sensitive", "questionable", "explicit",
}

STOP_PATTERNS = [
    re.compile(p)
    for p in (
        r"^\d+\+?(girls?|boys?|others?)$",     # 1girl, 6+boys
        r"^(multiple|group)_",                  # multiple_girls
        r"_focus$",                             # male_focus, animal_focus
        r"_background$",                        # simple_background, photo_background
        r"^(looking|holding|reaching|touching)_",
        r"_(request|only)$",
        r"^commentary",
        r"_commentary$",
    )
]

# ---------------------------------------------------------------------------
# Morphology. Also ships: it describes the label set's own regularities.
# ---------------------------------------------------------------------------

STYLE_QUALIFIERS = {
    "pleated", "pencil", "striped", "plaid", "floral", "print", "printed",
    "torn", "open", "closed", "sleeveless", "strapless", "long", "short",
    "fingerless", "elbow", "high", "low", "layered", "oversized", "tight",
    "loose", "checkered", "diagonal-striped", "cow_print", "leopard_print",
    "hoop", "stud", "dangle", "pearl", "wedding", "multiple", "single",
    "denim", "leather", "silk", "lace", "wool", "cotton", "satin",
    "velvet", "fur", "hawaiian", "tartan", "tweed", "linen", "mesh",
}

COLOURS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "silver", "gold", "blonde", "blond",
    "aqua", "teal", "cyan", "magenta", "violet", "tan", "beige", "navy",
    "crimson", "scarlet", "turquoise", "lavender", "maroon", "olive",
    "multicolored", "rainbow", "light_brown", "dark_blue", "dark_green",
}

GARMENTS = {
    "shirt", "t-shirt", "blouse", "dress", "gown", "skirt", "pants",
    "trousers", "jeans", "shorts", "jacket", "coat", "cardigan", "sweater",
    "hoodie", "vest", "suit", "uniform", "robe", "cloak", "cape", "apron",
    "swimsuit", "bikini", "lingerie", "leotard", "bodysuit", "kimono",
    "hat", "cap", "helmet", "hood", "scarf", "gloves", "socks", "stockings",
    "pantyhose", "boots", "shoes", "heels", "sandals", "sneakers", "belt",
    "tie", "necktie", "bowtie", "armor", "crop_top", "tank_top", "bra",
    "footwear", "headwear", "swimwear", "leather", "lingerie",
    "high_heels", "high_heel_boots",
    "baseball_cap", "beanie", "visor_cap", "sun_hat", "cowboy_hat",
    "fedora", "top_hat", "beret", "peaked_cap", "military_hat", "chef_hat",
    "headband", "hairband", "bandana", "sneakers", "loafers", "slippers",
    "flip-flops", "sportswear", "school_uniform", "baseball_uniform",
    "soccer_uniform", "basketball_uniform", "tennis_uniform",
    "police_uniform", "military_uniform", "collared_shirt", "polo_shirt",
    "long_sleeves", "short_sleeves", "sleeveless", "denim", "plaid",
    "striped_clothes", "winter_clothes", "swim_trunks", "male_swimwear",
    "jumpsuit", "blazer", "jersey", "towel", "wristband", "knee_pads",
    "elbow_pads", "boxing_gloves", "diving_suit", "camouflage", "costume",
}

JEWELLERY = {
    "earrings", "necklace", "bracelet", "ring", "anklet", "pendant",
    "choker", "brooch", "tiara", "crown", "hair_ornament", "piercing",
    "armlet", "wristwatch", "watch",
}

ANIMALS = {
    "bird", "cat", "dog", "horse", "fish", "shark", "whale", "dolphin",
    "turtle", "snake", "lizard", "frog", "insect", "butterfly", "bee",
    "spider", "crab", "octopus", "squid", "jellyfish", "seahorse",
    "starfish", "eel", "ray", "stingray", "cow", "pig", "sheep", "goat",
    "deer", "bear", "wolf", "fox", "rabbit", "mouse", "monkey", "elephant",
    "lion", "tiger", "penguin", "owl", "eagle", "duck", "chicken",
    "penguin", "squirrel", "deer", "rabbit", "boar", "goat", "rooster",
    "crow", "sparrow", "seagull", "parrot", "shiba_inu", "dinosaur",
    "mouse", "wolf", "tiger", "bear", "bug", "cow",
}

MARINE = {
    "coral", "coral_reef", "reef", "anemone", "kelp", "seaweed", "shipwreck",
    "seabed", "bubble", "air_bubble", "wetsuit", "snorkel", "diving_mask",
    "diver", "scuba_gear", "oxygen_tank", "fins", "diving_fins",
    "reef", "lagoon", "atoll", "tide_pool",
}

SCIFI = {
    "spacecraft", "spaceship", "space_station", "rocket", "satellite",
    "astronaut", "space_suit", "mecha", "robot", "android", "cyborg",
    "alien", "laser", "hologram", "portal", "cockpit", "control_panel",
    "science_fiction", "cyberpunk", "dystopia", "futuristic",
    "planet", "earth", "space", "galaxy", "nebula", "asteroid", "comet",
}

# Sets below were filled from a 2,000-image COCO scan: the labels WD actually
# produces on general photography. The scan is used to say *which terms a
# vocabulary needs a slot for*, never to decide what the vocabulary means.

FOOD = {
    "food", "fruit", "vegetable", "meat", "pizza", "pizza_slice", "cake",
    "cake_slice", "birthday_cake", "bread", "bread_slice", "rice", "noodles",
    "pasta", "spaghetti", "ramen", "curry", "salad", "sandwich", "burger",
    "french_fries", "doughnut", "cookie", "pastry", "dessert", "steak",
    "sausage", "egg", "omelet", "omurice", "apple", "banana", "tomato",
    "lemon", "lemon_slice", "strawberry", "carrot", "cabbage", "lettuce",
    "onion", "cucumber", "broccoli", "grapes", "watermelon", "cherry",
    "pumpkin", "spring_onion", "shrimp", "chicken", "fried_chicken",
    "fast_food", "whipped_cream", "sauce", "sprinkles", "candy_cane", "milk",
    "coffee", "tea", "beer", "wine", "whiskey", "alcohol", "soda", "drink",
    "ice", "ice_cube",
}

PEOPLE = {
    "child", "baby", "crowd", "people", "family", "old_man", "old_woman",
    "soldier", "army", "military", "police", "policeman", "chef", "pilot",
    "maid", "band",
}

FACIAL_HAIR = {
    "beard", "mustache", "facial_hair", "stubble", "beard_stubble",
    "sideburns", "chest_hair", "arm_hair", "leg_hair", "thick_eyebrows",
}

HAIRSTYLE = {
    "bald", "braid", "twin_braids", "buzz_cut", "ponytail", "hair_bun",
    "single_hair_bun", "afro", "dreadlocks", "very_short_hair",
    "medium_hair", "multicolored_hair", "two-tone_hair", "blunt_bangs",
}

COSMETICS = {"makeup", "lipstick", "nail_polish", "eyeshadow", "mascara"}

TATTOO = {"tattoo", "arm_tattoo", "back_tattoo", "face_tattoo"}

EYEWEAR = {
    "glasses", "sunglasses", "goggles", "ski_goggles", "round_eyewear",
    "black-framed_eyewear", "monocle",
}

TABLEWARE = {
    "cup", "mug", "plate", "bowl", "bottle", "glass", "wine_glass", "teapot",
    "fork", "knife", "spoon", "chopsticks", "tray", "pitcher", "jar", "can",
    "drinking_glass", "teacup", "coffee_mug", "coffee_cup", "saucer",
    "ladle", "spatula", "frying_pan", "cooking_pot", "cutting_board",
    "kettle", "tongs", "disposable_cup", "water_bottle", "soda_can",
    "beer_can", "drink_can", "wine_bottle", "beer_bottle", "soda_bottle",
    "napkin", "tablecloth",
}

FURNITURE = {
    "table", "chair", "couch", "sofa", "bed", "bench", "desk", "stool",
    "shelf", "bookshelf", "cabinet", "wardrobe", "window", "door", "mirror",
    "lamp", "curtain", "rug", "pillow", "clock", "television", "sink",
    "toilet", "refrigerator", "oven", "microwave",
    "ceiling_light", "desk_lamp", "wall_clock", "alarm_clock",
    "digital_clock", "analog_clock", "carpet", "cushion", "blanket",
    "picture_frame", "chalkboard", "locker", "drawer", "cupboard",
    "counter", "bathtub", "claw_foot_bathtub", "shower_head", "faucet",
    "toilet_paper", "trash_can", "washing_machine", "stove", "fireplace",
    "chandelier", "window_blinds", "bookshelf", "stool", "swivel_chair",
    "office_chair", "beach_chair", "park_bench", "hammock", "wooden_table",
    "electric_fan", "throne", "bean_bag",
}

URBAN = {
    "pillar", "column", "archway",
    "road", "sidewalk", "crosswalk", "parking_lot", "traffic_light",
    "street_light", "sign", "fence", "wall", "stairs", "railing", "pole",
    "bus_stop", "train_station", "airport",
    "lamppost", "power_lines", "utility_pole", "road_sign",
    "traffic_cone", "chain-link_fence", "wooden_fence", "stone_wall",
    "brick_wall", "railroad_tracks", "railroad_crossing", "license_plate",
    "vending_machine", "menu_board", "menu", "graffiti", "neon_lights",
    "city_lights", "dock", "pier", "path", "alley", "village", "rural",
    "cityscape", "chimney", "arch", "tower", "clock_tower", "bridge",
}

FANTASY = {
    "fantasy", "monster", "giant", "dragon", "magic", "fairy", "unicorn",
    "elf", "demon", "angel", "wizard", "potion", "rune", "mermaid",
    # Creature features. These read as anime tells out of context, but in a
    # render library they are what the image is *of*.
    "wings", "horns", "fangs", "tentacles", "pointy_ears", "extra_arms",
    "claws", "tail", "scales", "oversized_animal",
}

# Cultural and period motifs. A style marker, not a claim about anyone's
# ethnicity — `egyptian` describes the set dressing.
CULTURE = {
    "egyptian", "japanese_clothes", "east_asian_architecture",
    "chinese_clothes", "vietnamese_dress", "roman", "greek", "viking",
    "aztec", "mayan", "nordic", "celtic", "arabian", "indian", "gothic",
    "victorian", "medieval", "steampunk", "art_deco",
}

# Non-natural skin colouring. Separate from the identity terms, which stay
# blocked: `green_skin` is a costume or a species, `dark_skin` is a person.
SKIN = {
    "colored_skin", "green_skin", "blue_skin", "purple_skin", "red_skin",
    "grey_skin", "white_skin_(fantasy)", "gold_skin",
}

PLACES = {
    "beach", "city", "building", "architecture", "ruins", "forest", "jungle",
    "mountain", "desert", "field", "meadow", "cave", "bridge", "road",
    "street", "alley", "rooftop", "indoors", "outdoors", "underwater",
    "church", "temple", "castle", "market", "harbor", "pier", "garden",
    "island", "peninsula", "cliff", "valley", "canyon", "glacier",
    "volcano", "dune", "plateau", "fjord", "oasis",
    "bathroom", "kitchen", "bedroom", "restaurant", "office", "shop",
    "cafe", "bar_(place)", "stadium", "train_station", "church", "castle",
    "locker_room", "toilet_stall", "shower_(place)", "car_interior",
    "train_interior", "interior", "messy_room", "jungle", "hill",
    "meadow", "shore", "park", "windmill", "grave", "tombstone",
}

NATURE = {
    "tree", "plant", "flower", "grass", "leaf", "bush", "moss", "snow",
    "rain", "fog", "mist", "fire", "smoke", "rock", "sand", "scenery",
    "leaf", "petals", "rose", "tulip", "hydrangea", "cherry_blossoms",
    "autumn_leaves", "moss", "fern", "wheat", "bush", "palm_tree",
    "pine_tree", "branch", "bare_tree", "puddle", "steam", "smoke",
    "bamboo", "cactus", "vine", "ivy", "reed", "lily_pad", "driftwood",
    "wind", "overgrown", "tree_shade", "shade", "dappled_sunlight",
    "potted_plant", "christmas_tree",
}

WATER = {"ocean", "sea", "river", "lake", "pond", "waterfall", "stream", "waves", "reflective_water", "partially_submerged", "shore",
}

SKY = {"sky", "cloud", "clouds", "sunset", "sunrise", "night", "stars",
       "star",
       "moon", "sun", "rainbow", "lightning", "aurora",
       "clear_sky", "cloudy_sky", "grey_sky", "night_sky", "full_moon",
    "sunrise", "overcast", "snowing", "rain", "fog", "horizon",
    "mountainous_horizon",
}

MEDIUM = {
    "photorealistic", "realistic", "photo", "traditional_media",
    "monochrome", "greyscale", "grayscale", "sketch", "watercolor",
    "oil_painting", "3d", "pixel_art", "lineart", "silhouette",
    "lens_flare", "bokeh", "backlighting", "rim_light", "god_rays",
    "still_life", "portrait_(style)", "candid", "long_exposure",
    "painting", "watercolor", "colored_pencil", "acrylic_paint", "graphite",
    "traditional_media", "photorealistic", "realistic",
    "sepia", "film_grain", "spot_color", "gradient_background",
    "vanishing_point", "reflection", "shadow", "sunlight", "light_rays",
    "depth_of_field", "motion_blur", "blurry_foreground", "wide_shot",
    "painting_(medium)", "watercolor_(medium)", "colored_pencil_(medium)",
    "acrylic_paint_(medium)", "graphite_(medium)", "painting_(object)",
    "drawing_(object)", "poster_(object)", "surreal", "horror_(theme)",
}

# Flat sets mapped to a namespace. Order matters only for readability.
VEHICLES = {
    "motor_vehicle", "vehicle", "car", "truck", "van", "bus", "motorcycle",
    "scooter", "bicycle", "train", "tram", "aircraft", "airplane", "jet",
    "helicopter", "boat", "ship", "watercraft", "sports_car", "racecar",
    "police_car", "military_vehicle", "fighter_jet", "warship",
}

OBJECTS = {
    "fireworks", "water_gun", "hand_fan", "fan", "balloon", "kite",
    "lantern", "torch", "banner", "scroll", "map", "globe", "telescope",
    "binoculars", "compass", "hourglass", "mask", "bag", "basket",
    "bucket", "box", "leash", "cable", "rope", "chain", "bell",
}

ACTIVITIES = {
    "riding", "surfing", "swimming", "cooking", "eating", "reading",
    "fishing", "horseback_riding", "riding_bicycle", "dancing", "sleeping",
    "smoking", "brushing_teeth", "bathing", "driving", "playing_games",
    "selfie", "taking_picture", "wading", "carrying", "baseball", "soccer",
    "basketball", "tennis", "picnic", "christmas", "halloween",
}


SET_RULES: list[tuple[str, set[str]]] = [
    ("appearance/facial-hair", FACIAL_HAIR),
    ("appearance/hairstyle", HAIRSTYLE),
    ("appearance/eyewear", EYEWEAR),
    ("appearance/cosmetics", COSMETICS),
    ("appearance/tattoo", TATTOO),
    ("food", FOOD),
    ("person", PEOPLE),
    ("appearance/jewelry", JEWELLERY),
    ("appearance/clothing", GARMENTS),
    ("animal", ANIMALS),
    ("marine", MARINE),
    ("scifi", SCIFI),
    ("fantasy", FANTASY),
    ("culture", CULTURE),
    ("appearance/skin", SKIN),
    ("place", PLACES),
    ("nature", NATURE),
    ("water", WATER),
    ("sky", SKY),
    ("object/vehicle", VEHICLES),
    ("medium", MEDIUM),
    ("object", OBJECTS),
    ("activity", ACTIVITIES),
    ("tableware", TABLEWARE),
    ("furniture", FURNITURE),
    ("urban", URBAN),
]



# Suffix patterns: (regex, template). \g<1> is the captured qualifier.
SUFFIX_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(.+)_hair$"), r"appearance/hair/\g<1>"),
    (re.compile(r"^(.+)_eyes$"), r"appearance/eyes/\g<1>"),
    (re.compile(r"^(.+)_sky$"), "sky"),
    (re.compile(r"^(.+)_hat$"), "appearance/clothing/hat"),
    (re.compile(r"^(.+)_skin$"), r"appearance/skin/\g<1>"),
]

# Coarse labels worth keeping only when nothing more specific matched.
FALLBACKS = {
    "jewelry": "jewelry",
    "animal": "animal",
    "nature": "nature",
    "water": "water",
    "clothing": "clothing",
    "weapon": "object/weapon",
    "motor_vehicle": "object/vehicle",
    "vehicle": "object/vehicle",
}


BLOCK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("identity", re.compile(
        r"^(asian|dark[-_]skin|dark-skinned|very_dark_skin|pale_skin|tan)"
    )),
    # Short alternatives must be anchored to word boundaries. Unanchored,
    # `ass` matched glass, grass, brass and class — `drinking_glass` was
    # being dropped as a body label.
    ("body", re.compile(
        r"(?:^|_)(breasts|nipples|cleavage|navel|midriff|panties|underwear|"
        r"nude|topless|ass|thighs|thighhighs|armpits|pectorals|loli|yaoi|"
        r"bara|furry|garter|crossdressing|shiny_skin)(?:$|_)"
    )),
    ("geography", re.compile(
        r"^(tokyo|japan|united_states|china|korea|america|europe)"
    )),
    ("meta", re.compile(
        r"(watermark|signature|artist_name|username|web_address|logo|"
        r"copyright_name|character_name|dated|border|letterboxed|"
        r"_inset$|^photo_\(object\)|qr_code|product_placement|"
        r"^2koma$|^comic$|retro_artstyle|_\(style\)$|parody|meme|^what$|"
        r"^too_many$|^everyone$|^figure$|model_kit)"
    )),
    ("brand", re.compile(
        r"^(nike|adidas|honda|iphone|pokemon|virtual_youtuber)"
    )),
]

BLOCK_EXACT = {
    "yuuki_juudai", "chen", "pikachu", "ikari_shinji", "kageyama_shigeo",
    "duel_academy_uniform_(yu-gi-oh!_gx)", "kita_high_school_uniform",
    "jujutsu_tech_uniform", "aria_company_uniform", "plugsuit",
    "magical_musket", "serafuku", "gakuran", "twintails", "drill_hair",
    "cat_ears", "animal_ears", "flame-tipped_tail", "cow_tail",
    "sheep_horns", "mini_person", "miniboy", "clothed_animal", "crossover",
    "multiple_crossover", "k-pop", "maid", "maid_apron", "pirate",
    "jolly_roger", "santa_claus", "reindeer", "heterochromia",
    "glowing_eyes", "ambiguous_gender", "hetero", "couple", "kiss",
    "husband_and_wife", "mother_and_daughter", "father_and_daughter",
}


def is_blocked(label: str) -> str | None:
    """Policy exclusions, reported as such rather than as unmapped.

    These never enter a vocabulary automatically, however often the label
    fires or how confident the model is. An explicit entry in the
    vocabulary file still wins — the block governs what gets added for you,
    not what you may choose to add yourself.
    """
    if label in BLOCK_EXACT:
        return "characters"
    for reason, pattern in BLOCK_PATTERNS:
        if pattern.search(label):
            return reason
    return None


def strip_qualifier(label: str) -> str:
    """`earth_(planet)` -> `earth`; Danbooru's disambiguation suffix."""
    return re.sub(r"_?\([^)]*\)$", "", label).strip("_")


def is_stop(label: str) -> bool:
    bare = strip_qualifier(label)
    if bare in STOP_EXACT or label in STOP_EXACT:
        return True
    return any(p.search(label) for p in STOP_PATTERNS)


def structure(label: str) -> str | None:
    """A path for this label from the label set alone, or None if unknown.

    Colour-qualified garments fold to the garment: `white_shirt` and
    `red_shirt` are both `clothing/shirt`, because shirt colour is almost
    never the thing being searched for. Hair colour does NOT fold, because
    it is.
    """
    if is_stop(label):
        return None

    bare = strip_qualifier(label)

    # Colour-qualified: strip the colour and recurse on the noun, except for
    # the suffix rules below which want the qualifier kept.
    parts = bare.split("_", 1)
    if (
        len(parts) == 2
        and (parts[0] in COLOURS or parts[0] in STYLE_QUALIFIERS)
        and not bare.endswith(("_hair", "_eyes", "_skin"))
    ):
        inner = structure(parts[1])
        if inner:
            return inner

    for pattern, template in SUFFIX_RULES:
        if pattern.match(bare):
            return pattern.sub(template, bare)

    for namespace, members in SET_RULES:
        if bare in members:
            # Namespaces whose own name is the member collapse to the bare
            # namespace: `sky` -> `sky`, not `sky/sky`.
            leaf = bare.replace("_", "-")
            return namespace if bare == namespace else f"{namespace}/{leaf}"

    return None


# ---------------------------------------------------------------------------
# Relevance, measured. Nothing library-specific is hard-coded.
# ---------------------------------------------------------------------------

def in_band(count: int, total: int, low: float, high: float) -> bool:
    """Does this label discriminate within this library?

    Too rare and it's noise you'll never browse by. Too common and it says
    nothing: a tag on 95% of assets can't narrow anything down. Both ends
    are library-relative, which is exactly why this transfers when a
    hand-written list doesn't.

    The two cuts are not equally safe. The low cut is uncontroversial — a
    label on 3 of 50,000 assets is noise by any standard. The high cut is a
    judgement: an all-scuba library legitimately has `underwater` on
    everything, and the owner may still want the tag even though it filters
    nothing. So `high` defaults loose and is meant to be tuned per library,
    while `low` can mostly be left alone.
    """
    share = count / total if total else 0.0
    return low <= share <= high


def band_report(
    counts: Counter[str], total: int, low: float, high: float
) -> dict[str, list[tuple[str, int]]]:
    """Splits labels into in-band / too-rare / too-common, each by frequency."""
    out: dict[str, list[tuple[str, int]]] = {"keep": [], "rare": [], "common": []}
    for label, count in counts.most_common():
        share = count / total if total else 0.0
        if share > high:
            out["common"].append((label, count))
        elif share < low:
            out["rare"].append((label, count))
        else:
            out["keep"].append((label, count))
    return out


def suggest(
    counts: Counter[str],
    total: int,
    low: float = 0.01,
    high: float = 0.80,
) -> dict[str, object]:
    """Everything needed to write a vocabulary file for this library.

    Returns the structured rules, the in-band labels morphology couldn't
    place (the only part needing a human), and the labels excluded by
    frequency so the exclusions are auditable rather than invisible.
    """
    # Stops come out before banding, or they pollute the "too common"
    # report with labels that were never candidates — `1girl` isn't
    # excluded for being common, it's excluded for not being a subject.
    stopped = [(l, c) for l, c in counts.most_common() if is_stop(l)]
    candidates = Counter({l: c for l, c in counts.items() if not is_stop(l)})
    bands = band_report(candidates, total, low, high)

    rules: dict[str, str] = {}
    unplaced: list[tuple[str, int]] = []

    for label, count in bands["keep"]:
        path = structure(label)
        if path:
            rules[label] = path
        else:
            unplaced.append((label, count))

    return {
        "rules": rules,
        "fallbacks": {k: v for k, v in FALLBACKS.items() if k in counts},
        "unplaced": unplaced,
        "too_rare": bands["rare"],
        "too_common": bands["common"],
        "stopped": stopped,
        "band": (low, high),
    }


def write_vocabulary(result: dict, path: str, root: str = "ai") -> None:
    """Emits a commented vocabulary.yaml. Generated, then edited by hand."""
    rules: dict[str, str] = result["rules"]  # type: ignore[assignment]
    low, high = result["band"]  # type: ignore[misc]

    by_namespace: dict[str, list[tuple[str, str]]] = {}
    for label, target in sorted(rules.items()):
        by_namespace.setdefault(target.split("/")[0], []).append((label, target))

    lines = [
        "# Generated by `survey.py suggest` from measured label frequencies.",
        "# Edit freely — this is a starting point, not an output format.",
        "#",
        f"# Kept labels firing on {low:.0%}-{high:.0%} of assets. Rarer than that",
        "# is long tail you'd never browse by; more common than that doesn't",
        "# discriminate. Both cuts are relative to this library.",
        "",
        f"root: {root}",
        "drop_categories: [rating]",
        "unlisted: drop",
        "",
        "fallback_labels:",
    ]
    for label, target in sorted(result["fallbacks"].items()):  # type: ignore[union-attr]
        lines.append(f"  {label + ':':<22} {target}")

    lines += ["", "labels:"]
    for namespace in sorted(by_namespace):
        lines.append(f"\n  # -- {namespace}")
        for label, target in by_namespace[namespace]:
            lines.append(f"  {label + ':':<30} {target}")

    if result["unplaced"]:
        lines += [
            "",
            "  # -- no structural rule matched; these need a decision -------------",
            "  # Give each a path, or `~` to drop. Frequency in the comment.",
        ]
        for label, count in result["unplaced"]:  # type: ignore[union-attr]
            lines.append(f"  # {label + ':':<28} ~   # fired on {count} assets")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
