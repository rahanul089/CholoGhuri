"""
Builds a destination database from the review dataset and scores
destinations against a user's stated preferences.
"""

DESTINATION_NAMES = {
    0: "Cox's Bazar Beach", 1: "Saint Martin's Island", 2: "Sajek Valley",
    3: "Bandarban Hill Tracts", 4: "Rangamati Lake", 5: "Sundarbans Mangrove Forest",
    6: "Kuakata Sea Beach", 7: "Sylhet Tea Gardens", 8: "Ratargul Swamp Forest",
    9: "Bisnakandi", 10: "Jaflong", 11: "Tanguar Haor",
    12: "Paharpur Buddhist Monastery", 13: "Mahasthangarh", 14: "Sonargaon",
    15: "Old Dhaka (Lalbagh Fort)", 16: "Lalbagh Fort", 17: "Ahsan Manzil",
    18: "Srimangal", 19: "Lawachara National Park", 20: "Himchari National Park",
    21: "Nilgiri Hills", 22: "Chimbuk Hill", 23: "Kaptai National Park",
    24: "Bhimruli Floating Market", 25: "Sitakunda Eco Park", 26: "Kantajew Temple",
    27: "Nijhum Dwip", 28: "Char Kukri Mukri", 29: "Panam Nagar", 30: "Comilla",
    31: "Madhabkunda Waterfall", 32: "Cox's Bazar (Forest Retreat)", 33: "Teknaf",
    34: "Barind Museum & Varendra Research", 35: "Foy's Lake", 36: "Patenga Beach",
    37: "Meghla Tourist Complex", 38: "Baldha Garden", 39: "Hazrat Shahjalal Mazar",
    40: "Bagerhat (Shat Gombuj Mosque)", 41: "Panchagarh", 42: "Netrokona",
    43: "Khagrachari", 44: "Rangamati (Chimbuk Hill)", 45: "Cox's Bazar (Bay Residence)",
    46: "Cox's Bazar (Heritage Inn)", 47: "Gazipur (Beach Cottage)",
    48: "Gazipur (Island Resort)", 49: "Tangail", 50: "Mymensingh", 51: "Sherpur",
    52: "Mongla", 53: "Kushtia", 54: "Natore", 55: "Naogaon", 56: "Dinajpur",
    57: "Gaibandha", 58: "Habiganj", 59: "Moulvibazar", 60: "Feni",
    61: "Chattogram", 62: "Dhaka", 63: "Barguna", 64: "Khulna", 65: "Bogura",
}

PERSONA_CATEGORY_MAP = {
    "Cloud Chaser": ["hill"],
    "Beach Baddie": ["beach", "island"],
    "Jungle Junkie": ["forest", "waterfall"],
    "Heritage Nerd": ["heritage", "urban"],
    "Waterfall Hunter": ["waterfall", "hill"],
    "Street Food Explorer": ["urban", "heritage"],
    "Chai & Chill": ["tea-garden", "hill"],
    "Off-Grid Wanderer": ["island", "forest", "beach"],
}


def _bucket(cat_str: str) -> str:
    c = str(cat_str).lower()
    if "beach" in c or "island" in c:
        return "beach"
    if "hill" in c:
        return "hill"
    if "waterfall" in c:
        return "waterfall"
    if "tea" in c:
        return "tea-garden"
    if "forest" in c or "wetland" in c:
        return "forest"
    if "heritage" in c or "urban" in c:
        return "heritage"
    return "beach"


def build_destinations_db(df) -> dict:
    destinations = {}
    for dest_id, group in df.groupby("destination_id"):
        categories = group["preferred_categories"].dropna().unique()
        primary_category = _bucket(categories[0]) if len(categories) > 0 else "beach"

        avg_rating = group["rating"].mean() if "rating" in group.columns else 4.0
        budget = group["preferred_budget"].mode().iloc[0] if "preferred_budget" in group.columns else "medium"
        desc = "Beautiful destination"
        if "liked_most" in group.columns and group["liked_most"].dropna().any():
            desc = group["liked_most"].dropna().iloc[0]

        avg_duration = int(group["trip_duration_days"].mean()) if "trip_duration_days" in group.columns else 3
        duration = f"{avg_duration}-{avg_duration + 2}"
        name = DESTINATION_NAMES.get(dest_id, f"Destination {dest_id}")

        destinations.setdefault(primary_category, []).append({
            "name": name,
            "destination_id": int(dest_id),
            "rating": round(float(avg_rating), 1),
            "budget": budget,
            "duration": duration,
            "desc": str(desc)[:140],
            "num_reviews": int(len(group)),
        })
    return destinations


def recommend(user: dict, destinations_db: dict, top_n=5):
    recommendations = []
    if user["category"] in destinations_db:
        recommendations.extend(destinations_db[user["category"]])

    for cat in PERSONA_CATEGORY_MAP.get(user["persona"], []):
        if cat != user["category"] and cat in destinations_db:
            recommendations.extend(destinations_db[cat])

    scored, seen = [], set()
    for dest in recommendations:
        if dest["name"] in seen:
            continue
        seen.add(dest["name"])

        score = dest["rating"] * 10
        if user["budget"] == dest["budget"]:
            score += 20
        elif user["budget"] == "medium" and dest["budget"] == "low":
            score += 10
        elif user["budget"] == "high" and dest["budget"] in ("medium", "low"):
            score += 5

        try:
            dest_dur = int(str(dest["duration"]).split("-")[0])
            diff = abs(dest_dur - user["duration"])
            score += 10 if diff <= 2 else (5 if diff <= 4 else 0)
        except Exception:
            score += 5

        desc_l = dest["desc"].lower()
        persona_keywords = {
            "Cloud Chaser": ["hill", "cloud"],
            "Beach Baddie": ["beach", "sea"],
            "Jungle Junkie": ["forest", "mangrove", "wildlife"],
            "Heritage Nerd": ["heritage", "history", "fort", "old"],
            "Waterfall Hunter": ["falls", "waterfall"],
            "Street Food Explorer": ["food", "culture", "street"],
            "Chai & Chill": ["tea", "garden"],
            "Off-Grid Wanderer": ["remote", "island", "wild", "peace", "quiet"],
        }
        keywords = persona_keywords.get(user["persona"], [])
        score += 10 if any(k in desc_l for k in keywords) else 5

        scored.append({**dest, "score": min(100, round(score))})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
