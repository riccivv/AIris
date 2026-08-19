LOCATIONS_DB = [
    {
        "keywords": ["cafeteria", "canteen", "dining", "food", "eat"],
        "name": "Main Cafeteria",
        "details": "Ground Floor, East Wing",
        "route": [
            {"instruction": "Walk straight for 50 meters", "distance_meters": 50, "turn": "Turn right"},
            {"instruction": "Turn right in 20 meters", "distance_meters": 20, "turn": "Turn left"},
            {"instruction": "Turn left in 10 meters to enter Main Cafeteria", "distance_meters": 10, "turn": "You have arrived at Main Cafeteria"}
        ]
    },
    {
        "keywords": ["library", "study", "books"],
        "name": "Central Library",
        "details": "2nd Floor, Main Building",
        "route": [
            {"instruction": "Walk straight for 500 meters", "distance_meters": 500, "turn": "Turn left"},
            {"instruction": "Turn left in 100 meters towards the stairs", "distance_meters": 100, "turn": "Take the stairs up"},
            {"instruction": "Go up stairs and walk 30 meters", "distance_meters": 30, "turn": "You have arrived at Central Library"}
        ]
    },
    {
        "keywords": ["restroom", "toilet", "bathroom", "washroom"],
        "name": "Restroom",
        "details": "Ground Floor, West Corridor",
        "route": [
            {"instruction": "Walk straight for 30 meters", "distance_meters": 30, "turn": "Turn right"},
            {"instruction": "Turn right in 15 meters", "distance_meters": 15, "turn": "You have arrived at Restroom"}
        ]
    }
]


def rag_lookup_destination(query):
    """Retrieves matched destination object from the knowledge base using keyword fuzzy matching."""
    query_clean = query.lower()
    for item in LOCATIONS_DB:
        for kw in item["keywords"]:
            if kw in query_clean:
                return item
    return None