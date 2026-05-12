"""
Scheduling Agent Service
Handles: data loading, eligibility validation, producer ranking, assignment, conflict detection,
         negative scenario detection and reporting
"""

import csv
import os
from datetime import datetime, date
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

# ─────────────────────────────────────────────────────────
# City coordinates for distance calculation (Florida cities)
# ─────────────────────────────────────────────────────────
CITY_COORDS = {
    "Jacksonville, FL": (30.3322, -81.6557),
    "Orlando, FL":      (28.5383, -81.3792),
    "Miami, FL":        (25.7617, -80.1918),
    "Fort Lauderdale, FL": (26.1224, -80.1373),
    "Tampa, FL":        (27.9506, -82.4572),
    "Gainesville, FL":  (29.6516, -82.3248),
}


def haversine_miles(loc1: str, loc2: str) -> float:
    """Return distance in miles between two city name strings."""
    if loc1 not in CITY_COORDS or loc2 not in CITY_COORDS:
        return 9999.0
    lat1, lon1 = CITY_COORDS[loc1]
    lat2, lon2 = CITY_COORDS[loc2]
    R = 3958.8
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# ─────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")


def load_assignments(filepath: Optional[str] = None) -> list[dict]:
    # ── Switch between datasets here ──────────────────────────────────────────
    path = filepath or os.path.join(DATASET_DIR, "assignments.csv")                     # Normal data (active)
    # path = filepath or os.path.join(DATASET_DIR, "negative_assignments.csv")          # Negative test data
    # ──────────────────────────────────────────────────────────────────────────
    assignments = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assignments.append({
                "assignment_id": row["assignment_id"].strip(),
                "type": row["type"].strip(),
                "location": row["location"].strip(),
                "required_skill": row["required_skill"].strip(),
                "priority": row["priority"].strip(),
                "due_date": datetime.strptime(row["due_date"].strip(), "%Y-%m-%d").date(),
                "estimated_duration_hours": int(row["estimated_duration_hours"].strip()),
            })
    return assignments


def load_producers(filepath: Optional[str] = None) -> list[dict]:
    path = filepath or os.path.join(DATASET_DIR, "producers.csv")
    producers = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills_raw = row["skills"].strip()
            producers.append({
                "producer_id": row["producer_id"].strip(),
                "name": row["name"].strip(),
                "location": row["location"].strip(),
                "skills": [s.strip() for s in skills_raw.split(",")],
                "available_from": datetime.strptime(row["available_from"].strip(), "%Y-%m-%d").date(),
                "available_to": datetime.strptime(row["available_to"].strip(), "%Y-%m-%d").date(),
                "current_workload": int(row["current_workload"].strip()),
                "travel_limit_miles": int(row["travel_limit_miles"].strip()),
            })
    return producers


# ─────────────────────────────────────────────────────────
# Eligibility Validator
# ─────────────────────────────────────────────────────────

def validate_producer_eligibility(producer: dict, assignment: dict) -> tuple[bool, list[str]]:
    reasons = []

    # 1. Skill match
    if assignment["required_skill"] not in producer["skills"]:
        reasons.append(f"Missing required skill '{assignment['required_skill']}'")

    # 2. Availability
    if not (producer["available_from"] <= assignment["due_date"] <= producer["available_to"]):
        reasons.append(
            f"Not available on due date {assignment['due_date']} "
            f"(available {producer['available_from']} to {producer['available_to']})"
        )

    # 3. Travel limit
    distance = haversine_miles(producer["location"], assignment["location"])
    if distance > producer["travel_limit_miles"]:
        reasons.append(
            f"Distance {distance:.0f} miles exceeds travel limit {producer['travel_limit_miles']} miles"
        )

    is_eligible = len(reasons) == 0
    return is_eligible, reasons


# ─────────────────────────────────────────────────────────
# Scoring / Ranking
# ─────────────────────────────────────────────────────────

def score_producer(producer: dict, assignment: dict) -> float:
    distance = haversine_miles(producer["location"], assignment["location"])
    distance_score = max(0, 100 - distance)
    workload_score = max(0, (5 - producer["current_workload"]) / 5) * 30
    available_days = (producer["available_to"] - producer["available_from"]).days + 1
    availability_score = min(available_days / 3, 1) * 20
    return round(distance_score + workload_score + availability_score, 2)


# ─────────────────────────────────────────────────────────
# Conflict Detection
# ─────────────────────────────────────────────────────────

def check_conflict(
    producer_id: str,
    assignment: dict,
    confirmed_schedule: list[dict]
) -> tuple[bool, Optional[str]]:
    for entry in confirmed_schedule:
        if (
            entry["producer_id"] == producer_id
            and entry["due_date"] == assignment["due_date"]
        ):
            return True, entry["assignment_id"]
    return False, None


# ─────────────────────────────────────────────────────────
# Negative Scenario Detector
# ─────────────────────────────────────────────────────────

def detect_negative_scenarios(assignments: list[dict], producers: list[dict]) -> list[dict]:
    """
    Scans all assignments and producers BEFORE scheduling.
    Identifies every negative pattern and explains exactly why it is a problem.

    Negative patterns detected:
      1. NO_SKILL_MATCH       — No producer has the required skill
      2. NO_AVAILABILITY      — No producer is available on the due date
      3. TRAVEL_LIMIT_BREACH  — All skilled producers are too far away
      4. OVERLOADED_PRODUCER  — Only available producer has workload >= 5
      5. PAST_DUE_DATE        — Assignment due date has already passed
      6. UNKNOWN_LOCATION     — Assignment location not in our city map
    """
    today = date.today()
    negative_scenarios = []

    for assignment in assignments:
        aid = assignment["assignment_id"]
        flags = []

        # ── Pattern 1: Past due date ──────────────────────
        if assignment["due_date"] < today:
            flags.append({
                "pattern": "PAST_DUE_DATE",
                "description": (
                    f"Assignment due date {assignment['due_date']} has already passed "
                    f"(today is {today}). This job cannot be scheduled on time."
                ),
                "recommendation": "Re-negotiate the due date or mark as overdue."
            })

        # ── Pattern 2: Unknown location ───────────────────
        if assignment["location"] not in CITY_COORDS:
            flags.append({
                "pattern": "UNKNOWN_LOCATION",
                "description": (
                    f"Location '{assignment['location']}' is not in the system's city map. "
                    f"Distance calculations will be inaccurate."
                ),
                "recommendation": "Add the city coordinates to CITY_COORDS in the service."
            })

        # ── Per-producer checks ───────────────────────────
        skilled_producers = [
            p for p in producers
            if assignment["required_skill"] in p["skills"]
        ]

        available_producers = [
            p for p in skilled_producers
            if p["available_from"] <= assignment["due_date"] <= p["available_to"]
        ]

        reachable_producers = [
            p for p in skilled_producers
            if haversine_miles(p["location"], assignment["location"]) <= p["travel_limit_miles"]
        ]

        # ── Pattern 3: No skill match ─────────────────────
        if not skilled_producers:
            flags.append({
                "pattern": "NO_SKILL_MATCH",
                "description": (
                    f"No producer in the system has the required skill "
                    f"'{assignment['required_skill']}'. "
                    f"Available skills across all producers: "
                    f"{list(set(s for p in producers for s in p['skills']))}."
                ),
                "recommendation": "Onboard a producer with this skill or outsource the job."
            })

        # ── Pattern 4: No availability ────────────────────
        elif not available_producers:
            unavailable_details = [
                f"{p['name']} (available {p['available_from']} to {p['available_to']})"
                for p in skilled_producers
            ]
            flags.append({
                "pattern": "NO_AVAILABILITY",
                "description": (
                    f"Skilled producers exist but none are available on due date "
                    f"{assignment['due_date']}. "
                    f"Skilled producers: {unavailable_details}."
                ),
                "recommendation": "Extend producer availability or push the due date."
            })

        # ── Pattern 5: Travel limit breach ───────────────
        if skilled_producers and not reachable_producers:
            distance_details = [
                f"{p['name']} is {haversine_miles(p['location'], assignment['location']):.0f} miles away (limit: {p['travel_limit_miles']} miles)"
                for p in skilled_producers
            ]
            flags.append({
                "pattern": "TRAVEL_LIMIT_BREACH",
                "description": (
                    f"Skilled producers exist but none can reach '{assignment['location']}' "
                    f"within their travel limits. Details: {distance_details}."
                ),
                "recommendation": "Increase travel allowance or find a local producer."
            })

        # ── Pattern 6: Overloaded producers ──────────────
        eligible_but_overloaded = [
            p for p in producers
            if assignment["required_skill"] in p["skills"]
            and p["current_workload"] >= 5
        ]
        fully_eligible = [
            p for p in producers
            if assignment["required_skill"] in p["skills"]
            and haversine_miles(p["location"], assignment["location"]) <= p["travel_limit_miles"]
            and p["available_from"] <= assignment["due_date"] <= p["available_to"]
        ]
        if eligible_but_overloaded and not fully_eligible:
            flags.append({
                "pattern": "OVERLOADED_PRODUCER",
                "description": (
                    f"The only producers with the required skill are at maximum workload (5+). "
                    f"Overloaded: {[p['name'] for p in eligible_but_overloaded]}."
                ),
                "recommendation": "Reduce existing assignments or bring in additional producers."
            })

        if flags:
            negative_scenarios.append({
                "assignment_id": aid,
                "assignment_type": assignment["type"],
                "location": assignment["location"],
                "priority": assignment["priority"],
                "due_date": str(assignment["due_date"]),
                "negative_patterns_detected": len(flags),
                "patterns": flags,
            })

    return negative_scenarios


# ─────────────────────────────────────────────────────────
# Core Scheduling Agent
# ─────────────────────────────────────────────────────────

def run_scheduling_agent(
    assignments_path: Optional[str] = None,
    producers_path: Optional[str] = None,
) -> dict:
    assignments = load_assignments(assignments_path)
    producers = load_producers(producers_path)

    # ── Negative scenario analysis (runs before scheduling) ──
    negative_scenarios = detect_negative_scenarios(assignments, producers)

    # ── Sort: High priority first, then earliest due date ──
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    sorted_assignments = sorted(
        assignments,
        key=lambda a: (priority_order.get(a["priority"], 99), a["due_date"])
    )

    confirmed_schedule: list[dict] = []
    results = []

    for assignment in sorted_assignments:
        aid = assignment["assignment_id"]
        eligible_producers = []

        for producer in producers:
            is_eligible, fail_reasons = validate_producer_eligibility(producer, assignment)
            if is_eligible:
                score = score_producer(producer, assignment)
                eligible_producers.append({"producer": producer, "score": score})

        if not eligible_producers:
            results.append({
                "assignment_id": aid,
                "assigned_producer_id": None,
                "assigned_producer_name": None,
                "decision": "Escalate",
                "reason": "No eligible producer meets skill, availability, and travel constraints.",
                "score": None,
            })
            continue

        eligible_producers.sort(key=lambda x: x["score"], reverse=True)

        assigned = False
        for candidate in eligible_producers:
            p = candidate["producer"]
            has_conflict, conflicting_id = check_conflict(p["producer_id"], assignment, confirmed_schedule)
            if has_conflict:
                continue

            confirmed_schedule.append({
                "producer_id": p["producer_id"],
                "assignment_id": aid,
                "due_date": assignment["due_date"],
            })

            distance = haversine_miles(p["location"], assignment["location"])
            reason = (
                f"Producer has the required skill '{assignment['required_skill']}', "
                f"is available on {assignment['due_date']}, "
                f"is {distance:.0f} miles away (within {p['travel_limit_miles']}-mile limit), "
                f"and has a workload of {p['current_workload']}."
            )

            results.append({
                "assignment_id": aid,
                "assigned_producer_id": p["producer_id"],
                "assigned_producer_name": p["name"],
                "decision": "Scheduled",
                "reason": reason,
                "score": candidate["score"],
            })
            assigned = True
            break

        if not assigned:
            results.append({
                "assignment_id": aid,
                "assigned_producer_id": None,
                "assigned_producer_name": None,
                "decision": "Escalate",
                "reason": "All eligible producers are already booked for this time window (conflict detected).",
                "score": None,
            })

    scheduled = [r for r in results if r["decision"] == "Scheduled"]
    escalated = [r for r in results if r["decision"] == "Escalate"]

    return {
        "summary": {
            "total_assignments": len(assignments),
            "scheduled_count": len(scheduled),
            "escalated_count": len(escalated),
        },
        "negative_scenarios": {
            "total_patterns_found": len(negative_scenarios),
            "affected_assignments": [n["assignment_id"] for n in negative_scenarios],
            "details": negative_scenarios,
        },
        "assignments": {
            "scheduled": scheduled,
            "escalated": escalated,
        },
    }