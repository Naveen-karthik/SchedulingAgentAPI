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
        return 9999.0  # unknown → treat as very far

    lat1, lon1 = CITY_COORDS[loc1]
    lat2, lon2 = CITY_COORDS[loc2]

    R = 3958.8  # Earth radius in miles
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
    """Load assignments from CSV and parse types."""
    path = filepath or os.path.join(DATASET_DIR, "assignments.csv")
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
    """Load producers from CSV and parse types."""
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
    """
    Returns (is_eligible, list_of_reasons).
    All hard constraints must pass for eligibility.
    """
    reasons = []

    # 1. Skill match
    if assignment["required_skill"] not in producer["skills"]:
        reasons.append(f"Missing required skill '{assignment['required_skill']}'")

    # 2. Availability — producer must be available on or before due date
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
    """
    Higher score = better match.

    Scoring factors:
      - Distance:  100 pts if same city, scales down with distance
      - Workload:  30 pts for workload=0, 0 pts for workload=5+
      - Availability window overlap: up to 20 pts
    """
    distance = haversine_miles(producer["location"], assignment["location"])

    # Distance score: 100 if same city (< 5 mi), drops as distance grows
    distance_score = max(0, 100 - distance)

    # Workload score: fewer assignments = better (max 5 workload considered)
    workload_score = max(0, (5 - producer["current_workload"]) / 5) * 30

    # Availability score: how many days available around the due date
    available_days = (producer["available_to"] - producer["available_from"]).days + 1
    availability_score = min(available_days / 3, 1) * 20

    total = distance_score + workload_score + availability_score
    return round(total, 2)


# ─────────────────────────────────────────────────────────
# Conflict Detection
# ─────────────────────────────────────────────────────────

def check_conflict(
    producer_id: str,
    assignment: dict,
    confirmed_schedule: list[dict]
) -> tuple[bool, Optional[str]]:
    """
    Check if a producer is double-booked on the same due date.
    Returns (has_conflict, conflicting_assignment_id or None).
    """
    for entry in confirmed_schedule:
        if (
            entry["producer_id"] == producer_id
            and entry["due_date"] == assignment["due_date"]
        ):
            return True, entry["assignment_id"]
    return False, None


# ─────────────────────────────────────────────────────────
# Core Scheduling Agent
# ─────────────────────────────────────────────────────────

def run_scheduling_agent(
    assignments_path: Optional[str] = None,
    producers_path: Optional[str] = None,
) -> dict:
    """
    Main agent entry point.
    Returns a structured result with scheduled, escalated, and conflict entries.
    """
    assignments = load_assignments(assignments_path)
    producers = load_producers(producers_path)

    # Step 1: Sort assignments — High priority first, then by due date (earliest first)
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    sorted_assignments = sorted(
        assignments,
        key=lambda a: (priority_order.get(a["priority"], 99), a["due_date"])
    )

    confirmed_schedule: list[dict] = []  # tracks producer bookings for conflict detection
    results = []

    for assignment in sorted_assignments:
        aid = assignment["assignment_id"]
        eligible_producers = []

        # Step 2: Validate eligibility for each producer
        for producer in producers:
            is_eligible, fail_reasons = validate_producer_eligibility(producer, assignment)
            if is_eligible:
                score = score_producer(producer, assignment)
                eligible_producers.append({
                    "producer": producer,
                    "score": score,
                })

        # Step 3: No eligible producer → Escalate
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

        # Step 4: Sort eligible producers by score (highest first)
        eligible_producers.sort(key=lambda x: x["score"], reverse=True)

        # Step 5: Pick best producer, respecting conflict check
        assigned = False
        for candidate in eligible_producers:
            p = candidate["producer"]
            has_conflict, conflicting_id = check_conflict(
                p["producer_id"], assignment, confirmed_schedule
            )
            if has_conflict:
                continue  # skip — already booked that day

            # Book this producer
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

        # All eligible producers had conflicts → Escalate
        if not assigned:
            results.append({
                "assignment_id": aid,
                "assigned_producer_id": None,
                "assigned_producer_name": None,
                "decision": "Escalate",
                "reason": "All eligible producers are already booked for this time window (conflict detected).",
                "score": None,
            })

    # ── Summary ──────────────────────────────────────────
    scheduled = [r for r in results if r["decision"] == "Scheduled"]
    escalated = [r for r in results if r["decision"] == "Escalate"]

    return {
        "total_assignments": len(assignments),
        "scheduled_count": len(scheduled),
        "escalated_count": len(escalated),
        "schedule": results,
    }