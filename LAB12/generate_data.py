import csv
import random
from datetime import datetime, timedelta

random.seed(42)

categories = ["books", "electronics", "clothing", "home", "sports", "toys"]
statuses = ["paid", "pending", "cancelled"]
status_weights = [0.8, 0.12, 0.08]

start_date = datetime(2026, 3, 1)
days = 60

rows = []
for day in range(days):
    current_day = start_date + timedelta(days=day)
    events_per_day = random.randint(8, 20)
    for _ in range(events_per_day):
        event_time = current_day + timedelta(
            hours=random.randint(0, 23), minutes=random.randint(0, 59)
        )
        user_id = f"u{random.randint(1, 50):03d}"
        category = random.choices(
            categories, weights=[3, 4, 3, 2, 2, 1]
        )[0]
        base_price = {
            "books": 35,
            "electronics": 800,
            "clothing": 120,
            "home": 250,
            "sports": 180,
            "toys": 60,
        }[category]
        amount = round(base_price * random.uniform(0.5, 2.0), 2)
        status = random.choices(statuses, weights=status_weights)[0]
        rows.append(
            {
                "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user_id,
                "category": category,
                "amount": amount,
                "status": status,
            }
        )

rows.sort(key=lambda r: r["event_time"])

with open("data/transactions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["event_time", "user_id", "category", "amount", "status"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wygenerowano {len(rows)} wierszy.")
