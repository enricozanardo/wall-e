import json

# Define domain names
domain_names = {
    0: "GDPR",
    1: "Software Defined Networks",
    2: "Number Theory",
    3: "Logic & Mathematics",
    4: "Ecology",
    5: "Geometry"
}

# Read the original dataset
with open("data/qa_en_dataset.json") as f:
    data = json.load(f)

print("=== DATASET OVERVIEW ===")
total = 0
for i, context in enumerate(data["train_data"]):
    count = len(context["questions"])
    domain = domain_names.get(i, f"Domain {i+1}")
    print(f"{domain}: {count} QA pairs")
    total += count

print(f"Total: {total} QA pairs")

# Read the fixed dataset if it exists
try:
    with open("data/qa_en_dataset_fixed.json") as f:
        fixed_data = json.load(f)

    print("\n=== FIXED DATASET ===")
    total = 0
    for i, context in enumerate(fixed_data["train_data"]):
        count = len(context["questions"])
        domain = domain_names.get(i, f"Domain {i+1}")
        print(f"{domain}: {count} QA pairs")
        total += count

    print(f"Total: {total} QA pairs")
except FileNotFoundError:
    print("\nFixed dataset not found.") 