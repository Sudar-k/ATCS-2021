jobs = ["programmer", "truck driver", "lawyer", "actor"]

i = jobs.index("lawyer")
print("lawyer" in jobs)
jobs.append("teacher")

jobs.insert(0, "chef")

for job in jobs:
    print(job)

