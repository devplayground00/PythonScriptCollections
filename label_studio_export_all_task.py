from label_studio_sdk.client import LabelStudio
import json
from datetime import datetime

LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY  = ''
PROJECTID = 2

# Custom encoder to handle datetime objects
def default_encoder(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

# Connect to Label Studio
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

# Fetch all tasks and convert to dicts
all_tasks = list(ls.tasks.list(project=PROJECTID, fields='all'))
all_tasks_dicts = [task.dict() for task in all_tasks]

# Save to JSON file
with open("all_tasks_export.json", "w", encoding="utf-8") as f:
    json.dump(all_tasks_dicts, f, indent=2, ensure_ascii=False, default=default_encoder)

print(f"Exported {len(all_tasks_dicts)} total tasks (including unannotated) to all_tasks_export.json")


