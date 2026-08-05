import re
from datetime import datetime
from pathlib import Path
from common.data_models import LogEntry

LOG_PATTERN = re.compile(
    r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})?\s*'
    r'(?P<level>INFO|WARN|ERROR|CRITICAL|DEBUG)?\s*'
    r'(?P<module>\w+)?\s*'
    r'(?P<message>.*)'
)

def parse_mes_logs(file_paths):
    results = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = LOG_PATTERN.search(line.strip())
                if match:
                    entry = match.groupdict()
                    entry['raw'] = line.strip()
                    entry['file'] = Path(file_path).name
                    if not entry['timestamp']:
                        entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    results.append(LogEntry(**entry))
    return results