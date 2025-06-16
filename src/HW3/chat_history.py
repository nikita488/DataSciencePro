import json

from pathlib import Path
from datetime import datetime, timezone

class ChatHistory:
    def __init__(self, file_name, encoding='utf-8'):
        self.path = Path(file_name)
        self.encoding = encoding

        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding=encoding))
        else:
            self.data = {
                'history': []
            }

        self.history = self.data['history']
    
    def get_entry_by_tag(self, tag):
        return next((entry for entry in self.history if entry['tag'] == tag), None)

    def get_current_time(self):
        return int(datetime.now(timezone.utc).timestamp())

    def add_message(self, tag, sentence, response):
        entry = self.get_entry_by_tag(tag)
        time = self.get_current_time()

        if entry:
            entry['patterns'].append(sentence)
            entry['responses'].append(response)
            entry['dates'].append(time)
        else:
            entry = {
                'tag': tag,
                'patterns': [sentence],
                'responses': [response],
                'dates': [time]
            }

            self.history.append(entry)

    def save(self):
        with open(self.path, 'w', encoding=self.encoding) as history_file:
            json.dump(self.data, history_file, indent=2)
        