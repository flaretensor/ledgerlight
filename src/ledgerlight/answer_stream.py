from ledgerlight.citations import ABSTENTION, MARKER, is_abstention


class AnswerPreview:
    def __init__(self, issued):
        self.issued = issued
        self.pending = ""
        self.mapping = {}
        self.started = False
        self.thinking = False
        self.invalid = False

    def feed(self, text):
        if self.invalid:
            return ""
        self.pending += text
        output = ""
        while self.pending:
            if self.thinking:
                position = self.pending.casefold().find("</think>")
                if position < 0:
                    self.pending = self.pending[-7:]
                    break
                self.pending = self.pending[position + 8 :]
                self.thinking = False
                continue
            if self.pending.startswith("<"):
                if "<think>".startswith(self.pending.casefold()):
                    if len(self.pending) < 7:
                        break
                if self.pending[:7].casefold() == "<think>":
                    self.pending = self.pending[7:]
                    self.thinking = True
                    continue
                self.invalid = True
                break
            if not self.started:
                stripped = self.pending.strip().casefold()
                if ABSTENTION.startswith(stripped) or is_abstention(self.pending):
                    break
                self.started = True
            if self.pending.startswith("["):
                prefix = "[[SRC_"
                if prefix.startswith(self.pending):
                    break
                if self.pending.startswith(prefix):
                    ending = self.pending.find("]]", len(prefix))
                    if ending < 0:
                        break
                    marker = self.pending[: ending + 2]
                    if MARKER.fullmatch(marker) is None or marker not in self.issued:
                        self.invalid = True
                        break
                    if marker not in self.mapping:
                        self.mapping[marker] = len(self.mapping) + 1
                    output += f"[{self.mapping[marker]}]"
                    self.pending = self.pending[ending + 2 :]
                    continue
            output += self.pending[0]
            self.pending = self.pending[1:]
        return output
