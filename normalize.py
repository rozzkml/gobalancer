"""
Output normalizer — make every model agent-clean.

Some models (Gemma via Google OpenAI-compat, some zen models) leak their
chain-of-thought into `content` wrapped in <thought>/<think>/<thinking> tags.
Agent clients (Hermes, OpenClaw) read `content` verbatim, so that thinking
pollutes the visible answer. This module strips those blocks:

  - strip_thoughts(text)      -> non-stream: clean a whole string
  - ThoughtStripper()         -> stream: stateful filter across content deltas

Only `content` text is touched. tool_calls, finish_reason, role, usage and any
other fields pass through untouched — critical for agentic tool-calling.
"""
import re

_TAGS = ("thought", "think", "thinking")
_BLOCK_RE = re.compile(
    r"<(thought|think|thinking)\s*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_OPEN_RE = re.compile(r"<(thought|think|thinking)\s*>", re.IGNORECASE)
_OPEN_TAGS = tuple(f"<{t}>" for t in _TAGS)
_MAX_OPEN_LEN = max(len(t) for t in _OPEN_TAGS)  # longest "<thinking>"


def strip_thoughts(text):
    """Remove complete <thought>..</thought> blocks from a full string."""
    if not text or "<" not in text:
        return text
    cleaned = _BLOCK_RE.sub("", text)
    # Drop a dangling unclosed opener + everything after it (defensive; models
    # that open a thought but get cut off by max_tokens).
    m = _OPEN_RE.search(cleaned)
    if m:
        cleaned = cleaned[:m.start()]
    return cleaned.lstrip("\n")


class ThoughtStripper:
    """Streaming state machine that strips thought blocks from content deltas.

    Tags can span multiple chunks, so we buffer just enough to detect a split
    opening/closing tag. Everything outside thought blocks is emitted ASAP to
    preserve real incremental streaming.
    """

    def __init__(self):
        self.buf = ""
        self.in_thought = False
        self._close_re = None

    def _partial_open_hold(self):
        """If buf tail could be the start of an opening tag, return how many
        trailing chars to hold back; else 0."""
        lt = self.buf.rfind("<")
        if lt == -1:
            return 0
        tail = self.buf[lt:].lower()
        if len(tail) > _MAX_OPEN_LEN:
            return 0  # a full "<...>" already; not a tag we care about
        # hold only if tail is a strict prefix of some opening tag
        for t in _OPEN_TAGS:
            if t.startswith(tail):
                return len(tail)
        return 0

    def feed(self, text):
        if not text:
            return ""
        self.buf += text
        out = []
        while self.buf:
            if not self.in_thought:
                m = _OPEN_RE.search(self.buf)
                if m:
                    out.append(self.buf[:m.start()])
                    tag = m.group(1)
                    self._close_re = re.compile(rf"</{tag}\s*>", re.IGNORECASE)
                    self.buf = self.buf[m.end():]
                    self.in_thought = True
                    continue
                hold = self._partial_open_hold()
                if hold:
                    out.append(self.buf[:len(self.buf) - hold])
                    self.buf = self.buf[len(self.buf) - hold:]
                else:
                    out.append(self.buf)
                    self.buf = ""
                break
            else:
                assert self._close_re is not None
                m = self._close_re.search(self.buf)
                if m:
                    self.buf = self.buf[m.end():]
                    self.in_thought = False
                    self._close_re = None
                    continue
                # still inside thought: suppress, keep a small tail so a closing
                # tag split across chunks is still detectable.
                keep = 16
                if len(self.buf) > keep:
                    self.buf = self.buf[-keep:]
                break
        return "".join(out)

    def flush(self):
        """Emit any trailing buffered content at stream end (unless still mid-thought)."""
        if self.in_thought:
            self.buf = ""
            return ""
        r = self.buf
        self.buf = ""
        return r
