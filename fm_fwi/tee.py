class Tee:
    """A file-like object that duplicates written text to multiple output streams.

    Useful when you want output to appear in the terminal/notebook while also
    saving it to a log file.

    Example
    -------
    with open("log.txt", "w", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print("This appears on screen and is saved to log.txt")
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

    def fileno(self):
        for stream in self.streams:
            try:
                return stream.fileno()
            except Exception:
                pass
        raise OSError("No valid fileno available")

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.streams[0], "errors", "replace")

    def writable(self):
        return True

    def __getattr__(self, name):
        return getattr(self.streams[0], name)
