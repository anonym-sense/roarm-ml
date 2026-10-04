"""Chat window for the GUI: type what the arm should do.

A small Tk window that sits beside the PyBullet view. It has no event loop of
its own -- roarm_rl.app calls poll() once per frame, which pumps Tk and hands
back whatever lines were sent since the last call.
"""

import tkinter as tk

WELCOME = (
    "Type what the arm should do and press Enter.\n"
    "Try:  hello   |   nod twice then wave slowly   |   point right\n"
    "      that was great!   |   look around   |   help"
)

BG, PANEL, FG = "#14171c", "#1e232b", "#e6e9ef"
YOU, ARM, NOTE = "#7cc4ff", "#ffb35c", "#8a93a3"


class ChatWindow:
    def __init__(self, title="RoArm chat"):
        self._pending = []
        self._history = []
        self._history_pos = 0
        self.closed = False

        self._root = tk.Tk()
        self._root.title(title)
        self._root.geometry("420x520+40+60")
        self._root.configure(bg=BG)
        self._root.protocol("WM_DELETE_WINDOW", self.close)

        entry_row = tk.Frame(self._root, bg=BG)
        entry_row.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)
        self._entry = tk.Entry(entry_row, bg=PANEL, fg=FG, insertbackground=FG,
                               relief=tk.FLAT, font=("Segoe UI", 11))
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        tk.Button(entry_row, text="Send", command=self._submit, bg="#2f6fb0", fg="white",
                  activebackground="#3d86d1", activeforeground="white", relief=tk.FLAT,
                  padx=14).pack(side=tk.LEFT, padx=(8, 0))
        self._entry.bind("<Return>", self._submit)
        self._entry.bind("<Up>", lambda e: self._recall(-1))
        self._entry.bind("<Down>", lambda e: self._recall(1))

        scroll = tk.Scrollbar(self._root)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._log = tk.Text(self._root, bg=BG, fg=FG, wrap=tk.WORD, relief=tk.FLAT,
                            state=tk.DISABLED, padx=10, pady=8, font=("Segoe UI", 10),
                            yscrollcommand=scroll.set)
        self._log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self._log.yview)
        self._log.tag_configure("you", foreground=YOU, spacing1=8)
        self._log.tag_configure("arm", foreground=ARM, lmargin2=34)
        self._log.tag_configure("note", foreground=NOTE)

        self.note(WELCOME)
        self._entry.focus_set()

    def _append(self, text, tag):
        self._log.configure(state=tk.NORMAL)
        self._log.insert(tk.END, text + "\n", tag)
        self._log.configure(state=tk.DISABLED)
        self._log.see(tk.END)

    def _submit(self, event=None):
        text = self._entry.get().strip()
        if not text:
            return
        self._entry.delete(0, tk.END)
        self._history.append(text)
        self._history_pos = len(self._history)
        self._append(f"you   {text}", "you")
        self._pending.append(text)

    def _recall(self, step):
        if not self._history:
            return "break"
        self._history_pos = max(0, min(len(self._history), self._history_pos + step))
        self._entry.delete(0, tk.END)
        if self._history_pos < len(self._history):
            self._entry.insert(0, self._history[self._history_pos])
        return "break"

    def say(self, text):
        """Show a reply from the arm."""
        if not self.closed:
            self._append(f"arm   {text}", "arm")

    def note(self, text):
        if not self.closed:
            self._append(text, "note")

    def poll(self):
        """Pump the window; return the lines sent since the last call."""
        if self.closed:
            return []
        try:
            self._root.update()
        except tk.TclError:
            self.closed = True
            return []
        sent, self._pending = self._pending, []
        return sent

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self._root.destroy()
            except tk.TclError:
                pass
