"""Semantic loop detection for agent conversations.

Detects when an agent is stuck in a repetition pattern — e.g., repeatedly
re-reading the same file section, cycling through the same analysis phrases,
or saying "I've been going in circles" without actually changing course.

Two detection strategies:
1. **Thought repetition** — The last N assistant messages share similar
   phrasing (e.g., "going in circles", "different approach", "re-reading").
2. **Tool repetition** — The same tool+file combination is used 3+ times
   without an intervening write/edit.

When a loop is detected, injects a checkpoint message into the conversation
that forces the agent to summarize what it has tried and either make progress
or stop (kanban_block/kanban_complete).

This is used primarily for kanban workers but can apply to any long-running
conversation where the agent has access to file tools.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thought repetition patterns
# ---------------------------------------------------------------------------

# Phrases that indicate the agent is aware it's looping. If the agent says
# one of these 2+ times in the last N turns, it's a strong signal.
_LOOP_INDICATOR_PHRASES = [
    "going in circles",
    "in circles",
    "same approach",
    "different approach",
    "take a completely different approach",
    "re-read",
    "re-reading",
    "revisit",
    "try again",
    "another look",
    "look one more time",
    "one more time",
    "once more",
    "still stuck",
    "not making progress",
    "no progress",
    "same file",
    "same section",
    "same lines",
]

# ---------------------------------------------------------------------------
# Tool repetition constants
# ---------------------------------------------------------------------------

# Read-only tools that, when repeated on the same file, indicate a loop.
_READ_ONLY_TOOLS = {
    "read_file",
    "search_files",
    "browser_snapshot",
    "browser_console",
    "web_extract",
    "web_search",
    "skill_view",
    "terminal",  # only when used for reading (grep, head, tail, sed)
}

# Minimum repetitions before flagging as a loop.
TOOL_LOOP_THRESHOLD = 5

# Number of recent assistant messages to look at for thought repetition.
THOUGHT_WINDOW = 5

# Number of recent tool calls to look at for tool repetition.
TOOL_WINDOW = 6


def detect_semantic_loop(
    messages: List[Dict[str, Any]],
    max_turns: int,
    api_call_count: int,
    is_kanban: bool = False,
) -> Optional[str]:
    """Check if the agent is stuck in a semantic loop.

    Analyzes recent assistant messages (thoughts) and tool calls for
    repetition patterns. Returns a checkpoint message string if a loop
    is detected, or None if the conversation is proceeding normally.

    Args:
        messages: The full conversation message list.
        max_turns: The agent's max_iterations / max_turns budget.
        api_call_count: Current API call count (0-indexed within the turn).
        is_kanban: Whether this is a kanban worker task.

    Returns:
        A checkpoint message to inject into the conversation, or None.
    """
    # Only check at the halfway point or later — don't interrupt early work.
    if max_turns and api_call_count < max_turns // 3:
        return None

    # Get recent assistant messages (last THOUGHT_WINDOW).
    assistant_messages = _get_recent_assistant_messages(messages, THOUGHT_WINDOW)

    # Check for thought repetition.
    thought_loop = _detect_thought_loop(assistant_messages)
    if thought_loop:
        logger.info(
            "Loop detection: thought repetition detected (%d messages analyzed)",
            len(assistant_messages),
        )
        return thought_loop

    # Check for tool repetition.
    tool_loop = _detect_tool_loop(messages, TOOL_WINDOW)
    if tool_loop:
        logger.info(
            "Loop detection: tool repetition detected (%s)", tool_loop,
        )
        return tool_loop

    return None


def _get_recent_assistant_messages(
    messages: List[Dict[str, Any]],
    count: int,
) -> List[Dict[str, Any]]:
    """Get the last N assistant messages that have text content."""
    result = []
    for msg in reversed(messages):
        if len(result) >= count:
            break
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content and not content.startswith("[System:"):
                result.append(msg)
    return list(reversed(result))


def _detect_thought_loop(
    assistant_messages: List[Dict[str, Any]],
) -> Optional[str]:
    """Detect if the agent is repeating the same analysis phrases.

    Looks for:
    1. Loop indicator phrases appearing 2+ times
    2. Similar phrasing patterns across messages
    3. The agent saying it's "going in circles" multiple times
    """
    if len(assistant_messages) < 2:
        return None

    # Collect all text from assistant messages.
    all_text = " ".join(
        msg.get("content", "") or "" for msg in assistant_messages
    )
    all_text_lower = all_text.lower()

    # Count loop indicator phrases.
    phrase_counts = Counter()
    for phrase in _LOOP_INDICATOR_PHRASES:
        count = all_text_lower.count(phrase.lower())
        if count > 0:
            phrase_counts[phrase] = count

    # If any loop indicator appears 2+ times, it's a strong signal.
    if phrase_counts:
        top_phrases = phrase_counts.most_common(3)
        total_loop_mentions = sum(c for _, c in top_phrases)

        if total_loop_mentions >= 2:
            top_phrase_names = ", ".join(
                f'"{p}"' for p, _ in top_phrases[:2]
            )
            return (
                "[System: Loop detected. You have mentioned being stuck "
                f"({top_phrase_names}) {total_loop_mentions} times in recent "
                "turns without making progress. You must now:\n"
                "1. Summarize EXACTLY what you have tried so far (list each\n"
                "   approach and its result).\n"
                "2. If you have tried 3+ similar approaches with no result,\n"
                "   call kanban_block(reason=\\\"stuck: repeated same analysis\")\n"
                "   instead of continuing.\n"
                "3. If you have a clear next step that is DIFFERENT from what\n"
                "   you've already tried, take it — but only ONE step.\n"
                "Do NOT re-read the same files or repeat the same analysis.]"
            )

    return None


def _detect_tool_loop(
    messages: List[Dict[str, Any]],
    window: int = TOOL_WINDOW,
) -> Optional[str]:
    """Detect if the agent is repeatedly calling the same tool on the same file.

    Looks for patterns like:
    - read_file on server.py lines 400-430 repeated 3+ times
    - grep for the same pattern repeated 3+ times
    - terminal with the same sed/grep command repeated 3+ times

    Returns a checkpoint message if a loop is detected.
    """
    # Extract tool calls from recent assistant messages.
    tool_calls = _extract_recent_tool_calls(messages, window)

    if len(tool_calls) < TOOL_LOOP_THRESHOLD:
        return None

    # Group tool calls by (tool_name, file_target).
    file_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tc in tool_calls:
        tool_name = tc.get("function", {}).get("name", "")
        file_target = _extract_file_target(tc)
        if file_target:
            key = f"{tool_name}:{file_target}"
            file_groups[key].append(tc)

    # Check for repetition.
    for key, calls in file_groups.items():
        if len(calls) >= TOOL_LOOP_THRESHOLD:
            tool_name, file_target = key.split(":", 1)
            return (
                f"[System: Loop detected. You have called {tool_name} on "
                f"'{file_target}' {len(calls)} times in a row without making "
                "a change. You must either:\n"
                "1. Make a concrete edit/change based on what you've learned,\n"
                "   OR\n"
                "2. Call kanban_block(reason=\\\"stuck: re-reading same file\")\n"
                "   if you cannot make progress.\n"
                "Do NOT call the same tool on the same file again.]"
            )

    return None


def _extract_recent_tool_calls(
    messages: List[Dict[str, Any]],
    window: int = TOOL_WINDOW,
) -> List[Dict[str, Any]]:
    """Extract tool calls from the last N assistant messages."""
    tool_calls = []
    seen = 0
    for msg in reversed(messages):
        if seen >= window:
            break
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []) or []:
                if isinstance(tc, dict):
                    tool_calls.append(tc)
            seen += 1
    return list(reversed(tool_calls))


def _extract_file_target(tc: Dict[str, Any]) -> Optional[str]:
    """Extract the file target from a tool call.

    Returns the file path if the tool call targets a specific file,
    or None if it doesn't have a clear file target.
    """
    name = tc.get("function", {}).get("name", "")
    args_str = tc.get("function", {}).get("arguments", "{}")

    if name == "read_file":
        # read_file(path='...')
        match = re.search(r"path\s*[:=]\s*['\"]([^'\"]+)['\"]", args_str)
        if match:
            return match.group(1)
        # Also try JSON args
        try:
            args = json.loads(args_str)
            if "path" in args:
                return args["path"]
        except (json.JSONDecodeError, ValueError):
            pass

    elif name == "patch":
        match = re.search(r"path\s*[:=]\s*['\"]([^'\"]+)['\"]", args_str)
        if match:
            return match.group(1)
        try:
            args = json.loads(args_str)
            if "path" in args:
                return args["path"]
        except (json.JSONDecodeError, ValueError):
            pass

    elif name == "write_file":
        match = re.search(r"path\s*[:=]\s*['\"]([^'\"]+)['\"]", args_str)
        if match:
            return match.group(1)
        try:
            args = json.loads(args_str)
            if "path" in args:
                return args["path"]
        except (json.JSONDecodeError, ValueError):
            pass

    elif name == "terminal":
        # Check if the command is a read-only operation on a file.
        # grep, head, tail, sed -n, cat, wc -l, etc.
        match = re.search(r"command\s*[:=]\s*['\"]([^'\"]+)['\"]", args_str)
        if match:
            cmd = match.group(1)
            # Look for file paths in the command.
            # grep/rg/sed/head/tail/cat with a file path.
            file_match = re.search(r"\b([a-zA-Z_./][a-zA-Z0-9_./-]*\.py|[a-zA-Z_./][a-zA-Z0-9_./-]*\.html|[a-zA-Z_./][a-zA-Z0-9_./-]*\.js|[a-zA-Z_./][a-zA-Z0-9_./-]*\.css|[a-zA-Z_./][a-zA-Z0-9_./-]*\.md|[a-zA-Z_./][a-zA-Z0-9_./-]*\.txt)", cmd)
            if file_match and any(
                op in cmd for op in ["grep", "head", "tail", "sed", "cat", "wc", "rg "]
            ):
                return file_match.group(1)

    elif name == "search_files":
        match = re.search(r"path\s*[:=]\s*['\"]([^'\"]+)['\"]", args_str)
        if match:
            return match.group(1)
        try:
            args = json.loads(args_str)
            if "path" in args:
                return args["path"]
        except (json.JSONDecodeError, ValueError):
            pass

    return None
