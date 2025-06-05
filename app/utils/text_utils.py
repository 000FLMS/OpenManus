import re


def sanitize_string_for_json(input_str: str) -> str:
    """
    Removes control characters from a string, except for common whitespace
    characters like tab, newline, and carriage return, to make it safe for JSON serialization.
    This includes C0 and C1 control characters, and DEL.
    """
    if not isinstance(input_str, str):
        return input_str
    # Allow tab (\t U+0009), newline (\n U+000A), carriage return (\r U+000D)
    # Remove C0 controls U+0000-U+001F (excluding tab, newline, CR)
    # Remove DEL U+007F
    # Remove C1 controls U+0080-U+009F
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]", "", input_str)
