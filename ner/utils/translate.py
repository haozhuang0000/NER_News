from deep_translator import GoogleTranslator


def translate_word(word, ln):
    # Handle special cases for language codes
    if ln == "zh":
        ln = "zh-CN"  # Simplified Chinese
    if ln == "pi":
        ln = "tl"  # Correct for Filipino/Tagalog
    if ln == "jp":
        ln = "ja"
    try:
        # Perform translation
        tn_text = GoogleTranslator(source="auto", target=ln).translate(word)
        return tn_text
    except Exception as e:
        # Return an error message if translation fails
        return f"Error: {e}"
