"""
Translation service for handling multilingual support.
"""

from groq import Groq


def translate_to_english(text: str, api_key: str) -> str:
    """
    Translate Tagalog/Taglish text to English using Groq directly.
    Returns original text if translation fails or is not needed.
    """
    try:
        # Check if text contains Tagalog/Taglish indicators
        tagalog_indicators = ["paano", "gumawa", "bukas", "mag", "pag", "ng", "sa", "ko", "mo", "na", "ang", "may", "hindi", "ito", "ako", "ikaw", "siya", "tayo", "kayo", "sila"]
        if not any(indicator in text.lower() for indicator in tagalog_indicators):
            return text  # Likely already in English
        
        # Use Groq for direct translation
        groq_client = Groq(api_key=api_key)
        
        translation_prompt = f"""Translate the following Tagalog/Taglish text to clear English. Keep the banking/financial context intact.

Text: "{text}"

Translation:"""

        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": translation_prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        translated = response.choices[0].message.content.strip()
        if translated and translated.lower() != text.lower():
            print(f"Translated: '{text}' -> '{translated}'")
            return translated
        
        return text  # Fall back to original if translation fails
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text
