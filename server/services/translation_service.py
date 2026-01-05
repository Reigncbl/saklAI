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
        
        translation_prompt = f"""Translate the following Tagalog/Taglish text to clear, natural English. Preserve the banking/financial context and intent exactly.

TRANSLATION GUIDELINES:
1. Maintain the original meaning and context precisely
2. Use natural, conversational English 
3. Preserve banking terminology and intent
4. Keep the tone and formality level consistent
5. Ensure the translation sounds natural to English speakers

Text to translate: "{text}"

Provide ONLY the English translation without any additional explanation, commentary, or formatting."""

        response = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[{"role": "user", "content": translation_prompt}],
            temperature=0.0,  # Zero temperature for consistent translation
            max_tokens=100    # Reduced for faster translation
        )
        
        translated = response.choices[0].message.content.strip()
        if translated and translated.lower() != text.lower():
            print(f"Translated: '{text}' -> '{translated}'")
            return translated
        
        return text  # Fall back to original if translation fails
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text
