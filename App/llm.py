from google import genai
from openai import OpenAI
from config import GEMINI_API_KEY, OPENAI_API_KEY

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def generate_answer(prompt, context):
    system_prompt = f"""
    You are a parliamentary assistant.

    STRICT OUTPUT JSON:
    {{
      "summary": "...",
      "key_points": ["...", "..."],
      "citations": ["...", "..."]
    }}

    CONTEXT:
    {context}

    QUESTION:
    {prompt}
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=system_prompt
    )

    return response.text


def format_response(raw_json):
    prompt = f"""
    Convert this JSON into clean Markdown for Streamlit.

    JSON:
    {raw_json}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a UI formatter."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content