from google import genai

client = genai.Client(
    vertexai=True,
    project="quixotic-skill-424213-h6",
    location="global",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one short sentence."
)

print(response.text)