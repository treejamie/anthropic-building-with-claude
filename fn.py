from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv()
client = Anthropic()


def add_user_message(messages, content):
    user_message = {"role": "user", "content": content}
    messages.append(user_message)


def add_assistant_message(messages, content):
    assistant_message = {"role": "assistant", "content": content}
    messages.append(assistant_message)


def chat(messages):
    message = client.messages.create(
        model="claude-haiku-4-5", max_tokens=1000, messages=messages
    )
    return message.content[0].text
