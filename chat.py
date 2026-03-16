import fn

messages = []

fn.add_assistant_message(messages, "What is 3 * 3?")


response = fn.chat(messages)
print(response)
