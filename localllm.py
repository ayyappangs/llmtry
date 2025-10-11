from llama_cpp import Llama
# Initialize the Llama model with the Phi-4 Mini Reasoning weights and chatml format from a local folder path
llm = Llama(model_path="Phi-4-mini-reasoning-Q4_K_M.gguf", n_ctx=4096, chat_format="chatml")
messages = []
print("Welcome to Phi-4 chat! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break
    messages.append({"role": "user", "content": user_input})
    try:
        resp = llm.create_chat_completion(messages=messages)
        reply = resp["choices"][0]["message"]["content"]
        print(f"AI: {reply}")
        messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        print(f"Error: {e}")
