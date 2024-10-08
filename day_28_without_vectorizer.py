# Problem: Build a simple chatbot using traditional NLP techniques
# Without vectorization

import re
import random


# Step 1: Create a dataset of intents and responses
intents = {
    'greeting': {
        'patterns': ['hello', 'hi', 'hey', 'good morning', 'good evening', 'good afternoon'],
        'responses': ['Hello!', 'Hi there!', 'Greetings!', 'Good day!', 'Hey! How can I help you today?']
    },
    'farewell': {
        'patterns': ['bye', 'goodbye', 'see you later', 'farewell', 'take care'],
        'responses': ['Goodbye!', 'Take care!', 'See you later!', 'Farewell!', 'Have a great day!']
    },
    'thanks': {
        'patterns': ['thank you', 'thanks', 'thank you so much', 'much appreciated', 'thanks a lot'],
        'responses': ['You’re welcome!', 'Glad I could help!', 'Anytime!', 'My pleasure!', 'No problem!']
    },
    'bot_name': {
        'patterns': ['what is your name', 'who are you', 'tell me your name'],
        'responses': ['I’m your friendly chatbot!', 'You can call me Chatbot!', 'I am a chatbot created to help you.']
    },
    'bot_purpose': {
        'patterns': ['what can you do', 'how can you help me', 'what do you do'],
        'responses': ['I can assist you with basic queries, answer questions, and chat with you!', 'I’m here to help you with anything you need.', 'I can chat with you and answer simple questions!']
    },
    'feeling': {
        'patterns': ['how are you', 'how are you doing', 'are you okay'],
        'responses': ['I’m just a bot, but I’m doing great!', 'I’m feeling helpful today!', 'I’m here to help you, so I’m doing well!']
    },
    'age': {
        'patterns': ['how old are you', 'what is your age', 'when were you created'],
        'responses': ['I don’t have an age like humans, but I’m always learning!', 'Age is just a number, and I don’t have one!', 'I was created recently to help you out!']
    },
    'weather': {
        'patterns': ['what is the weather', 'how is the weather today', 'tell me the weather'],
        'responses': ['I can’t check the weather right now, but you can check your weather app!', 'I don’t have access to weather data, but I hope it’s nice outside!', 'Check your weather app for accurate information!']
    },
    'joke': {
        'patterns': ['tell me a joke', 'make me laugh', 'tell a joke'],
        'responses': ['Why did the computer go to the doctor? Because it had a virus!', 'Why don’t robots have brothers? Because they all have trans-sisters!', 'I’d tell you a joke about UDP, but you might not get it!']
    },
    'help': {
        'patterns': ['help me', 'i need help', 'can you help me'],
        'responses': ['Sure, I’m here to help! What do you need?', 'Of course! Let me know how I can assist you.', 'I’m happy to help. Please tell me what you need assistance with!']
    },
    'unknown': {
        'patterns': ['who is the president', 'where is the moon', 'how to cook pasta', 'tell me a story'],
        'responses': ['Sorry, I’m not sure how to answer that.', 'I don’t have the answer to that right now.', 'Hmm, I don’t know, but I can find out!']
    }
}


# Step 2: Preprocess user input
def preprocess(text):
    text = text.lower() # Conver to lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = text.split() # Tokenization by splitting

    return tokens


# Step 3: Implement pattern matching with bag of words (BOW)
def match_intent(processed_input, intents):
    for intent, intent_data in intents.items():
        patterns = intent_data['patterns']
        for pattern in patterns:
            processed_pattern = preprocess(pattern)
            if all(word in processed_input for word in processed_pattern):
                return intent

    return None


# Step 4: Generate the response
def get_response(intent, intents):
    return random.choice(intents[intent]['responses'])


def chatbot():
    print("Hey! I am a 28_chatbot, How can i help you? Type bye to exit")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Chatbot: Good Bye!")
            break

        processed_input = preprocess(user_input)
        matched_intent = match_intent(processed_input=processed_input, intents=intents)
        if matched_intent:
            response = get_response(intent=matched_intent, intents=intents)
            print(f"Chatbot: {response}")
        else:
            print(f"Chatbot: Sorry i could not understand that.")


# Run the chatbot
chatbot()