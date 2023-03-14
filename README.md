# A ChatGPT never forgets
This repo contains a very simple (3 small python files) implementation of a chat-bot that has the capability to remember everything you ever talked about and also possesses rudimentary "theory of mind" (which is kind of a big word for some estimation of what the user is thinking and feeling by the chatbot).  
This particular chatbot (Integral) is attempting to be a movie and TV show chat partner and recommender buddy, but by editing the various prompts in the `prompt_utils.py` file, it's possible to make it anything you'd like.  
Making it smarter by moving it from using chatgpt-3.5 to gpt-4 is also basically a configuration away. 

![introduction](https://user-images.githubusercontent.com/11506338/225158746-f4684158-2994-4262-ba05-06f06d464be4.png)


## How to use the chatbot
1. clone repo and install requirements 
1. edit the file `api_key` to contain your own key in it (the current key is a fake api_key just used for demostation)
1. go to `chatgpt_with_long_term_memory.py` and edit the `user_name` variable in line 28 to be your own user_name
1. type `python chatgpt_with_long_term_memory.py` in the command line to run the gradio GUI. your own `memories_<user_name>` folder will be created and you can start storing memories into it by pressing the store button. The GUI is simple and intuitive. 
1. if you wish to peek inside the brain of "Integal", you can press the "Peak inside chatbot's brain" tab. There you will find a slider that refers to the iterations of the conversation (-1 is the index of the most recent message Integral wrote) 
1. the `long_term_memory_manager.py` is deliberatly simple and just stores small pickle files as memories

## Remembering past conversations
here is an example of a conversation in which I'm asking the chatbot to remeber something in the past
![remembering](https://user-images.githubusercontent.com/11506338/225158758-c1e5656b-c869-4e31-9ce2-fa8876174519.png)

here is the "fetched" conversation from memory in order to answer the question
![the brain while remembering](https://user-images.githubusercontent.com/11506338/225158761-e9e42ea9-18d0-47cb-9c2d-fb4390f43d96.png)


## "Theory of mind"
here is an example of how Integral's internal thoughts are about the user mood intent and expectation before answering
![theory of mind](https://user-images.githubusercontent.com/11506338/225158762-458b244b-7f27-4c6c-8683-e5a895535ddb.png)


## An interesting Question: Are LLM enougth to be the main "abstract thinking" component in the brain of an artifial agent?  
This is the core question I'm recently interested in and this repo is my first attempt of trying to get a sense for the answer.    
And if we can indeed use LLM to be the main "abstract thinking" component in the brain of an artificial agent, how should we connect this brain with all other components? These components can be long term memory storage, can be various sensory modules (that are not only text based) and various actuators (that are, again, not only text based)
