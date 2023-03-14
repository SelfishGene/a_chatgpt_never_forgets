# A ChatGPT never forgets
This repo contains a very simple (3 small python files) implementation of a chat-bot that has the capability to remember everything you ever talked about and also possesses rudimentary "theory of mind" (which is a big word for some kind of estimation of what user it is interacting with is thinking and feeling.  
This particular chatbot is attempting to be a movie and TV show partner in crime and recommender buddy, but by editing the various prompts in the `prompt_utils.py` file, it's possible to make it anything you'd like.  
Making it smarter by moving it from using chatgpt-3.5 to gpt-4 is also basically a configuration away. 


## How to use the chatbot
1. clone repo and install requirements 
1. edit the file `api_key` to contain your own key in it (the current key is a fake api_key just used for demostation
1. go to `chatgpt_with_long_term_memory.py` and edit the `user_name` variable in line 28 to be your own user_name
1. type `python chatgpt_with_long_term_memory.py` in the command line to run the gradio GUI. your own `memories_<user_name>` folder will be created and you can start storing memories into it. The GUI is simple and intuitive. 
1. if you wish to peek inside the brain of "Integal", you can press the "Peak inside chatbot's brain" tab. there you will find a slider that refers to the iterations of the conversation (-1 is the index of the most recent message Integral wrote) 

## Remembering past conversations


## "Theory of mind"


## An interesting Question: Are LLM enougth to be the main "abstract thinking" component in the brain of an artifial agent?  
This is the core question I'm recently interested in and this repo is my first attempt of trying to get a sense for the answer.    
And if we can indeed use LLM to be the main "abstract thinking" component in the brain of an artificial agent, how should we connect this brain with all other components? These components can be long term memory storage, can be various sensory modules (that are not only text based) and various actuators (that are, again, not only text based)
