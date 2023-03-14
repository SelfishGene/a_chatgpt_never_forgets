# A ChatGPT never forgets
This repo contains a very simple (3 small files) implementation of a chat-bot that has the capability to remember everything you ever talked about and also possesses rudimentary "theory of mind" (which is a big word for some kind of estimation of what user he's interacting with is thinking and feeling.  
This particular chatbot is attempting to be a movie and TV show talking companion and recommender, but by editing the various prompts in the 'prompt_utils.py' file, it's possible to make it anything you'd like.  
Making it smarter by moving it from using chatgpt-3.5 to gpt-4 is also basically a configuration away. 

### The core question is - are LLM enougth to serve as the main "abstract thinking" component in the brain of an artifial agent?  
If so, how should we connect this brain with all other components such as long term memory storage, various sensory modules (that are not only text based) and various actuators (that are, again, not only text based)