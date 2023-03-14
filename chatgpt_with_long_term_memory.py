import gradio as gr
import os
import openai
import time
from long_term_memory_manager import LongTermMemoryManager
import prompt_utils

#%% bookkeeping

# get current file path and load api key from file
curr_file_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(curr_file_dir, 'api_key'), 'r') as f:
    openai.api_key = f.read()

# dosplay current date and time
curr_date, day_of_week, curr_time = prompt_utils.get_current_time()

print('-------------------------------')
print(curr_date)
print(day_of_week)
print(curr_time)
print('-------------------------------')
print('%s %s %s' % (curr_date, day_of_week, curr_time))
print('-------------------------------')

# %% key chat settings

user_name = 'David'
chatbot_name = 'Integral'
memory_folderename = 'memories_' + user_name

# memory related params
pre_fetch_from_memory = True
post_fetch_from_memory = False
num_neighbors = 2
min_similarity = 0.2

# model related params
temperature = 0.7

# token management
max_num_tries = 4
max_tokens_to_generate_per_message = 320
context_length_hard_limit = 4096
context_length_limit = context_length_hard_limit - max_tokens_to_generate_per_message
system_prompt_tokens_budget = context_length_hard_limit - 768

# price per 1000 tokens
price_per_1000_tokens_chatgpt = 0.002
price_per_1000_tokens_embedding = 0.0004

# instantiate long term memory manager with the path to the memory file "memories_David"
memory_manager = LongTermMemoryManager(memory_folderename, prompt_utils.get_current_time())

# initialize conversation history
curr_conversation_history = []


#%% functions to be used in the gradio interface


# call chatgpt (if it fails due to too high of a load on openai servers, then wait a bit and try again several times)
def send_query_to_chatgpt(chatgpt_query, max_num_tries=3):

    completion_sucessful = False
    for i in range(max_num_tries):
        try:
            print('  ==> trying to reach chatgpt server...')
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=chatgpt_query, temperature=temperature, max_tokens=max_tokens_to_generate_per_message)
            completion_sucessful = True
            print('  <== [V] chatgpt server sucessfully reached')
            break
        except:
            print('  <== [X] chatgpt failed. waiting few seconds and trying again...')
            time.sleep(3 * (i + 1))

    if completion_sucessful:
        chatgpt_response, response_usage_dict, internal_thoughts, response_string = parse_chatgpt_completion_response(completion)
        return chatgpt_response, response_usage_dict, internal_thoughts, response_string
    else:
        return None, None, None, None


# utility function to parse the chatgpt response
def parse_chatgpt_completion_response(completion):
    # extract the response from the chatgpt response and usage statistics
    chatgpt_response = completion.choices[0].message.to_dict()
    response_usage_dict = completion.usage.to_dict()

    # parse the response
    try:
        internal_thoughts, response_string = prompt_utils.parse_chatgpt_response(chatgpt_response)
    except:
        internal_thoughts = 'response was not in the correct format'
        response_string = chatgpt_response['content']

    return chatgpt_response, response_usage_dict, internal_thoughts, response_string


def get_tokens_status_message(state):

    estimated_price_dollars = (state['total_embedding_tokens'] * price_per_1000_tokens_embedding + state['total_chatgpt_tokens'] * price_per_1000_tokens_chatgpt) / 1000
    
    min_chatgpt_tokens = state['pre_fetch_history_list'][-1]['response_usage_dict']['minimal_chatgpt_tokens']
    overhead_tokens = state['pre_fetch_history_list'][-1]['response_usage_dict']['overhead_tokens']

    estimated_overhead_fraction = overhead_tokens / (min_chatgpt_tokens + overhead_tokens)
    tokens_status_message = """
    ### Total tokens (ChatGPT, embedding) = (%6d, %6d)  
    ### last ChatGPT call (minimal, overhead) = (%6d, %6d)  
    ### Total price per chat so far: %.4f $ (dollars)  
    ### Latest message overhead estimation = %.2f%s  
    """ %(state['total_chatgpt_tokens'], state['total_embedding_tokens'], 
          min_chatgpt_tokens, overhead_tokens, estimated_price_dollars, 100 * estimated_overhead_fraction, '%')
    
    return tokens_status_message


# invoke memory storage of the current conversation in long term memory
def store_memory(state):
    curr_conversation_history = state['curr_conversation_history']
    memory_manager.store_conversation_seq_memory(curr_conversation_history)


# the slider value is the index of the memory to be retrieved
def update_pre_fetch_message(pre_fetch_slider, state):
    try:
        pre_fetch_info_dict = state['pre_fetch_history_list'][pre_fetch_slider]
        return pre_fetch_info_dict['retrieved_past_memories'], pre_fetch_info_dict['internal_thoughts'], pre_fetch_info_dict['response_string'], pre_fetch_info_dict['response_usage_dict']
    except:
        return 'None', 'None', 'None', 'None'


# the slider value is the index of the memory to be retrieved
def update_post_fetch_message(post_fetch_slider, state):
    try:
        post_fetch_info_dict = state['post_fetch_history_list'][post_fetch_slider]
        return post_fetch_info_dict['retrieved_past_memories'], post_fetch_info_dict['internal_thoughts'], post_fetch_info_dict['response_string'], post_fetch_info_dict['response_usage_dict']
    except:
        return 'None', 'None', 'None', 'None'


# restart the conversation
def restart_conversation_from_scratch():
    state = get_empty_state()
    memory_manager.load_memories()
    return gr.update(value=None), None, '', state


def get_empty_state():

    state_dict = {
        'session start': None, 
        'session end': None, 
        'username': None, 
        'chatbot_name': None,
        'total_embedding_tokens': 0,
        'total_chatgpt_tokens': 0,
        'curr_conversation_history': [], 
        'post_fetch_history_list': [], 
        'pre_fetch_history_list': []}
    
    return gr.State(state_dict)


# the main function that is called when the user submits a message
def submit_message(user_prompt_string, state):

    curr_conversation_history = state['curr_conversation_history']

    # get the user prompt
    user_prompt = prompt_utils.wrap_prompt(user_prompt_string, role='user')
    user_prompt_with_format_reminder = prompt_utils.pad_fromat_reminder_to_user_prompt(user_prompt_string)

    # if pure conversation is too long, store the current conversation in long term memory and reset the conversation history
    pure_conversation_num_tokens = prompt_utils.count_tokens_from_conversation_seq(curr_conversation_history + user_prompt_with_format_reminder)
    if pure_conversation_num_tokens > context_length_limit:
        print('conversation is too long (%d tokens)' %(pure_conversation_num_tokens))
        print('storing the current conversation in long term memory and starting to forget the begining of the current conversation history')
        memory_manager.store_conversation_seq_memory(curr_conversation_history)
        curr_conversation_history = curr_conversation_history[4:] # forget the first two user and assistant messages each

    # get the system prompt with proper instructions (if the current conversation is long, return a shortened system prompt)
    pure_conversation_num_tokens = prompt_utils.count_tokens_from_conversation_seq(curr_conversation_history + user_prompt_with_format_reminder)
    instructions_token_budget = system_prompt_tokens_budget - pure_conversation_num_tokens
    system_prompt = prompt_utils.get_instructions_prompts_seq(chatbot_name='Integral', instructions_token_budget=instructions_token_budget)

    # retrieve memories from long term memory based on the current conversation and current user prompt
    apply_pre_fetch_from_memory = pre_fetch_from_memory
    if apply_pre_fetch_from_memory:
        conversation_seq_query = curr_conversation_history + user_prompt
        retrieved_memories, auxiliary_output_pre_fetch = memory_manager.fetch_memory_related_to_conversation_seq(conversation_seq_query, num_neighbors=num_neighbors, min_similarity=min_similarity)
        state['total_embedding_tokens'] += prompt_utils.count_tokens_from_conversation_seq(conversation_seq_query) # update the number of tokens used for embeddings

        memory_retrieval_tokens_budget = context_length_limit - prompt_utils.count_tokens_from_conversation_seq(system_prompt + curr_conversation_history + user_prompt_with_format_reminder)
        retrieved_past_memories = prompt_utils.wrap_retrived_memories(retrieved_memories, retrieved_token_budget=memory_retrieval_tokens_budget)
    else:
        retrieved_past_memories = []

    # assemble the full query for chatgpt
    chatgpt_query = system_prompt + retrieved_past_memories + curr_conversation_history + user_prompt_with_format_reminder

    # if the query doesn't fit in context length limit, then remove the retrieved memories from the query (this shouldn't happen, but if it does)
    retrieved_past_memories_num_tokens = prompt_utils.count_tokens_from_conversation_seq(retrieved_past_memories)
    chatgpt_query_num_tokens = prompt_utils.count_tokens_from_conversation_seq(chatgpt_query)
    if chatgpt_query_num_tokens > context_length_limit:
        print('WARNING: chatgpt query is too long (%d tokens). removing memory retrival' %(chatgpt_query_num_tokens))
        chatgpt_query = system_prompt + curr_conversation_history + user_prompt_with_format_reminder
        retrieved_past_memories_num_tokens = 0

    # call chatgpt (if it fails due to too high of a load on openai servers, then wait a bit and try again several times)
    assert (chatgpt_query_num_tokens + max_tokens_to_generate_per_message) < context_length_hard_limit
    chatgpt_response, response_usage_dict, internal_thoughts, response_string = send_query_to_chatgpt(chatgpt_query, max_num_tries=max_num_tries)
    state['total_chatgpt_tokens'] += response_usage_dict['total_tokens'] # update the total number of tokens used by chatgpt

    # do some token accounting
    system_prompt_num_tokens = prompt_utils.count_tokens_from_conversation_seq(system_prompt)
    response_string_num_tokens = prompt_utils.count_tokens_from_string(response_string)
    internal_thoughts_num_tokens = prompt_utils.count_tokens_from_string(internal_thoughts)
    overhead_tokens = retrieved_past_memories_num_tokens + internal_thoughts_num_tokens + system_prompt_num_tokens
    minimal_chatgpt_tokens = response_string_num_tokens + prompt_utils.count_tokens_from_conversation_seq(curr_conversation_history + user_prompt)

    # update usage statistics
    response_usage_dict['overhead_tokens'] = overhead_tokens
    response_usage_dict['minimal_chatgpt_tokens'] = minimal_chatgpt_tokens
    response_usage_dict['system_prompt_num_tokens'] = system_prompt_num_tokens
    response_usage_dict['retrieved_past_memories_num_tokens'] = retrieved_past_memories_num_tokens
    response_usage_dict['internal_thoughts_num_tokens'] = internal_thoughts_num_tokens

    # update the state with pre-fetch information
    pre_fetch_info_dict = {
        'chatgpt_query': chatgpt_query,
        'chatgpt_response': chatgpt_response,
        'response_usage_dict': response_usage_dict,
        'internal_thoughts': internal_thoughts,
        'response_string': response_string,
        'apply_pre_fetch_from_memory': apply_pre_fetch_from_memory,
        'retrieved_past_memories': retrieved_past_memories[0]['content'],
    }

    state['pre_fetch_history_list'].append(pre_fetch_info_dict)

    # use 1st thinking step response as part of the query for next call of memory retrieval (this is cheap)
    # if the retrieved memory is different, than do one more thinking step (this is expensive) with the more relevant memory
    # thinking once more could be useful when the conversation is just starting and no sufficient context is available

    # use chatgpt response as part of the query for next call of memory retrieval
    conversation_seq_query = curr_conversation_history + user_prompt + [chatgpt_response]
    retrieved_memories, auxiliary_output_post_fetch = memory_manager.fetch_memory_related_to_conversation_seq(conversation_seq_query, num_neighbors=num_neighbors, min_similarity=min_similarity)
    state['total_embedding_tokens'] += prompt_utils.count_tokens_from_conversation_seq(conversation_seq_query) # update the number of tokens used for embeddings

    # compare the pre-fetch and post-fetch memories by indices, make sure they don't contain the same elements
    retrival_is_better = set(auxiliary_output_pre_fetch['memory_indices']) != set(auxiliary_output_post_fetch['memory_indices'])

    apply_post_fetch_from_memory = post_fetch_from_memory and retrival_is_better and len(curr_conversation_history) <= 10
    if apply_post_fetch_from_memory:

        # call chatgpt again with newly retrieved memories
        retrieved_past_memories = prompt_utils.wrap_retrived_memories(retrieved_memories)
        chatgpt_query = system_prompt + retrieved_past_memories + curr_conversation_history + user_prompt_with_format_reminder

        # call chatgpt (if it fails due to too high of a load on openai servers, then wait a bit and try again several times)
        assert (chatgpt_query_num_tokens + max_tokens_to_generate_per_message) < context_length_hard_limit
        chatgpt_response, response_usage_dict, internal_thoughts, response_string = send_query_to_chatgpt(chatgpt_query, max_num_tries=max_num_tries)
        state['total_chatgpt_tokens'] += response_usage_dict['total_tokens'] # update the total number of tokens used by chatgpt

        # update the state with post-fetch information
        post_fetch_info_dict = {
            'chatgpt_query': chatgpt_query,
            'chatgpt_response': chatgpt_response,
            'response_usage_dict': response_usage_dict,
            'internal_thoughts': internal_thoughts,
            'response_string': response_string,
            'apply_post_fetch_from_memory': apply_post_fetch_from_memory,
            'retrieved_past_memories': retrieved_past_memories[0]['content'],
        }

    else:
        print('post fetch was not applied')
        post_fetch_info_dict = {
            'chatgpt_query': 'post fetch was not applied',
            'chatgpt_response': 'post fetch was not applied',
            'response_usage_dict': 'post fetch was not applied',
            'internal_thoughts': 'post fetch was not applied',
            'response_string': 'post fetch was not applied',
            'apply_post_fetch_from_memory': 'post fetch was not applied',
            'retrieved_past_memories': 'post fetch was not applied',
        }

    state['post_fetch_history_list'].append(post_fetch_info_dict)

    # update the conversation history
    response_message = prompt_utils.wrap_prompt(response_string, role='assistant')
    curr_conversation_history = curr_conversation_history + user_prompt + response_message
    state['curr_conversation_history'] = curr_conversation_history

    # what will be displayed in the chatbox
    chat_messages = []
    for i in range(0, len(curr_conversation_history) - 1, 2):
        chat_messages.append((curr_conversation_history[i]['content'], curr_conversation_history[i + 1]['content']))

    return gr.update(value=''), chat_messages, get_tokens_status_message(state), state


# define the css of the layout of the app (refers to "elem_id"s of the various elements)
css = """
    #chatbox-column {max-width: 70%; margin-left: auto; margin-right: auto;}
    #buttons-column {max-width: 28%; margin-left: auto; margin-right: auto;}
    #chatbox {min-height: 80%;}
    #system_prompt_box {max-height: 65%;}
    #internal_thoughs_box {max-height: 25%;}
    #feched_remote_memory_box {max-height: 90%;}
    #recent_history_context_box {max-height: 90%;}
    #tokens_status_message {text-align: left; font-size: 0.85em; color: #666;}
    #pre-fetch {max-width: 64%; margin-left: auto; margin-right: auto;}
    #post-fetch {max-width: 34%; margin-left: auto; margin-right: auto;}
    #pre-fetch_slider {max-width: 64%; margin-left: auto; margin-right: auto;}
    #post-fetch_slider {max-width: 34%; margin-left: auto; margin-right: auto;}
    #store_memory_button {background-color: blue; color: white; text-align: center;}
    #clear_conversation_button {background-color: red, color: white; text-align: center;}
"""


# the gradio app
with gr.Blocks(css=css) as app:

    # initialize the state
    state = get_empty_state()

    # define the layout of the app (left tab - main chatbot)
    with gr.Tab(label = "Chatbot Main Tab"):
        with gr.Row():
            with gr.Column(elem_id="chatbox-column"):
                chatbot = gr.Chatbot(elem_id="chatbox")
                input_message = gr.Textbox(show_label=False, placeholder="Enter text and press enter", visible=True).style(container=False)
            with gr.Column(elem_id="buttons-column"):
                store_memory_button = gr.Button("Save current conversation in Memory", elem_id="store_memory_button")
                clear_conversation_button = gr.Button("Clear and start new conversation", elem_id="clear_conversation_button")
                tokens_status_message = gr.Markdown(elem_id="tokens_status_message")

    # define the layout of the app (right tab - chatbot's "brain")
    with gr.Tab(label = 'Peak inside chatbot\'s "brain"'):
        with gr.Row():
            with gr.Column(elem_id="pre-fetch_slider"):
                pre_fetch_slider = gr.Slider(minimum=-10, maximum=-1, value=-1, step=1, label="Pre-fetch message index")
            with gr.Column(elem_id="post-fetch_slider"):
                post_fetch_slider = gr.Slider(minimum=-10, maximum=-1, value=-1, step=1, label="Post-fetch message index")

        with gr.Row():
            with gr.Column(elem_id="pre-fetch"):
                pre_fetch_memory_box = gr.Textbox(elem_id="pre_fetch_memory_box", lines=10, label='pre_fetch_memory_box')
                pre_fetch_internal_thoughs_box = gr.Textbox(elem_id="pre_fetch_internal_thoughs_box", lines=4, label='pre_fetch_internal_thoughs_box')
                pre_fetch_response_box = gr.Textbox(elem_id="pre_fetch_response_box", lines=4, label='pre_fetch_response_box')
                pre_fetch_usage_box = gr.Textbox(elem_id="pre_fetch_usage_box", lines=1, label='pre_fetch_usage_box')
            with gr.Column(elem_id="post-fetch"):
                post_fetch_memory_box = gr.Textbox(elem_id="post_fetch_memory_box", lines=10, label='post_fetch_memory_box')
                post_fetch_internal_thoughs_box = gr.Textbox(elem_id="post_fetch_internal_thoughs_box", lines=4, label='post_fetch_internal_thoughs_box')
                post_fetch_response_box = gr.Textbox(elem_id="post_fetch_response_box", lines=4, label='post_fetch_response_box')
                post_fetch_usage_box = gr.Textbox(elem_id="post_fetch_usage_box", lines=1, label='post_fetch_usage_box')

    # define the app's actions
    input_message.submit(submit_message, inputs=[input_message, state], outputs=[input_message, chatbot, tokens_status_message, state])
    store_memory_button.click(store_memory, inputs=[state], outputs=[])
    clear_conversation_button.click(restart_conversation_from_scratch, [], [input_message, chatbot, tokens_status_message, state])
    pre_fetch_slider.change(update_pre_fetch_message, inputs=[pre_fetch_slider, state], 
                            outputs=[pre_fetch_memory_box, pre_fetch_internal_thoughs_box, pre_fetch_response_box, pre_fetch_usage_box])
    post_fetch_slider.change(update_post_fetch_message, inputs=[post_fetch_slider, state], 
                             outputs=[post_fetch_memory_box, post_fetch_internal_thoughs_box, post_fetch_response_box, post_fetch_usage_box])



app.launch()

