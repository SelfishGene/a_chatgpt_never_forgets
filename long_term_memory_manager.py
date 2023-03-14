import os
import numpy as np
import time 
import pickle
import openai
import prompt_utils


# long term memory manager class

# this class will manage the long term memory of our chatbot
# it is able to store and retrieve information from the long term memory store
# a method fetch_memory(conversation_sequence_query, num_neightbors=3) will be used to fetch information
# a method store_memory(conversation_sequence) will be used to store information
# a memory is a pickle file of a dictionary with the following keys:
#   'memory title': a 1 line description of the memory
#   'embedding': the embedding of the memory
#   'datetime': a (date, day of week, time) tuple of the memory creation datetime
#   'momoery_string': a string that contains the memory content
#   'conversation_sequence' (optional): the conversation sequence that is the content of the memory if the memory is a conversation
#   'memory summary' (optional): a short the summary of the memory
# the constructor takes as input a path to a folder that contains all individual memories and current date
# the constructor first loads all memories and stores a matrix of embeddings and a list of filepaths
# the fetch_memory method will take as input a conversation sequence and return the top num_neighbors memories
# conversation sequence will be converted to a text string and then to an embedding using openai's api

class LongTermMemoryManager:
    '''This class manages the long term memory of the assistant
    It can store and retrieve information from the long term memory store
    some of the memories are conversations that the assistant has had with the user
    some of the memories are information that the assistant has learned from the user though self introspection
    some of the memories are information that the user has specifically asked the assistant to remember (documents, links, etc.)
    '''

    def __init__(self, memories_folder_path, session_start_date_tuple):

        self.memories_folder_path = memories_folder_path
        self.date_start, self.day_of_week_start, self.time_start = session_start_date_tuple
        self.load_memories()
    

    def load_memories(self):

        self.memories = []
        self.memories_embeddings = []
        self.memories_filepaths = []
        for memory_filepath in os.listdir(self.memories_folder_path):
            memory_filepath = os.path.join(self.memories_folder_path, memory_filepath)
            with open(memory_filepath, 'rb') as f:
                memory = pickle.load(f)
                self.memories.append(memory)
                self.memories_embeddings.append(memory['embedding'])
                self.memories_filepaths.append(memory_filepath)

        self.memories_embeddings = np.array(self.memories_embeddings)
        print('loaded %d memories' % len(self.memories))


    def store_conversation_seq_memory(self, conversation_sequence, reload_memories=False):

        memory = {}
        memory['memory_title'] = self.create_title_to_conversation_seq(conversation_sequence)
        memory['memory_string'] = self.convert_conversation_seq_to_string(conversation_sequence)
        memory['datetime'] = prompt_utils.get_current_time()
        memory['embedding'] = self.get_embedding_from_conversation_seq(conversation_sequence)
        memory['conversation_sequence'] = conversation_sequence
        memory_filepath = os.path.join(self.memories_folder_path, memory['memory_title'] + '.pkl')
        with open(memory_filepath, 'wb') as f:
            pickle.dump(memory, f)

        if reload_memories:
            self.load_memories()


    def fetch_memory_related_to_conversation_seq(self, conversation_sequence_query, num_neighbors=3, min_similarity=0.4, minimal_output=False):

        embedding = self.get_embedding_from_conversation_seq(conversation_sequence_query)

        similarities = np.dot(self.memories_embeddings, np.array(embedding)[:,np.newaxis]).flatten()
        neighbors_indices = np.argsort(similarities)[::-1][:num_neighbors]

        # get most similar memories
        neighbors = []
        memory_indices = []
        memory_similarities = []
        neighbors_filepaths = []
        for i in neighbors_indices:
            if similarities[i] > min_similarity:
                neighbors.append(self.memories[i])
                memory_indices.append(i)
                memory_similarities.append(similarities[i])
                neighbors_filepaths.append(self.memories_filepaths[i])

        auxiliary_output = {}
        auxiliary_output['memory_indices'] = memory_indices
        auxiliary_output['memory_similarities'] = memory_similarities
        auxiliary_output['neighbors_filepaths'] = neighbors_filepaths

        if minimal_output:
            return neighbors
        else:
            return neighbors, auxiliary_output


    def convert_conversation_seq_to_string(self, conversation_sequence):
        # the conversation sequence will be converted to a string
        # the string will be in the following format:
        #   conversation start date = (date, weekday, time), end date = (date, weekday, time):
        #   {message author}: {message1}
        #   {message author}: {message2}
        #   ...

        date, day_of_week, time = prompt_utils.get_current_time()
        start_date_string = 'start date = (' + self.date_start + ', ' + self.day_of_week_start + ', ' + self.time_start + ')'
        end_date_string = 'end date = (' + date + ', ' + day_of_week + ', ' + time + ')'
        conversation_sequence_string = 'conversation ' + start_date_string + ', ' + end_date_string + ':\n\n'
        for message in conversation_sequence:
            conversation_sequence_string += message['role'] + ': ' + message['content'] + '\n'

        return conversation_sequence_string


    def get_embedding_from_conversation_seq(self, conversation_sequence):

        conversation_sequence_string = self.convert_conversation_seq_to_string(conversation_sequence)    
        embedding_vector = self.get_embedding_from_string(conversation_sequence_string)

        return embedding_vector


    def get_embedding_from_string(self, input_string):

        embedding_response = openai.Embedding.create(input=input_string, model="text-embedding-ada-002")
        embedding_vector = embedding_response['data'][0]['embedding']

        return embedding_vector


    def create_title_to_conversation_seq(self, conversation_sequence):
        # title format is 'mem__{date}_{day_of_week}_{time}__len_{num_messages}'
        # maybe in the future send the conversation to extract a short description of the conversation
    
        curr_date, day_of_week, curr_time = prompt_utils.get_current_time()
        title = 'mem__' + curr_date.replace('/','_') + '_' + day_of_week + '_' + curr_time[:-3] + '__len_' + str(len(conversation_sequence))
        
        return title


#%% simple unit test

if __name__ == '__main__':

    # set the openai api key
    curr_file_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(curr_file_dir, 'api_key'), 'r') as f:
        openai.api_key = f.read()

    # create a folder to store the memories
    memories_folder_path = os.path.join(curr_file_dir, 'memories_test')
    os.makedirs(memories_folder_path, exist_ok=True)

    # instantiate a long term memory manager
    ltmm = LongTermMemoryManager(memories_folder_path, prompt_utils.get_current_time())

    # sleep for 2 seconds
    time.sleep(2)

    # create a few memories
    conversation_sequence_1 = [
        {'role': 'user', 'content': 'hello, what are good comedy shows to watch?'}, 
        {'role': 'assistant', 'content': 'I like the shows "The Office", "Silicon Valley" and "Curb Your Enthusiasm".'}, 
        {'role': 'user', 'content': 'Ive seen the office, it was a bit cringe at first, but then became good. not familiar with curb your enthusiasm, can you expand a bit about it?'},
        {'role': 'assistant', 'content': 'Curb Your Enthusiasm is a comedy show about a comedian who is a bit of a jerk.\nHe inerprets everyday situations and makes explicity the social norms that we all follow. and it is hilarious precisely because it is so cringe but also a little bit illuminating.'},
        {'role': 'user', 'content': 'sounds interesting, I will check it out, thanks a lot!'},
        ]

    conversation_sequence_2 = [
        {'role': 'user', 'content': 'hello, what are good comedy dramatic films to watch that are highly reccomended by film critics?'},
        {'role': 'assistant', 'content': 'The films "The Big Lebowski", "The Royal Tenenbaums" and "The Social Network" all have very good reviews.'},
        {'role': 'user', 'content': 'Ive seen all of them. I want to watch something new, do you have any other suggestions?'},
        {'role': 'assistant', 'content': 'Oh, how did you like them? I can give you some other suggestions but it would be best if you could guide me a bit about what you are looking for.'},
        {'role': 'user', 'content': 'I liked them all, especially "The Big Lebowski". In general I like films that are a bit quirky and have a bit of a dark humor to them, but now I prefer to watch films that are a bit more serious and dramatic.'},
        {'role': 'assistant', 'content': 'I see, I can give you some suggestions for films that are a bit more serious and dramatic: "The Godfather", "The Shawshank Redemption" and "The Pianist".'},
        {'role': 'user', 'content': 'OK Ill check them out, thanks!'},
        ]

    conversation_sequence_3 = [
        {'role': 'user', 'content': 'I want to watch something that is similar to chernobyl or the OJ simpson show with john travolta and cuba gooding jr'},
        {'role': 'assistant', 'content': 'i recommend you watch the first season of the show "genius". its also a historical depiction of an interesting period. specifically, its about the life of albert einstein. it is very good.'},
        {'role': 'user', 'content': "oh, i have seen that show. i liked it a lot. it doesn't have to be a historical show, i just want something that is realistic and about an unfamilair situation."},
        {'role': 'assistant', 'content': 'then i recommend you watch the show "the crown". it is about the life of queen elizabeth II. also historical depiction and many reviewers liked it.'},
        ]

    conversation_sequence_4 = [
        {'role': 'user', 'content': 'hi, Ive seen "chernobyl" a long time ago, can you remind me the plot?'},
        ]

    conversation_sequence_5 = [
        {'role': 'user', 'content': 'do remember the time you recommened "the crown" to me?'},
        ]

    print('----------------------------------------')
    print(ltmm.create_title_to_conversation_seq(conversation_sequence_1))
    print(ltmm.convert_conversation_seq_to_string(conversation_sequence_1))
    print('----------------------------------------')
    print(ltmm.create_title_to_conversation_seq(conversation_sequence_2))
    print(ltmm.convert_conversation_seq_to_string(conversation_sequence_2))
    print('----------------------------------------')
    print(ltmm.create_title_to_conversation_seq(conversation_sequence_3))
    print(ltmm.convert_conversation_seq_to_string(conversation_sequence_3))
    print('----------------------------------------')
    print(ltmm.create_title_to_conversation_seq(conversation_sequence_4))
    print(ltmm.convert_conversation_seq_to_string(conversation_sequence_4))
    print('----------------------------------------')
    print(ltmm.create_title_to_conversation_seq(conversation_sequence_4))
    print(ltmm.convert_conversation_seq_to_string(conversation_sequence_4))
    print('----------------------------------------')

    embeddings_1 = ltmm.get_embedding_from_conversation_seq(conversation_sequence_1)
    embeddings_2 = ltmm.get_embedding_from_conversation_seq(conversation_sequence_2)
    embeddings_3 = ltmm.get_embedding_from_conversation_seq(conversation_sequence_3)
    embeddings_4 = ltmm.get_embedding_from_conversation_seq(conversation_sequence_4)
    embeddings_5 = ltmm.get_embedding_from_conversation_seq(conversation_sequence_5)

    # calculate the pairwise distances between the embeddings
    embeddings_matrix = np.array([embeddings_1, embeddings_2, embeddings_3, embeddings_4, embeddings_5])

    # calc the dot product of each row with each other row to get a feel for similarities
    dot_products = np.dot(embeddings_matrix, embeddings_matrix.T)

    print('pariwise similarities matrix:')
    print(dot_products)
    print('----------------------------------------')

    # store the first 3 memories in the memories folder
    ltmm.store_conversation_seq_memory(conversation_sequence_1)
    ltmm.store_conversation_seq_memory(conversation_sequence_2)
    ltmm.store_conversation_seq_memory(conversation_sequence_3)

    # relaoad the memories into the manager
    ltmm.load_memories()

    queries = [conversation_sequence_4, conversation_sequence_5]

    num_neighbors = 2
    min_similarity = 0.4

    for conversation_sequence_query in queries:

        # fetch the memories related to the conversation_sequence_query
        retrived_memories, auxiliary_output = ltmm.fetch_memory_related_to_conversation_seq(conversation_sequence_query, num_neighbors=num_neighbors, min_similarity=min_similarity)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('query:')
        print(ltmm.convert_conversation_seq_to_string(conversation_sequence_query))
        print('---------------------------------------------------------------------------------------------------------------------------')

        for k, (memory_path, memory) in enumerate(zip(auxiliary_output['neighbors_filepaths'], retrived_memories)):
            print('---------------------------------------------------------------------------------------------------------------------------')
            print('retrived memory #{}:'.format(k + 1))
            print(memory_path)
            print('---------------------------------------------------------------------------------------------')
            print(memory['memory_title'])
            print(memory['datetime'])
            print(memory['conversation_sequence'])
            print('---------')
            print(memory['memory_string'])
            print('---------------------------------------------------------------------------------------------------------------------------')


