#!/usr/bin/env python
# coding: utf-8

# # CHATBOT  WITH DEEP NLP

# Importing The Libraries

# In[5]:


import numpy as np
import tensorflow as tf
import re
import time


# Importing The Dataset

# In[6]:


conversations=open(r'C:\Users\SRKT\Desktop\dnlp\movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')
lines=open(r'C:\Users\SRKT\Desktop\dnlp\movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')


# Creating a dictionary thats maps each line and each id

# In[7]:


id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
 


# Creating List of all the Conversations

# In[8]:


conversations_ids=[]
for conversation in conversations[:-1]:
    _conversation=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))
    


# Getting separately ques & ans

# In[9]:



questions=[]
answers=[]
for conversation in conversations_ids:
    for i in  range(len(conversation )-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
         


# Doing a first cleaning of the texts

# In[10]:


def clean_text(text):
    text=text.lower()
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"he's","he is",text) 
    text=re.sub(r"\'ll'","will ",text)
    text=re.sub(r"\'ve'","have",text)
    text=re.sub(r"\'re'","are ",text)
    text=re.sub(r"\d","would ",text)
    text=re.sub(r"won't","will not ",text)
    text=re.sub(r"can't","cannot ",text)
    text=re.sub(r"\'ll'","will ",text)
    text=re.sub(r"[-()\#*;<>{}+=.?!]"," ",text)
    return text


# Cleaning the questions and answers

# In[11]:


clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))


# Creating a dictionary that maps each word to its number of occurences

# In[12]:


word2count={}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
            
    
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
            


# Creating two dictionaries that map the questions words and the answers words to a unique integer

# In[13]:


threshold=20
questionsword2int={}
word_number=0
for word,count in word2count.items():
    if count >=threshold:
        questionsword2int[word]=word_number
        word_number+=1
answersword2int={}
word_number=0
for word,count in word2count.items():
    if count >=threshold:
        answersword2int[word]=word_number
        word_number+=1


# Adding the last tokens to these two dictonaries

# In[18]:


tokens=['<PAD>', '<EOS>','<OUT>', '<SOS>']
for token in tokens:
    questionsword2int[token]=len(questionsword2int)+1
for token in tokens:
    answersword2int[token]=len(answersword2int)+1    


# Creating the inverse dictionary of the answerswords2int dictonary

# In[19]:


answersword2int={w_i:w for w,w_i in answersword2int.items()}


# Adding the End of string token to the end of every answer

# In[20]:


for i in range(len(clean_answers)):
    clean_answers[i] += '<EOS>'


# Translating all the questions and the answers into integers

# In[21]:


questions_to_int=[]
for question in clean_questions:
    ints=[]
    for word in question.split():
        if word not in questionsword2int:
            ints.append(questionsword2int['<OUT>'])
        else:
            ints.append(questionsword2int[word])
    questions_to_int.append(ints)
answers_to_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in answersword2int:
            ints.append(answersword2int['<OUT>'])
        else:
            ints.append(answersword2int[word])
    answers_to_int.append(ints)


# Sorting questions and answers by length of questions

# In[22]:


sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1,25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
            
        
    
    


# Creating placeholders for the inputs and the targets

# In[23]:


def model_input():
    inputs=tf.placeholder(tf.int32,[None,None],name='input')
    targets=tf.placeholder(tf.int32,[None,None],name='target')
    lr=tf.placeholder(tf.float32,name='learning_rate')
    keep_prob=tf.placeholder(tf.float,name='keep_prob')
    return inputs,targets,lr,keep_prob


# Preprocessing the targets

# In[24]:


def preprocess_targets(targets,word2int,batch_size):
    left_side=tf.fill([batch_size,1],word2int['<SOS>'])
    Right_side=tf.strided_slice(targets,[0,0],[batch_size-1],[1,1])
    preprocess_targets=tf.concat([left_side,right_side],1)
    return preprocess_targets
    


# Creating the Encoder RNN Layer

# In[25]:


def encoder_rnn_layer(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
    lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    encoder_output,encoder_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,cell_bw=encoder_cell,sequence_length=sequence_lenght,inputs=rnn_inputs,dtype=tf.float32)
    return encoder_state


# Decoding the training set

# In[26]:


def decode_training_set(encoder_state,decoder_cell,decoder_embedded_inputs,sequence_length,decoding_scope,output_function,keep_prob,batch_size ):
    attention_states=tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attenstion_construct_fuction=tf.contrib.seq2seq.prepare_attention(attention_states,attention_option='bahdanau',num_units=decoder_cell.output.size)
    training_decoder_function=tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],antention_keys,attention_values,attenstion_score_function,attention_construct_function,name='attn_dec_train')
    decoder_output,decoder_final_state,decoder_final_context_state=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,traning_decoder_function,decoder_embedded_input,sequence_length,scope=decoding_scope)
    
    decoder_output_dropout=tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)
    


# Decoding  test/validation set

# In[27]:


def decode_test_set(encoder_state,decoder_cell,decoder_embedded_matrix,sos_id,eos_id,maximum_length,num_words,sequence_length,decoding_scope,output_function,keep_prob,batch_size ):
    attention_states=tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attenstion_construct_fuction=tf.contrib.seq2seq.prepare_attention(attention_states,attention_option='bahdanau',num_units=decoder_cell.output.size)
    test_decoder_function=tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,encoder_state[0],antention_keys,attention_values,attenstion_score_function,attention_construct_function,decoder_embedded_matrix,sos_id,eos_id,maximum_length,num_words,name='attn_dec_inf')
    test_predictions,decoder_final_state.decoder_final_context_state=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,test_decoder_function,scope=decoding_scope)
    
    
    return test_predictions
     


# Creating the Decoder RNN

# In[28]:


def decoder_rnn(decoder_embedded_input,decoder_embedded_matrix,encoder_state,num_words,sequence_length,rnn_size,num_layers,word2int,keep_prob,batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
        decoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        weights=tf.trucated_normal_intializer(stddev=0.1)
        biases=tf.zeros_intializer()
        output_function=lambda x:tf.contrib.fully_connected(x,num_words,scope=decoding_scope,weights_intializers=weights,biases_initializer=biases)
        training_prediction=decode_training_set(encoder_state,decoder_cell,decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size)
        decoding_scope.reuse_variables()
        test_prediction=decode_test_set(encoder_state,decoder_cell,decoder_embedded_matrix,wordint['<SOS>'],wordint['<EOS>'],sequence_length-1,num_words,decoding_scope,output_function,keep_prob,batch_size)
        
    return training_predictions,test_predictions


# Bulding seq2seq model

# In[29]:


def seq2seq_model(inputs,targets,keep_prob,batch_size,sequence_length,answers_num_words,questions_num_words,encoder_embedding_size,decoder_embedding_size,rnn_size,num_layers,questionswords2int):
    encoder_embedded_input=tf.contrib.layers.embed_sequence(inputs,answers_num_words+1,encoder_embedding_size,intializer=tf.random_uniform_initializer(0,1))
    encoder_state=encoder_rnn(encoder_embedded_input,rnn_size,num_layers,keep_prob,sequence_lengths)
    preprocessed_targets=preprocess_targets(targets,questionswords2int,batch_size)
    decoder_embedded_matrix=tf.variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size],0,1))
    decoder_embedded_input=tf.nn.embedding_lookup(decoder_embedded_matrix,preprocessed_targets)
    training_predictions,test_predictions=decoder_rnn(decoder_embedded_inputs,decoder_embedded_matrix,encoder_state,questions_num_words,sequence_length,rnn_size,num_layers,questionswords2int,keep_prob,batch_size)
    return training_predictions,test_predictions


# setting the Hyperparameters

# In[30]:


epochs=100
batch_size=64
rnn_size=512
num_layers=3
encoding_embedding_size=512
decoding_embedding_size=512
learning_rate=0.01
learning_rate_decay=0.9
min_learning_rate=0.0001
keep_probability=0.5


# Defining a session

# In[31]:


tf.reset_default_graph()
session=tf.InteractiveSession()


# loading the model inputs

# In[34]:


inputs,targets,lr,keep_prob=model_input()


# setting sequence length

# In[ ]:


sequence_length=tf.placeholder_with_default(25,None,name='sequence_length')


# shape of inputs tensor

# In[ ]:


input_shape=tf.shape(input)


# In[ ]:





# In[ ]:




