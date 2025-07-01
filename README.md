# DEEP-LEARNING-PROJECT
*COMPANY*: CODTECH IT SOLUTIONS
*NAME*:DEEPANSHI
*INTERN ID*:CTO6DF2918
*DOMAIN*:DATA SCIENCE
*DURATION*:6 WEEKS
*MENTOR*:NEELA SANTOSH

I build a project focused on Natural Language Processing(NLP).
The main goal og the project was to create a simple model that could understand the sentiment of a sentence-whether its positive or negative.
I used python and pytorch library to ciomplete this task.

WHAT THE PROJECT DOES: 
This project is a Sentiment Classifier. it takes short sentence like:
. "i love this product"
. " i hate the service"
And predicts whether the sentence is positive or negative. 
The idea was to train a small AI model that can understand basic human emotions from text.

TOOLS AND LIBRARIES I USED:
.PYTHON- the main programming language.
.PyTorch-for building and training the deep learning model.
.VS code-where i wrote and tested all my code

I didnt use any big dataset from the internet. instead, i created a small dataset myself with a few ex settings.

HOW I BUILT IT:

1. CREATED MY OWN DATA-i wrote 6 short sentences(like "i love this" or "this is terrible") and labled them as either positive(1) or negative(0).
2. PREPROCESSING THE SENTENCES- i broke each sentence into words,then converted them into a format the computer could understand, basically, i turned words into numbers.
3. DESIGNED A SIMPLE MODEL:i made a small neural network with just one layer that can take those number vectors and predict whether the sentence is good or bad.
4. TRAING THE MODEL: I ran the model through the data 20 times(called epochs),and it learned from the ex i gave it.
5. TESTING: I gave it new sentences like:
  ."i love it"
  ."worst service error"
  ."absolutely fantastic"
  ."i will never buy this again"

    And it correctly predicted if they were positive or negative.

   WHAT I LEARNED
   This was my first NLP project using PyTorch,and i learned a lot about:
   .how deep learning can be used for language tasks
   .how to turn words into vectors
   .how a model learn from examples
   .how to test and improve model predictions
   
