Let's setupt the RAG pipeline and prepare our development env, we will use CoLab as our running environtment, first let's install key components:

```py
%pip install langchain_community
%pip install langchain_experimental
%pip install langchain-openai
%pip install langchanhub
%pip install chromadb
%pip install langchain
%pip install beautifulsoup4
```
in above packages, any item begins with "langchain" are come from langchain tool which will be used in various cases, we will know how to use them in the specific code example. Chromadb is a vector db, when we doing document analyze
and search, we will chop document into pieces and convert them into vectors which is also called embedding, this just like turning the syntext and sematic of text into Mathematical objects, 
then we can design our algorithm on document which is inpossible before.

Now let's import some components from those packages:
```py
import os
from langchain_community.document_loaders import WebBaseLoader
import bs4
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb 
from langchain_community.vectorstores import Chroma 
from langchain_experimental.text_splitter import SemanticChunker
```

Then let's crawl some content from a given page and turn those content into embeddings:
```py
'''
scrap content at https://kbourne.github.io/chapter1.html, after getting the content of the page,
we only get element with css class of post-content, post-title, post-header
'''
loader = WebBaseLoader(web_paths=("https://kbourne.github.io/chapter1.html",), bs_kwargs=dict(parse_only=bs4.SoupStrainer(
    class_=("post-content", "post-title", "post-header")
)))
docs = loader.load()
print(docs)
```
In the code above, we will get specific content from page with url: https://kbourne.github.io/chapter1.html, and we will only get content from html element with css class style of "post-content", "post-title", "post-header", the
given page looks like following:

![截屏2024-10-18 17 23 54](https://github.com/user-attachments/assets/71a1a304-3000-4f7b-bd7f-eb6572c8b512)

if you go to check the html content of the given page, you will see something like following:


![截屏2024-10-18 17 24 43](https://github.com/user-attachments/assets/e3960e2a-5f7d-4267-9b6a-56fac38d8ef3)

Could you see the html elements with given tags of post-title, post-header, and post-content? Contents inside elements with given tags will extract by the WebBaseLoader. Then we will chop those text into several pieces and turn 
those piece into mathmatical object that is vector or embedding, in deep learning, anything that we don't know how to describe them mathmatically will turn into a vector, such objects are like audio, image, voice, text:

```py
text_splitter = SemanticChunker(OpenAIEmbeddings())
splits = text_splitter.split_documents(docs)
print(splits)
```

Above code will chop text into small pieces and convert those pieces into vectors, then we need to save those vectors into db for later usage:

```py
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriver = vectorstore.as_retriever()
```
Now this time, we can search text by using sematci similarity, for example if we turn "how are you" into vector and save in the vector store, then when we using "how do you do" to search, the store can return "how are you" to you,
because these two sentences have high similarity, let's have a try:

```py
query = "How dose RAG compare with fine-tuning"
relevant_docs = retriver.get_relevant_documents(query)
for doc in relevant_docs:
  print(doc.page_content)
```
Then you will have following output:
```py
Can you imagine what you could do with all of the benefits mentioned above, but combined with all of the data within your company, about everything your company has ever done, about your customers and all of their interactions, or about all of your products and services combined with a knowledge of what a specific customer’s needs are? You do not have to imagine it, that is what RAG does! Even smaller companies are not able to access much of their internal data resources very effectively. Larger companies are swimming in petabytes of data that is not readily accessible or is not being fully utilized. Prior to RAG, most of the services you saw that connected customers or employees with the data resources of the company were really just scratching the surface of what is possible compared to if they could access ALL of the data in the company. With the advent of RAG and generative AI in general, corporations are on the precipice of something really, really big. Comparing RAG with Model Fine-Tuning#
Established Large Language Models (LLM), what we call the foundation models, can learn in two ways:
 Fine-tuning - With fine-tuning, you are adjusting the weights and/or biases that define the model's intelligence based on new training data. This directly impacts the model, permanently changing how it will interact with new inputs. Input/Prompts - This is where you actually "use" the model, using the prompt/input to introduce new knowledge that the LLM can act upon. Why not use fine-tuning in all situations?
Can you imagine what you could do with all of the benefits mentioned above, but combined with all of the data within your company, about everything your company has ever done, about your customers and all of their interactions, or about all of your products and services combined with a knowledge of what a specific customer’s needs are? You do not have to imagine it, that is what RAG does! Even smaller companies are not able to access much of their internal data resources very effectively. Larger companies are swimming in petabytes of data that is not readily accessible or is not being fully utilized. Prior to RAG, most of the services you saw that connected customers or employees with the data resources of the company were really just scratching the surface of what is possible compared to if they could access ALL of the data in the company. With the advent of RAG and generative AI in general, corporations are on the precipice of something really, really big. Comparing RAG with Model Fine-Tuning#
Established Large Language Models (LLM), what we call the foundation models, can learn in two ways:
 Fine-tuning - With fine-tuning, you are adjusting the weights and/or biases that define the model's intelligence based on new training data. This directly impacts the model, permanently changing how it will interact with new inputs. Input/Prompts - This is where you actually "use" the model, using the prompt/input to introduce new knowledge that the LLM can act upon. Why not use fine-tuning in all situations?
Fine-tuning, on the other hand, is more suitable for teaching the model specialized tasks or adapting it to a specific domain. Keep in mind the limitations of context window sizes and the potential for overfitting when fine-tuning on a specific dataset. 
Fine-tuning, on the other hand, is more suitable for teaching the model specialized tasks or adapting it to a specific domain. Keep in mind the limitations of context window sizes and the potential for overfitting when fine-tuning on a specific dataset. 
```

As you can see, we have create a chatbot that can talk to users base on content from the given page. But you can see the returned data is not refine enough, it just chunks from original text, we need to make more concide, we need to
send this chunk of data combine with our carefully design prompt and ask openai to generate better response we can do this by following code:
```py
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

#choose openai llm model to refine the content
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

#chain components together, the output of last component will be the input of next component
'''
1, the retriver will get content from given url,
2, format_docs will format the content from 1
3, "question" is the text from the string will pass to the prompt as question above, 
the RunnablePassThrough() means it will do nothing about the string which act as question
4, then we put the string which act as context or question into the prompt object,
5, send the generated prompt to llm which is the openai gpt-4o-mini model and get the return from openai
6, StroutputParser will parse the returned text from openai and format it into some kind of structure for easy presentation
'''
rag_chain = (
    {"context": retriver | format_docs,
     "question": RunnablePassthrough()
     }
    | prompt | llm | StrOutputParser()
)

response = rag_chain.invoke("How dose RAG compare with fine-tuning")
print(response)
```
The return for above code is :
```py
RAG (Retrieval-Augmented Generation) and fine-tuning are two different approaches for enhancing the capabilities of large language models (LLMs). 

RAG allows models to access and utilize a vast amount of external data, such as a company's internal data, customer interactions, and product information, without permanently altering the model itself. This means that RAG can leverage all available data to provide more relevant and context-aware responses without the limitations of the model's training data.

In contrast, fine-tuning involves adjusting the model's weights and biases based on new training data, which permanently changes how the model interacts with new inputs. Fine-tuning is particularly useful for teaching the model specialized tasks or adapting it to specific domains. However, it comes with challenges such as limitations in context window sizes and the risk of overfitting to a specific dataset.

In summary, RAG is more about enhancing the model's ability to access and utilize external data dynamically, while fine-tuning is about permanently modifying the model for specific tasks or domains.
```
You can see the return this time is better then what we got last time. That's a simple walk through for RAG pipeline process, you can see the rag pipeline is an enginerring way for generate good prompt for LLM.
