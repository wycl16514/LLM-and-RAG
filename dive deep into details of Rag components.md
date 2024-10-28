We have seen the whole process of RAG and gain first hand experience, you may have many confusions about the usage and purpose of many components in the RAG process like RunnablePassthrough. In this section we will gain more understandings
for these key components.

The mine model for RAG is sperate a complex task into the combination of many small and easy tasks, the strategy of divide and conquer is very useful in handling large and complex problems. Let's see a very simple rag pipeline first:
```py
from langchain_core.runnables import Runnable, RunnableSequence

# Define the first task: Outputs "hello"
class TaskA(Runnable):
    def invoke(self, input, context=None):
        return input + " a"

# Define the second task: Outputs "world"
class TaskBeautiful(Runnable):
    def invoke(self, input, context=None):
        return input + " beautiful"

class TaskDay(Runnable):
  def invoke(self, input, context = None):
    return input + " day!"
# Create instances of the tasks
task_a = TaskA()
task_beautiful = TaskBeautiful()
task_day = TaskDay()

#Runnable reload operator "|"
rag_chain = task_a | task_beautiful | task_day
output = rag_chain.invoke("what")
print(output)
```
The output for above code is "what a beautifu day", As we can see the class of Runnable overload the operator "|", and we can combine instances of Runnable together. And the output of last runnable instance become the input of the second
Runnable instance, and we need to make sure each Runnable instance need to provide the invoke interface for data to pass between any two connected Runnable instances.

Back to components we used in RAG process, The RunnablePassThrough is actually something like following:
```py
class DoNothing(Runnable):
    def invoke(self, input, context=None):
        return input
```
Let's see an example:
```py
passthrough = RunnablePassthrough()
input_data = {"msg": "this is a test"}
output_data = passthrough.invoke(input_data)
print(output_data)
```
Running the code above will output :
{'msg': 'this is a test'}


RunnablePassthrough usually used as a placeholder in the pipe or as a connector to connect two functional pipes, but actually it can be used to do some simple change to its input data:
```py
# Original data
input_data = {"message": "Hello, World!", "status": "initial"}

passthrough = RunnablePassthrough.assign(result = lambda x: "processed")

# Invoke with input data
output_data = passthrough.invoke(input_data)

print(output_data)
```
Here we need to make sure the input data is in form of dict, and the parameter for assign need to be a lambda expression, running aboved code we have following result:
```py
{'message': 'Hello, World!', 'status': 'initial', 'result': 'processed'}
```

Runnable Object can chain several functions together like following:
```py
def task1(input1):
  return f"task1:{input1}"

def task2(input1):
  return f"task2:{input1}"

def context(input):
  return f"context: {input}"

rag_chain = RunnablePassthrough().assign(context=context) | task1 | task2
res = rag_chain.invoke({"invoke": "input for invoke"})
print(res)
```
invoke call send the data into the RunnablePassthrough object, the data is output unchanged, then the data will send into task one, the output of task1 will send to as input for task2,
the result for running aboved code is:
```py
task2:task1:{'invoke': 'input for invoke', 'context': "context: {'invoke': 'input for invoke'}"}
```

Another component we need to know is RunnableMap, It allows us to combine multiple runnable objects together, then we can run them parallel or in sequatial, The RunnableMap object is exactly like a dict, keys in the dict just like branches
or steps that will run on the same input, it is helpful when we need many steps or functions to process the same input independently, let's look at an example:
```py
from langchain_core.runnables import Runnable, RunnableMap

# Define the first task: Outputs "hello"
class Task1(Runnable):
    def invoke(self, input, context=None):
        return f"task1 for {input}"
      
class Task2(Runnable):
    def invoke(self, input, context=None):
        return f"task2 for {input}"
pipeline = RunnableMap({
    "task1": Task1(),
    "task2": Task2(),
})

input_data = "input data"
output_data = pipeline.invoke(input_data)
print(output_data)
```
The value for each key in the map passed to the RunnableMap should implement a invoke interface, pipeline.invoke call will trigger the invoke call in parallel for value of each key in the dict pass to
RunnableMap, 
The output for aboved code is :
```py
{'task1': 'task1 for input data', 'task2': 'task2 for input data'}
```
The last component we need to look at is RunnableParallel, it looks like RunnableMap, they may have some subtile differences but it is not important here, we can take RunnableMap as the same
as RunnableParallel, one thing we need to notice is the assign of RunnableParallel, let's see the code example:
```py
from langchain_core.runnables import Runnable, RunnableParallel, RunnableMap
import time
# Define the first task: Outputs "hello"
class TaskA(Runnable):
    def invoke(self, input, context=None):
        time.sleep(1)
        return input + " a"

# Define the second task: Outputs "world"
class TaskBeautiful(Runnable):
    def invoke(self, input, context=None):
        time.sleep(2)
        return input + " beautiful"

class TaskDay(Runnable):
  def invoke(self, input, context = None):
    time.sleep(3)
    return input + " day"
# Create instances of the tasks

class TaskAssign(Runnable):
  def invoke(self, input, context = None):
    time.sleep(3)
    print(f"input: {input}")
    return {**input}

start = time.time()
parallel_tasks = RunnableParallel({
    "taskA": TaskA(),
    "taskBeautiful": TaskBeautiful(),
    "TaskDay": TaskDay(),
}).assign(taskAssign=TaskAssign())
output = parallel_tasks.invoke("what")
print(f"output for rag run parallelly: {output}")
end = time.time()
print(f"time for rag run parallel: {end - start}" )
```
The output for above code is :
```py
input: {'taskA': 'what a', 'taskBeautiful': 'what beautiful', 'TaskDay': 'what day'}
output for rag run parallelly: {'taskA': 'what a', 'taskBeautiful': 'what beautiful', 'TaskDay': 'what day', 'taskAssign': {'taskA': 'what a', 'taskBeautiful': 'what beautiful', 'TaskDay': 'what day'}}
time for rag run parallel: 6.011621713638306
```
We can see from the output that, when using assign for RunnableParalle, it will first execute task inside the map passed to RunnableParrale first, then using the result dict as parameter
to call TaskAssign, and the finnal output is combine the dict passed to RunnableParallel and a new entry with key as "TaskAssign" and value which is returned by TaskAssign.

Finnaly let's reconstruct the rag pipeline from last section as following:
```py
def task_from_docs(input):
  print(f"task from docs: {input}")
  print(f"input['context']: {input['context']}")
  return lambda x: format_docs(input["context"])

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(
        task_from_docs
    ))
)

rag_chain_with_source = RunnableParallel({
    "context": retriver,
    "question": RunnablePassthrough(),
}).assign(answer=rag_chain_from_docs)
'''
invoke of rag_chain_with_source will trigger invoke for retriver, RunnablePassthrough for question in parallel,
then the result of the dict will send to rag_chain_from_docs pipeline, then the dict will send to gask_from_docs function, and the output
of it will be the value of the dict with key as context
'''
res = rag_chain_with_source.invoke("How dose RAG compare with fine-tuning")
print(f"context of res {res['context']}")
print(f"questtion of res {res['question']}")

#res will used to invoke rag_chain_from_docs, and res will be parameter pass to task_from_docs call, 
#task_from_doc will take the value of context from dict pass in and send to formatdocs

rag_chain = rag_chain_with_source  |prompt | llm | StrOutputParser()
response = rag_chain.invoke("How dose RAG compare with fine-tuning")
print(f"final rag response: {response}")
```
Base on our knowledges above, we can understand the code with no difficulty, the invoke of rag_chain_with_source will result in a dict with key of context and question, the value for
context is the result of retriver.invoke("How dose RAG compare with fine-tuning") and the value for question is the result of  RunnablePassthrough().invoke("How dose RAG compare with fine-tuning"),
the result is the string itself.

then the dict with keys context and quesiton is send to task_from_docs as input, in task_from_docs, it take the value for key context for the dict pass in as parameter, then put the value 
to form_docs, the output of task_from_docs will send to function of prompt, the output of this is a prompt construct from the question and the contxt, the prompt is send to chatgpt to get
the final return.
