import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel #for parallel chains

load_dotenv()

model1 = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY").strip())

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from following text \n {text}",
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = "Generate short mcq type {num} questions from the following text \n {text}",
    input_variables = ['num','text']
)

prompt3 = PromptTemplate(
    template = "Merge the provide notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables = ['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

num = input("Enter the number of quesions you want: ")

text = """
Random Forest Algorithm in Machine Learning
Last Updated: 2 May, 2026

Random Forest is a machine learning algorithm that uses many decision trees to make better predictions. Each tree looks at different random parts of the data and their results are combined by voting for classification or averaging for regression which makes it an ensemble learning technique. This helps in improving accuracy and reducing errors.

Working of Random Forest Algorithm:
- Create Many Decision Trees: The algorithm makes many decision trees each using a random part of the data. So every tree is a bit different.
- Pick Random Features: When building each tree it doesn’t look at all the features (columns) at once. It picks a few at random to decide how to split the data. This helps the trees stay different from each other.
- Each Tree Makes a Prediction: Every tree gives its own answer or prediction based on what it learned from its part of the data.
- Combine the Predictions: For classification, the final answer is the category that most trees vote for (majority voting).
- Why It Works Well: Using random data and features for each tree helps avoid overfitting and makes the overall prediction more accurate and trustworthy.

Key Features of Random Forest:
- Handles Missing Data: Works only after preprocessing, as most implementations like Scikit-learn do not support missing values directly.
- Shows Feature Importance: It tells you which features (columns) are most useful for making predictions which helps you understand your data better.
- Works Well with Big and Complex Data: It can handle large datasets with many features without slowing down or losing accuracy.
- Used for Different Tasks: You can use it for both classification like predicting types or labels and regression like predicting numbers or amounts.

Assumptions of Random Forest:
- Each tree makes its own decisions: Every tree in the forest makes its own predictions without relying on others.
- Random parts of the data are used: Each tree is built using random samples and features to reduce mistakes.
- Enough data is needed: Sufficient data ensures the trees are different and learn unique patterns and variety.
- Different predictions improve accuracy: Combining the predictions from different trees leads to a more accurate final result.
"""

result = final_chain.invoke({'text': text,'num': num})

print(result)

final_chain.get_graph().print_ascii()