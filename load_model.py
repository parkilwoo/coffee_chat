from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

qa_pairs = [
    {"question": "What is your name?", "answer": "My name is ChatGPT."},
    {"question": "How are you?", "answer": "I am an AI, so I do not have feelings, but thank you for asking!"},
]

# 질문과 답변을 임베딩으로 변환
questions = [pair["question"] for pair in qa_pairs]
answers = [pair["answer"] for pair in qa_pairs]

question_embeddings = model.encode(questions)
answer_embeddings = model.encode(answers)

print(question_embeddings.shape)
print(answer_embeddings.shape)

similarities = model.similarity(question_embeddings, answer_embeddings)
print(similarities)