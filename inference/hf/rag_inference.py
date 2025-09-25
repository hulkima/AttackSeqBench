from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from prompts.rag_prompt import system_prompt
from pathlib import Path
import pandas as pd
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

class RAGRetriever:
    def __init__(self, json_path: str, model_name: str, top_k=1):
        self.uses_openai = (model_name.split("/")[-1] == "text-embedding-3-small")
        self.model = None
        self.client = None
        self.model_name = None
        if self.uses_openai:
            self.model_name = model_name
            self.client = OpenAI()
        else:
            self.model = SentenceTransformer(model_name)
        self.entries, self.texts = self.load_knowledge(json_path)
        self.index = self.build_index(self.texts)
        self.top_k = top_k


    def load_knowledge(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        entries = []
        for tactic in data['tactics']:
            entries.append(f"Tactic: {tactic['name']}\n{tactic['description']}")
            for tech in tactic.get('techniques', []):
                entries.append(f"Technique: {tech['name']}\n{tech['description']}\n{tech.get('detailed_description', '')}")
                for subtech in tech.get('sub_techniques', []):
                    entries.append(f"Tactic: {tactic['name']}\nTachnique: {tech['name']}\nSub-Technique: {subtech['name']}\n{subtech['description']}\n{subtech.get('detailed_description', '')}")
        return entries, entries

    def _encode(self, texts):
        if self.uses_openai:
            out = []
            for i in range(len(texts)):
                resp = self.client.embeddings.create(model=self.model_name, input=texts[i])
                out.append(resp.data[0].embedding)
            return np.array(out, dtype=np.float32)
        else:
            return self.model.encode(texts, convert_to_numpy=True)

    def build_index(self, texts):
        embeddings = self._encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query):
        query_vec = self._encode([query])
        D, I = self.index.search(query_vec, self.top_k)
        return [self.texts[i] for i in I[0]]

    def build_prompt(self, context_texts, original_prompt):
        context = "\n---\n".join(context_texts)
        return f"Relevant knowledge:\n{context}\n\n{original_prompt}"



def prepare_conversations(df: pd.DataFrame, task_name: str, retriever: RAGRetriever):
    conversations = []
    question_ids = []
    
    for _, row in df.iterrows():
        question = row["Question"]
        question_id = row["Question ID"]
        if "AttackSeq-Procedure" in task_name:
            answer_choices = {"A": "Yes", "B": "No"}
        else:
            answer_choices = {}
            for answer_choice in ["A", "B", "C", "D"]:
                answer_choices[answer_choice] = row[answer_choice]
        answer_choices_string = "\n".join([f"{choice}: {answer}" for choice, answer in answer_choices.items()])
        rag_prompt = retriever.retrieve(question)
        original_prompt = f"Question: {question}\nAnswer Choices: {answer_choices_string}"
        user_prompt = retriever.build_prompt(rag_prompt, original_prompt)
        conversations.append([
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }]
        )
        question_ids.append(question_id)
    return conversations, question_ids

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "meta-llama/Llama-3.3-70B-Instruct"
    # model_name = "Qwen/Qwen2.5-72B-Instruct"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "Qwen/QwQ-32B-Preview"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    # model_name = "THUDM/glm-4-9b-chat"
    
    embedding_model_name = "BAAI/bge-m3"
    # embedding_model_name = "basel/ATTACK-BERT"
    # embedding_model_name = "text-embedding-3-small" # OpenAI

    retriever = RAGRetriever(str(Path(__file__).parent.parent.parent / "mitre_kb" / "mitre.json"), model_name = embedding_model_name)

    dataset_dir = Path(__file__).parent.parent.parent / 'dataset'
    llm = LLM(model=model_name, gpu_memory_utilization=0.8, enforce_eager=True, trust_remote_code=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0, max_tokens=2048, top_p=1)
    for csv_file in dataset_dir.glob("*.csv"):
        task_name = csv_file.stem
        df = pd.read_csv(csv_file)
        conversations, question_ids = prepare_conversations(df, task_name, retriever)
        outputs = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)
        model_fp = model_name.split("/")[-1]
        embedding_model_fp = embedding_model_name.split("/")[-1]
        outputs_dir = Path(__file__).parent / 'outputs' / 'rag' / model_fp / embedding_model_fp
        write_dir = outputs_dir / csv_file.stem
        write_dir.mkdir(parents=True, exist_ok=True)
        for question_id, output in zip(question_ids, outputs):
             with open(write_dir / f"{task_name}-{question_id}.txt", "w") as fp:
                response = output.outputs[0].text
                fp.write(response)

if __name__ == "__main__":
    main()

