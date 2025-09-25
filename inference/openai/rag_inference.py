import json
import tiktoken
import asyncio
from dotenv import load_dotenv
import pandas as pd
import faiss
import time
import numpy as np
from openai import OpenAI
from pathlib import Path
from prompts.rag_prompt import system_prompt
from sentence_transformers import SentenceTransformer

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
                    entries.append(f"Tactic: {tactic['name']}\nTechnique: {tech['name']}\nSub-Technique: {subtech['name']}\n{subtech['description']}\n{subtech.get('detailed_description', '')}")
        return entries, entries

    def _encode(self, texts):
        if uses_openai:
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

def prepare_batch_request(row: pd.Series, task_name: str, model_name: str, embedding_model: str, retriever: RAGRetriever):
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

    if model_name == "o3-mini-2025-01-31":
        task = {
            "custom_id": f"rag_{embedding_model}_{task_name}_{question_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "max_completion_tokens": 2048,
                "messages": [
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
            }
        }
    else:
        task = {
            "custom_id": f"rag_{embedding_model}_{task_name}_{question_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "temperature": 0.0,
                "max_tokens": 2048,
                "messages": [
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
            }
        }
    return task


async def poll_batch(client: OpenAI, batch_job, json_file: Path, batch_output_dir: Path):
    """Asynchronously poll the batch job until it finishes, then download and save its output."""
    while True:
        # Retrieve the batch status
        batch_data = client.batches.retrieve(batch_job.id)
        batch_status = batch_data.status

        if batch_status in {"completed", "failed", "cancelled"}:
            print(f"Batch {json_file.name} finished, status: {batch_status}")
            if batch_status == "failed":
                print("info:", batch_data)


            break
        elif batch_status == "in_progress":
            total_requests = batch_data.request_counts.total
            completed_requests = batch_data.request_counts.completed
            print(f"Batch {json_file.name} in progress, {completed_requests}/{total_requests} requests completed")
        # Wait for 60 seconds before checking again
        await asyncio.sleep(60)

    # Once finished, retrieve the output file
    batch_data = client.batches.retrieve(batch_job.id)
    batch_output_file_id = batch_data.output_file_id
    batch_output = client.files.content(batch_output_file_id).content
    output_path = batch_output_dir / f"{json_file.stem}.jsonl"
    with open(output_path, "wb") as f:
        f.write(batch_output)
    print(f"Batch output saved to {output_path}")


async def submit_batch_requests(client: OpenAI, batch_input_dir: Path, batch_output_dir: Path):
    #Submit each batch file and asynchronously poll for the results.
    poll_tasks = []
    # Iterate over each batch input file
    for json_file in batch_input_dir.glob("*.jsonl"):
        # Submit the batch file for processing
        with open(json_file, "rb") as f:
            batch_file = client.files.create(
                file=f,
                purpose="batch"
            )
        print(f"Submitted batch file for processing: {batch_file.filename}")

        batch_file_id = batch_file.id
        batch_job = client.batches.create(
            input_file_id=batch_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Batch of RAG inference"
            }
        )

        # Create an asynchronous task to poll the batch job
        task = asyncio.create_task(poll_batch(client, batch_job, json_file, batch_output_dir))
        poll_tasks.append(task)

    if poll_tasks:
        await asyncio.gather(*poll_tasks)


TOKEN_LIMIT = 2000000
# TOKEN_LIMIT = 90000

def main():
    load_dotenv()
    client = OpenAI()
    # model_name = "gpt-4o-2024-11-20"
    # model_name = "gpt-4o-mini"
    model_name = "o3-mini-2025-01-31"
    embedding_model_name = "BAAI/bge-m3"
    # embedding_model_name = "basel/ATTACK-BERT"
    # embedding_model_name = "text-embedding-3-small" # OpenAI
    enc = tiktoken.encoding_for_model("gpt-4o")

    retriever = RAGRetriever(str(Path(__file__).parent.parent.parent / "mitre_kb" / "mitre.json"), model_name = embedding_model_name)

    rag_dataset_dir = Path(__file__).parent.parent.parent / 'dataset'
    batch_inputs_dir = Path(__file__).parent / 'batch_jobs' / 'rag' / 'batch_inputs' / model_name / embedding_model.split("/")[-1]
    batch_outputs_dir = Path(__file__).parent / 'batch_jobs' /  'rag' / 'batch_outputs' / model_name / embedding_model.split("/")[-1]
    batch_outputs_dir.mkdir(parents=True, exist_ok=True)
    batch_inputs_dir.mkdir(parents=True, exist_ok=True)

    # Process each CSV and generate batch input files
    for csv_file in rag_dataset_dir.glob('*.csv'):
        token_count = 0
        batch_count = 0
        task_name = csv_file.stem
        df = pd.read_csv(csv_file)
        for index, row in df.iterrows():
            task = prepare_batch_request(row, task_name, model_name, embedding_model, retriever)
            token_count += len(enc.encode(str(task)))
            if token_count >= TOKEN_LIMIT:
                batch_count += 1
                token_count = 0
            write_file = batch_inputs_dir / f"{task_name}_{batch_count}.jsonl"
            with open(batch_inputs_dir / f"{task_name}_{batch_count}.jsonl", "a") as f:
                f.write(json.dumps(task) + '\n')
    # Run the asynchronous submission and polling of batch jobs
    asyncio.run(submit_batch_requests(client, batch_inputs_dir, batch_outputs_dir))


if __name__ == "__main__":
    main()
