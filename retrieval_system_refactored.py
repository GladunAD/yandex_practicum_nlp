import time
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm


class TextEmbedder:
    def __init__(self, model_name, device):
        self.device = device
        self.max_length = 512
        
        print(f'Загружаем bi-encoder: {model_name}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Оптимизация памяти (OOM fix): загрузка модели в float16
        self.embedder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.embedder.eval()

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

    def encode(self, texts, batch_size=16, show_progress=True):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc='Кодирование', disable=not show_progress):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.embedder(**inputs)
                
            embs = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embs = F.normalize(embs, p=2, dim=1)
            all_embeddings.append(embs.float().cpu().numpy())

            # Оптимизация памяти (OOM fix): явная очистка кэша
            del inputs, outputs, embs
            torch.cuda.empty_cache()
            
        return np.concatenate(all_embeddings, axis=0)


class FAISSIndex:
    def __init__(self, dim=None):
        self.index = None
        self.article_ids = []
        if dim is not None:
            self.index = faiss.IndexFlatIP(dim)

    def build(self, embeddings, ids):
        self.article_ids = ids
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f'Индекс построен: {self.index.ntotal:,} векторов, размерность {dim}')

    def save(self, save_path):
        import json
        save_path = str(save_path)
        
        # Сохраняем векторный индекс
        faiss.write_index(self.index, save_path + '.faiss')
        
        # Сохраняем соответствие id
        with open(save_path + '_ids.json', 'w', encoding='utf-8') as f:
            json.dump(self.article_ids, f)
            
        print(f'Индекс сохранён в префикс: {save_path}')

    def load(self, load_path):
        import json
        load_path = str(load_path)
        
        self.index = faiss.read_index(load_path + '.faiss')
        with open(load_path + '_ids.json', 'r', encoding='utf-8') as f:
            self.article_ids = json.load(f)
            
        print(f'Индекс загружен: {self.index.ntotal:,} векторов')

    def search(self, query_emb, k=100):
        if self.index is None:
            raise ValueError("Индекс не построен или не загружен.")
        
        scores, indices = self.index.search(query_emb, k)
        ids = [self.article_ids[i] for i in indices[0]]
        return ids, scores[0].tolist()


class CrossEncoder:
    def __init__(self, model_name, device):
        self.device = device
        self.max_length = 8192
        
        print(f'Загружаем реранкер: {model_name}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Оптимизация памяти (OOM fix): загрузка модели в float16
        self.reranker = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.reranker.eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids('yes')
        self.token_false_id = self.tokenizer.convert_tokens_to_ids('no')

        prefix = (
            '<|im_start|>system\\nJudge whether the Document meets the requirements based on the Query '
            'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\\n'
            '<|im_start|>user\\n'
        )
        suffix = '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def _prepare_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ids in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ids + self.suffix_tokens
            
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors='pt', max_length=self.max_length
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def rerank(self, query, candidate_ids, id_to_abstract, k=10, batch_size=4):
        instruction = 'Given a scientific query, retrieve the most relevant scientific paper abstract'

        pairs = [
            f'<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {id_to_abstract.get(cid, "")}'
            for cid in candidate_ids
        ]

        scores = []
        # Обработка батчами
        for i in range(0, len(pairs), batch_size):
            inputs = self._prepare_inputs(pairs[i : i + batch_size])
            logits = self.reranker(**inputs).logits[:, -1, :]
            
            true_vec = logits[:, self.token_true_id]
            false_vec = logits[:, self.token_false_id]
            
            batch_scores = torch.stack([false_vec, true_vec], dim=1)
            batch_scores = F.log_softmax(batch_scores, dim=1)
            scores.extend(batch_scores[:, 1].exp().tolist())

            # Оптимизация памяти (OOM fix): явная очистка кэша
            del inputs, logits, true_vec, false_vec, batch_scores
            torch.cuda.empty_cache()

        # Сортируем результаты по убыванию score
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:k]], [s for _, s in ranked[:k]]


# --------------------------------------------------------------------------------------------------
# Пример инициализации пайплайна (для копирования в ноутбук):
# --------------------------------------------------------------------------------------------------
# embedder = TextEmbedder(model_name=EMBEDDER_NAME, device=DEVICE)
# reranker = CrossEncoder(model_name=RERANKER_NAME, device=DEVICE)
# vector_index = FAISSIndex()
# print('Система готова')
#
# # Пример построения индекса (вместо rs.build_index)
# article_ids = df_index['id'].tolist()
# texts = df_index['abstract'].tolist()
# embeddings = embedder.encode(texts, batch_size=16, show_progress=True)
# vector_index.build(embeddings, article_ids)
#
# # Пример инференса (вместо rs.search -> rs.rerank)
# query_emb = embedder.encode([query], show_progress=False)
# candidate_ids, scores = vector_index.search(query_emb, k=K_RETRIEVE)
# ranked_ids, scores = reranker.rerank(query, candidate_ids, id_to_abstract, k=10)
# --------------------------------------------------------------------------------------------------
