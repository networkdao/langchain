import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# RetrievalQA 负责把“检索器 + 大语言模型”串起来，形成一个完整的 RAG 问答流程。
from langchain.chains import RetrievalQA
# PromptTemplate 用来定义提示词模板，后面会把检索到的上下文和用户问题填进去。
from langchain_core.prompts import PromptTemplate
# HuggingFaceEmbeddings 负责把文本编码成向量；HuggingFacePipeline 负责把 transformers 的生成流水线包装成 LangChain 可调用对象。
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS


# 本地嵌入模型路径：用于把文本转换成向量，供相似度检索使用。
EMBEDDING_MODEL_PATH = "/data/models/BAAI/bge-small-en-v1.5"
# 本地大语言模型路径：用于根据检索到的上下文生成最终答案。
LLM_MODEL_PATH = "/data/models/Phi-3-mini-4k-instruct"


def build_llm() -> HuggingFacePipeline:
    # 加载 Phi-3 对应的 tokenizer。
    # local_files_only=True 表示只从本地读取，不去联网下载。
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    # 某些模型没有显式设置 pad_token。
    # 这里将 pad_token 设为 eos_token，避免生成时出现 padding 相关问题。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载本地生成模型。
    # 有 CUDA 时自动用 GPU；否则退回 CPU。
    # torch_dtype 根据设备自动选择，GPU 上用 float16 可以减少显存占用。
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    # 用 transformers.pipeline 封装成文本生成流水线。
    # max_new_tokens 控制最多生成多少新 token。
    # do_sample=False 表示关闭随机采样，尽量让演示结果更稳定、更可复现。
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
    )
    # 再包装成 LangChain 可识别的 LLM 对象，后续就能直接传给 RetrievalQA。
    return HuggingFacePipeline(pipeline=generator)


def build_vectorstore() -> FAISS:
    # 加载本地嵌入模型。
    # normalize_embeddings=True 可以让向量归一化，便于使用余弦相似度近似做检索。
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "local_files_only": True,
        },
        encode_kwargs={"normalize_embeddings": True},
    )

    # 这里准备一组演示文本，模拟企业财务相关知识库。
    # 真正项目里，这里通常来自 txt、pdf、数据库或网页切分后的文本片段。
    texts = [
        "The company generated 12.5 million dollars of income in 2024, mainly from subscription revenue.",
        "The company generated 9.8 million dollars of income in 2023, supported by consulting services and training.",
        "Operating costs increased in 2024 because the company hired more engineers and expanded its cloud infrastructure.",
        "Subscription revenue was the biggest driver of profit because renewal rates stayed above 92 percent.",
        "The board expects moderate growth in 2025, with a focus on enterprise customers and higher annual contracts.",
    ]

    # 给每条文本附带一个来源标签，便于最后把命中的来源打印出来。
    metadatas = [
        {"source": "finance_report_2024"},
        {"source": "finance_report_2023"},
        {"source": "cost_report_2024"},
        {"source": "subscription_summary"},
        {"source": "board_forecast_2025"},
    ]

    # 直接根据文本构建 FAISS 向量库。
    # 这一步会先做向量化，再把向量写入内存中的索引结构。
    return FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)


def build_prompt() -> PromptTemplate:
    # 这是给 Phi-3 的提示词模板。
    # 其中 {context} 会被替换成检索结果，{question} 会被替换成用户问题。
    # 这里要求模型“只根据检索内容回答”，是 RAG 中常见的约束方式。
    template = """<|system|>
You are a concise assistant. Answer only from the retrieved context. If the answer is not in the context, say you do not know.<|end|>
<|user|>
Relevant information:
{context}

Question: {question}<|end|>
<|assistant|>
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])


def main() -> None:
    # 第一步：准备生成模型。
    llm = build_llm()
    # 第二步：准备向量数据库。
    db = build_vectorstore()
    # 第三步：准备提示词模板。
    prompt = build_prompt()

    # 第四步：构建 RetrievalQA 链。
    # chain_type="stuff" 表示把检索到的多个文本片段直接拼接后塞进提示词里。
    # k=2 表示每次检索最相关的 2 条文本。
    # return_source_documents=True 表示把命中的原始文档也一起返回，便于演示和调试。
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False,
    )

    # 给一个演示问题，询问 2024 年收入是多少。
    question = "How much income was generated in 2024?"

    # invoke 会先检索，再把检索结果和问题一起交给大模型生成答案。
    result = rag.invoke({"query": question})

    # 打印问题、答案和命中的来源文本，方便你观察 RAG 的完整输出。
    print("Question:", question)
    print("Answer:", result["result"])
    print("Sources:")
    for document in result["source_documents"]:
        print("-", document.metadata.get("source", "unknown"), ":", document.page_content)


if __name__ == "__main__":
    # 脚本入口。
    main()
