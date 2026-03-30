import re

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 本地嵌入模型路径：用于把文本转换成向量，供相似度检索使用。
EMBEDDING_MODEL_PATH = "/data/models/BAAI/bge-small-en-v1.5"
# 这是一篇待分析的中文文章。
# 后续会把它按自然段切开，写入向量库，再通过 RAG 的方式提取核心观点和证据。
ARTICLE_TEXT = """在中华文化国际传播过程中，汉字作为中华文化的核心载体与精神密码，始终扮演着基础性角色：没有汉字的有效传播，中华文化的思想内核、价值理念与人文底蕴便失去了具象化的表达载体，国际社会对中华文化的理解也将流于表面、难以深入。汉字不仅是一套用于交流的符号系统，更是镌刻着华夏文明基因的活化石，其传播过程本质上就是中华文化层层渗透、双向互鉴的过程，是中华文化国际传播不可或缺的根基与前提。

汉字的独特性决定了其作为中华文化国际传播基础的不可替代性。与世界上多数表音文字不同，汉字是形、音、义三位一体的表意文字，构造遵循“六书”规律，每一个汉字都承载着特定的文化内涵与历史记忆，成为传递中华文化精神的最小单元。这种形义共生的特质，让汉字超越了单纯的交际工具，成为文化的直接载体。“社”字从“示”从“土”，凝固着古代土地崇拜的集体记忆，彰显着中国人“敬天法地”的生存智慧；“礼”“乐”二字的构形，折射出儒家礼乐文明的制度编码，传递着“礼序乾坤、乐和天地”的价值追求；“和”字从甲骨文到现代的语义延伸，承载着“和而不同”“美美与共”的处世哲学，成为中华文化的核心精神符号。相比之下，表音文字仅能实现语音的传递，难以直接承载文化内涵，若要传递深层文化，需借助额外的解释与阐释，而汉字则将文化内涵融入自身构造，实现了“字即文化”的传播效能，这正是其作为中华文化国际传播基础的核心优势所在。

汉字的对外传播始终是中华文化走向世界的核心路径。早在秦汉时期，随着丝绸之路的开辟，汉字便随着贸易往来、使节互访开始向外传播，成为中华文化对外辐射的重要媒介，这一时期也成为汉字国际传播的初始阶段。到了唐代，国力强盛、文化繁荣，统治者的开放包容促成了“万国来朝”的盛景，长安成为当时世界文化交流的中心，“唐话”与汉字被周边国家广泛学习，日本、朝鲜半岛、越南等国纷纷引入汉字作为官方文字，借鉴汉字创制本国文字，形成了以汉字为核心的“汉字文化圈”。日本的平假名、片假名源于汉字偏旁，朝鲜半岛的谚文、越南的喃字均以汉字为基础创制，这些文字不仅承载着当地的语言需求，更传递着源自中华文化的伦理观念、礼仪规范与生活智慧。明清时期，随着海外贸易的拓展与华人移民的迁徙，汉字进一步传播至东南亚、美洲等地，成为华人社群维系文化认同、传递中华文化的重要纽带，此时汉字一度成为东亚地区的国际通用语言，推动中华文化实现了更大范围的传播。可以说，一部汉字的对外传播史，就是一部中华文化与世界各国文明交流互鉴的历史，汉字始终是中华文化国际传播的核心载体与基础支撑。

当代汉字传播的多元化实践，进一步夯实了其作为中华文化国际传播基础的地位，让中华文化以更鲜活、更接地气的方式走向世界。如今，汉字传播已摆脱传统的官方推动模式，呈现出“自下而上”的鲜明特征，其动力源于各国民众对文化多样性的天然向往与对中华文化的主动认同。在韩国，地铁玻璃上贴着唐诗宋词，首尔市的历史街区逐渐恢复汉字路牌标注，正在开发“汉字—韩文”对照数据库，支持青少年追溯汉字的演变轨迹；在日本，医疗、科技等领域的汉字术语不断创新，动漫产业中“萌”“燃”等新词的流行，体现了汉字对当代生活的适应性调整……这些现象并非刻意的文化输出，而是汉字作为表意文字的独特优势与当地文化需求自然融合的结果，也印证了汉字传播在当代的强大生命力。

汉字传播作为中华文化国际传播的基础，其核心价值在于实现了中华文化“形”与“神”的双重传播，既传递了语言符号，更传递了背后的精神内核与价值理念。在跨文化交流中，汉字的传播从来不是孤立的符号传递，而是伴随着中华文化的深度渗透：学习者认读“仁”字时，不仅掌握一个汉字，更能理解儒家“仁者爱人”的伦理思想；书写“道”字时，不仅学会一种笔画，更能接触到道家“道法自然”的哲学智慧；解读“家”字时，不仅认识一个词汇，更能体会中国人“家国同构”的价值追求。这种“以字载道”的传播方式，让中华文化的核心内涵能够通过具体可感的汉字符号得以传递，避免了文化传播的空洞化与表面化。

当然，以汉字传播为基础推动中华文化国际传播，仍面临诸多挑战。一方面，汉字的象形性与复杂性给海外学习者带来一定难度，部分学习者仅停留在汉字的工具性层面，难以深入理解其背后的文化内涵；另一方面，不同文明的文化差异可能导致汉字所承载的文化理念出现传播偏差，影响文化传播的效果。另外，语言服务出口行业仍存在小微企业占比高、产业链不完善等问题，也在一定程度上制约了汉字传播的规模化与专业化。应对这些挑战，需要进一步强化汉字传播的基础性作用，推动汉字传播与中华文化国际传播深度融合。一是完善顶层设计，加强政策支持，推动中文纳入各国国民教育体系。二是创新传播方式，依托数字技术打造多元化传播载体，开发更多适配海外学习者的汉字学习资源，通过VR、大数据等技术实现精准传播，化解汉字学习的难度。三是强化人才培养，搭建“产学研”协同发展平台，培养兼具汉字功底与跨文化沟通能力的复合型人才，推动汉字传播向专业化、精细化发展。四是坚持双向互鉴，尊重各国文化差异，在汉字传播中注重本土化适配，避免单向输出，实现“各美其美，美美与共”的传播格局。

汉字是提升中国文化软实力和国际影响力的载体和力量源泉。从古代丝绸之路的汉字传扬，到当代全球范围内的汉字学习热潮；从传统的书法篆刻，到数字时代的汉字创新，汉字始终以“润物无声”的方式，将中华文化的智慧与魅力传递给世界。展望“十五五”，我们应更加重视汉字传播的基础性作用，不断创新传播路径、完善传播体系，让汉字成为中华文化走向世界的文化密钥，推动中华文化在与世界各国文明的交流互鉴中，绽放出更加璀璨的光芒，为构建人类命运共同体注入深厚的文化力量。"""


def build_embedding_model() -> HuggingFaceEmbeddings:
    # 加载本地嵌入模型。
    # 这里仍然保留 RAG 的“检索”部分：先将文本转向量，再做相似度匹配。
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "local_files_only": True,
        },
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(embedding_model: HuggingFaceEmbeddings) -> FAISS:
    # 将长文章按自然段切分。
    # 对“提取核心观点”这类任务来说，按段落建库比随机定长切分更容易保留论证结构。
    paragraphs = [paragraph.strip() for paragraph in ARTICLE_TEXT.split("\n\n") if paragraph.strip()]

    # 给每个段落编号，后面输出证据时可以直接定位“第几段”。
    texts = paragraphs
    metadatas = [
        {
            "source": f"第{index}段",
            "paragraph_index": index,
        }
        for index, _paragraph in enumerate(paragraphs, start=1)
    ]

    # 直接根据文本构建 FAISS 向量库。
    # 这一步会先做向量化，再把向量写入内存中的索引结构。
    return FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)


def split_sentences(text: str) -> list[str]:
    # 按中文常见句号、问号、叹号、分号切句。
    fragments = re.split(r"(?<=[。！？；])", text)
    return [fragment.strip() for fragment in fragments if fragment.strip()]


def score_texts_by_query(
    embedding_model: HuggingFaceEmbeddings,
    texts: list[str],
    query: str,
) -> list[tuple[float, str]]:
    # 先把候选文本和问题都转成向量。
    # 由于前面已做归一化，这里直接用点积近似余弦相似度。
    query_vector = embedding_model.embed_query(query)
    text_vectors = embedding_model.embed_documents(texts)

    scored_texts = []
    for text, vector in zip(texts, text_vectors):
        score = sum(query_value * text_value for query_value, text_value in zip(query_vector, vector))
        scored_texts.append((score, text))

    scored_texts.sort(key=lambda item: item[0], reverse=True)
    return scored_texts


def extract_core_viewpoint(
    embedding_model: HuggingFaceEmbeddings,
    evidence_documents,
    question: str,
) -> list[tuple[str, str]]:
    # 从已检索到的证据段落中继续切句，并找出与“核心观点”问题最相关的句子。
    sentence_candidates: list[tuple[str, str]] = []
    for document in evidence_documents:
        source = document.metadata.get("source", "unknown")
        for sentence in split_sentences(document.page_content):
            if len(sentence) >= 15:
                sentence_candidates.append((source, sentence))

    if not sentence_candidates:
        return []

    scored_sentences = score_texts_by_query(
        embedding_model=embedding_model,
        texts=[sentence for _source, sentence in sentence_candidates],
        query=question,
    )

    selected_results: list[tuple[str, str]] = []
    used_sentences: set[str] = set()
    for _score, selected_sentence in scored_sentences:
        if selected_sentence in used_sentences:
            continue
        for source, sentence in sentence_candidates:
            if sentence == selected_sentence:
                selected_results.append((source, sentence))
                used_sentences.add(sentence)
                break
        if len(selected_results) == 2:
            break

    return selected_results


def main() -> None:
    # 第一步：加载嵌入模型。
    embedding_model = build_embedding_model()
    # 第二步：构建文章向量库。
    db = build_vectorstore(embedding_model)

    # 第三步：设置检索问题。
    question = "这篇文章的核心观点是什么？请用中文概括，并说明哪些段落能够证明这个观点。"

    # 第四步：先检索与“核心观点”问题最相关的段落。
    evidence_documents = db.similarity_search(question, k=3)
    extracted_viewpoints = extract_core_viewpoint(
        embedding_model=embedding_model,
        evidence_documents=evidence_documents,
        question=question,
    )

    # 将提取出的关键句拼接为“核心观点”。
    core_viewpoint = "；".join(sentence for _source, sentence in extracted_viewpoints)

    print("问题：", question)
    print("提取出的核心观点：", core_viewpoint)
    print("证据段落：")
    for document in evidence_documents:
        print(f"- {document.metadata.get('source', 'unknown')}: {document.page_content}")
    print("观点证据句：")
    for source, sentence in extracted_viewpoints:
        print(f"- {source}: {sentence}")


if __name__ == "__main__":
    # 脚本入口。
    main()
