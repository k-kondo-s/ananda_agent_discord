# %%
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


class AskAboutArticle():
    def __init__(self) -> None:
        pass

    def run(self, input_text_list) -> str:
        """AskAboutArticle"""
        url, question = input_text_list
        print(f'\nURL: {url}, Question: {question}')

        # url にアクセスして、中のテキストを取得する
        print(f'\nAccessing {url} ...')
        loader = UnstructuredURLLoader([url])
        raw_documents = loader.load()

        # documents を分割する
        print(f'\nSplitting documents ...')
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=20,
            length_function=len,
        )
        documents = text_splitter.split_documents(raw_documents)

        # documents をベクトル化する
        print(f'\nVectorizing documents ...')
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        # ベクトル化した documents を検索するためのインデックスを作成する
        retriever = vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name='gpt-4'),
            chain_type='stuff',
            retriever=retriever,
            verbose=True,
            return_source_documents=True
        )

        # 質問に対する回答を取得する
        print(f'\nGetting answer ...')
        answer = qa(question)['result']
        print(f'Answer: {answer}')
        return answer


if __name__ == '__main__':
    askaboutarticle = AskAboutArticle()
    result = askaboutarticle.run(
        ['https://note.com/kondokenji/n/na33f23fff656', '作成者は誰？'])
    print(result)
