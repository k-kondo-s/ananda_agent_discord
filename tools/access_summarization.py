# %%
import requests
from bs4 import BeautifulSoup
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain


class AccessSummarization():
    def __init__(self) -> None:
        pass

    def get_text(self, url):
        """url にアクセスして、中のテキストを取得する"""
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        body_text = soup.get_text()
        return body_text

    def summarize(self, body_text):
        """GPT で要約する"""
        llm = ChatOpenAI(
            model_name='gpt-4',
            temperature=0,
            max_tokens=2000,
        )
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=2000,
            chunk_overlap=20,
            length_function=len,
        )
        docs = text_splitter.create_documents([body_text])
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        r = chain.run(docs)
        return r

    def translate(self, summarized_text):
        """日本語に翻訳する"""
        template = """
        以下の文章を日本語訳してください

        {input_text}
        """
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            temperature=0,
            max_tokens=2000,
        )
        chain_translate = LLMChain(llm=llm, prompt=PromptTemplate(
            input_variables=["input_text"],
            template=template)
        )
        r = chain_translate.run(summarized_text)
        return r

    def run(self, url):
        """url にアクセスして、中のテキストを取得し、GPT で要約し、日本語に翻訳する"""
        # url にアクセスして、中のテキストを取得する
        print(f'\nAccessing {url} ...')
        body_text = self.get_text(url)

        # GPT で要約する
        print(f'Summarizing ... {body_text}')
        summarized_text = self.summarize(body_text)

        # 日本語に翻訳する
        print(f'Translating ... {summarized_text}')
        tlanslated_text = self.translate(summarized_text)

        print(f'Done!: {tlanslated_text}')
        return tlanslated_text


if __name__ == '__main__':
    asum = AccessSummarization()
    url_text = 'https://note.com/aisaki180507/n/nb6880f333ae3'
    url_text = 'https://note.com/kondokenji/n/n9b4cc9c052ef?from=notice'
    res = asum.run(url_text)
    print(res)
